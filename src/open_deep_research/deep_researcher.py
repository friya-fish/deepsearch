"""Main LangGraph implementation for the Deep Research agent."""
from pathlib import Path
import asyncio,json,os
from typing import Literal,Any
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.open_deep_research.configuration import (
    Configuration,
)
from src.open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from src.open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from src.open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
    local_search_async
)
#原先为了解决deepseek模型返回格式有问题
async def call_and_parse(model_instance: Any, messages: list, pydantic_model: type[BaseModel]) -> Any:
    json_instruction = "\n\nCRITICAL: Respond in valid JSON format. Do not include any pre-text or post-text, only return the JSON."

    # 调用模型
    response = await model_instance.ainvoke(messages + [HumanMessage(content=json_instruction)])

    # 清理 markdown 标记并解析
    content = response.content.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(content)
        return pydantic_model(**data)
    except Exception as e:
        raise ValueError(f"JSON 解析失败: {e}. 模型原始内容: {response.content}")

# 初始化可配置模型
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[
    Literal["write_research_brief", "__end__"]]:
    """分析用户消息，若研究范围不清晰则向用户询问澄清问题。

    该函数判断用户请求是否需要在开始研究前进行澄清。
    若澄清功能被关闭或无需澄清，则直接进入研究环节。

    参数：
        state: 包含用户消息的当前智能体状态
        config: 运行时配置，包含模型设置与偏好

    返回：
        要么以澄清问题结束，要么进入研究大纲撰写环节的指令
    """
    # 步骤1：检查配置中是否启用澄清功能
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 跳过澄清步骤，直接进入研究
        return Command(goto="write_research_brief")

    # 步骤2：准备用于结构化澄清分析的模型
    messages = state["messages"]
    # 使用local_search工具分析用户资料库的内容
    try:
        print("开始查询本地")
        document_summary  = await local_search_async(config)
    except Exception as e:
        # 如果读取本地目录或汇总失败，给一个容错的提示
        document_summary = f"警告：本地资料库检索过程中出现错误：{str(e)}"
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 步骤3：分析是否需要澄清
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        document_summary = document_summary,
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    #如果需要澄清
    if response.need_clarification:
        # 结束流程并向用户返回澄清问题
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=response.question)],
                "document_summary": document_summary
            }
        )
    else:
        # 进入研究环节并附带确认信息
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "document_summary": document_summary
            }
        )

async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转换为结构化研究大纲，并初始化主管节点。

    该函数分析用户消息，生成聚焦的研究大纲，用于指导研究主管。
    同时使用合适的提示词与指令初始化主管上下文。

    参数：
        state: 包含用户消息的当前智能体状态
        config: 运行时配置，包含模型设置

    返回：
        进入研究主管环节并完成上下文初始化的指令
    """
    # 步骤1：设置用于结构化输出的研究模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    # 配置用于结构化研究问题生成的模型
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 步骤2：从用户消息生成结构化研究大纲
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    #response = await call_and_parse(llm, [HumanMessage(content=prompt_content)], ResearchQuestion)
    # 步骤3：使用研究大纲和指令初始化主管
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """主导研究的主管节点，负责规划研究策略并分配任务给研究员。

    主管分析研究大纲，决定如何将研究拆解为可管理的任务。
    可使用思考工具进行策略规划，使用研究执行工具分配子任务，
    或在满意结果时使用研究完成工具结束流程。

    参数：
        state: 包含消息与研究上下文的当前主管状态
        config: 运行时配置，包含模型设置

    返回：
        进入主管工具执行环节的指令
    """
    # 步骤1：为主管模型配置可用工具
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 可用工具：研究分配、完成标记、策略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # 配置带工具、重试逻辑与模型设置的模型
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 步骤2：基于当前上下文生成主管回复
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    # 步骤3：更新状态并进入工具执行
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行主管调用的工具，包括研究分配与策略思考。

    该函数处理三类主管工具调用：
    1. think_tool - 策略反思，继续对话
    2. ConductResearch - 将研究任务分配给子研究员
    3. ResearchComplete - 标记研究阶段完成

    参数：
        state: 包含消息与迭代次数的当前主管状态
        config: 运行时配置，包含研究上限与模型设置

    返回：
        继续主管循环或结束研究阶段的指令
    """
    # 步骤1：提取当前状态并检查退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 定义研究阶段的退出条件
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )
    # 满足任一终止条件则退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # 步骤2：统一处理所有工具调用（思考工具与研究执行）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 处理思考工具调用（策略反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # 处理研究执行调用（任务分配）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # 限制并发研究单元以避免资源耗尽
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

            # 并行执行研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # 使用研究结果创建工具消息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research",
                                            "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # 对溢出的研究调用返回错误信息
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # 汇总所有研究结果的原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # 处理研究执行异常
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # 超出 token 限制或其他错误 - 结束研究阶段
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )

    # 步骤3：返回包含所有工具结果的指令
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )


# 主管子图构建
# 创建用于管理研究分配与协调的主管工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加研究管理相关的主管节点
supervisor_builder.add_node("supervisor", supervisor)  # 主管主逻辑
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具执行处理器

# 定义主管工作流边
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# 编译主管子图供主流程使用
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """针对特定主题执行聚焦研究的独立研究员。

    该研究员接收主管分配的具体研究主题，
    使用可用工具（搜索、思考工具、MCP 工具）收集全面信息。
    可在搜索之间使用思考工具进行策略规划。

    参数：
        state: 包含消息与主题上下文的当前研究员状态
        config: 运行时配置，包含模型设置与工具可用性

    返回：
        进入研究员工具执行环节的指令
    """
    # 步骤1：加载配置并校验工具可用性
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # 获取所有可用研究工具（搜索、MCP、思考工具）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )

    # 步骤2：为研究员模型配置工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 若有 MCP 上下文则准备系统提示
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    # 配置带工具、重试逻辑与设置的模型
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 步骤3：结合系统上下文生成研究员回复
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # 步骤4：更新状态并进入工具执行
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


# 工具执行安全辅助函数
async def execute_tool_safely(tool, args, config):
    """带异常处理的安全工具执行。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[
    Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具，包括搜索工具与策略思考。

    该函数处理各类研究员工具调用：
    1. think_tool - 策略反思，继续研究对话
    2. 搜索工具（tavily_search、web_search）- 信息收集
    3. MCP 工具 - 外部工具集成
    4. ResearchComplete - 标记单个研究任务完成

    参数：
        state: 包含消息与迭代次数的当前研究员状态
        config: 运行时配置，包含研究上限与工具设置

    返回：
        继续研究循环或进入研究压缩环节的指令
    """
    # 步骤1：提取当前状态并检查提前退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # 未调用任何工具（包括原生网页搜索）则提前退出
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
            openai_websearch_called(most_recent_message) or
            anthropic_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # 步骤2：处理其他工具调用（搜索、MCP 等）
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # 并行执行所有工具调用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # 从执行结果创建工具消息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # 步骤3：检查后置退出条件（处理工具后）
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # 结束研究并进入压缩环节
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # 使用工具结果继续研究循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """将研究发现压缩并合成为简洁、结构化的摘要。

    该函数接收研究员所有研究结果、工具输出与 AI 消息，
    提炼为清晰、全面的摘要，同时保留所有重要信息与结论。

    参数：
        state: 包含累计研究消息的当前研究员状态
        config: 运行时配置，包含压缩模型设置

    返回：
        包含压缩研究摘要与原始笔记的字典
    """
    # 步骤1：配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })

    # 步骤2：准备用于压缩的消息
    researcher_messages = state.get("researcher_messages", [])

    # 添加指令，从研究模式切换为压缩模式
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    # 步骤3：尝试压缩，针对 token 限制进行重试
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # 创建专注于压缩任务的系统提示
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            # 执行压缩
            response = await synthesizer_model.ainvoke(messages)

            # 从所有工具与 AI 消息中提取原始笔记
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            # 返回成功的压缩结果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }

        except Exception as e:
            synthesis_attempts += 1

            # 超出 token 限制时清理较早消息
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # 其他错误继续重试
            continue

    # 步骤4：所有尝试失败后返回错误结果
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }


# 研究员子图构建
# 创建针对特定主题执行聚焦研究的独立研究员工作流
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration
)

# 添加研究执行与压缩相关的研究员节点
researcher_builder.add_node("researcher", researcher)  # 研究员主逻辑
researcher_builder.add_node("researcher_tools", researcher_tools)  # 工具执行处理器
researcher_builder.add_node("compress_research", compress_research)  # 研究压缩

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")  # 研究员入口
researcher_builder.add_edge("compress_research", END)  # 压缩后退出

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终综合研究报告，并针对 token 限制实现重试逻辑。

    该函数接收所有收集到的研究发现，
    使用配置好的报告生成模型合成为结构良好、全面的最终报告。

    参数：
        state: 包含研究发现与上下文的智能体状态
        config: 运行时配置，包含模型设置与 API 密钥

    返回：
        包含最终报告与清空状态的字典
    """
    # 步骤1：提取研究发现并准备状态清理
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # 步骤2：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 步骤3：尝试生成报告，针对 token 限制进行渐进式截断重试
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # 创建包含全部研究上下文的综合提示
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            # 生成最终报告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            # 返回成功的报告结果
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            # 超出 token 限制时进行渐进式截断
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # 第一次重试：确定初始截断上限
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # 使用 4 倍 token 限制作为字符截断近似值
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续重试：每次减少 10%
                    findings_token_limit = int(findings_token_limit * 0.9)

                # 截断研究发现并重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # 非 token 限制错误：立即返回错误
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

    # 步骤4：所有重试耗尽后返回失败结果
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }


# 深度研究者主图构建
# 创建从用户输入到最终报告的完整深度研究工作流
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# 添加完整研究流程的主工作流节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)  # 用户澄清阶段
deep_researcher_builder.add_node("write_research_brief", write_research_brief)  # 研究规划阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)  # 研究执行阶段
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成阶段

# 定义主工作流顺序执行边
deep_researcher_builder.add_edge(START, "clarify_with_user")  # 流程入口
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")  # 研究转报告
deep_researcher_builder.add_edge("final_report_generation", END)  # 最终出口

# 编译完整深度研究者工作流
deep_researcher = deep_researcher_builder.compile()