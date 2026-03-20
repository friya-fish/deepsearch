import json
from typing import Literal, Any
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from legacy.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from legacy.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from legacy.configuration import Configuration
from legacy.utils import (
    format_sections,
    get_config_value,
    get_search_params,
    select_and_execute_search,
    get_today_str
)


async def call_and_parse_json(llm: Any, messages: list, pydantic_model: type[BaseModel]) -> Any:
    json_instruction = "\n\nCRITICAL: Respond in valid JSON format. Do not include any pre-text or post-text, only return the JSON."

    # 构造请求
    response = await llm.ainvoke(messages + [HumanMessage(content=json_instruction)])

    # 清理回复内容 (兼容处理)
    content = response.content.replace("```json", "").replace("```", "").strip()

    # 解析 JSON
    try:
        data = json.loads(content)
        return pydantic_model(**data)
    except Exception as e:
        raise ValueError(f"解析 JSON 失败: {e}. 模型原始内容: {response.content}")


## Nodes --

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    topic = state["topic"]
    feedback_list = state.get("feedback_on_report_plan", [])
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    configurable = Configuration.from_runnable_config(config)
    report_structure = str(configurable.report_structure) if isinstance(configurable.report_structure,
                                                                        dict) else configurable.report_structure

    writer_model = init_chat_model(
        model=get_config_value(configurable.writer_model),
        model_provider=get_config_value(configurable.writer_provider),
        model_kwargs={**(get_config_value(configurable.writer_model_kwargs) or {}), "response_format": None}
    )

    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic, report_organization=report_structure, number_of_queries=configurable.number_of_queries,
        today=get_today_str()
    )

    results = await call_and_parse_json(writer_model, [SystemMessage(content=system_instructions_query),
                                                       HumanMessage(content="Generate search queries.")], Queries)

    source_str = await select_and_execute_search(get_config_value(configurable.search_api),
                                                 [q.search_query for q in results.queries],
                                                 get_search_params(get_config_value(configurable.search_api),
                                                                   configurable.search_api_config or {}))

    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure,
                                                                      context=source_str, feedback=feedback)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})
    planner_llm = init_chat_model(model=get_config_value(configurable.planner_model),
                                  model_provider=get_config_value(configurable.planner_provider),
                                  model_kwargs=planner_model_kwargs)

    report_sections = await call_and_parse_json(planner_llm, [SystemMessage(content=system_instructions_sections),
                                                              HumanMessage(content="Generate sections.")], Sections)

    return {"sections": report_sections.sections}


def human_feedback(state: ReportState, config: RunnableConfig) -> Command[
    Literal["generate_report_plan", "build_section_with_web_research"]]:
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(f"Section: {s.name}\nDescription: {s.description}\n" for s in sections)
    feedback = interrupt(f"Please provide feedback:\n\n{sections_str}")

    if feedback is True:
        return Command(
            goto=[Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) for s
                  in sections if s.research])
    elif isinstance(feedback, str):
        return Command(goto="generate_report_plan", update={"feedback_on_report_plan": [feedback]})
    else:
        raise TypeError("Invalid feedback type")


async def generate_queries(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(model=get_config_value(configurable.writer_model),
                                   model_provider=get_config_value(configurable.writer_provider))

    system_instructions = query_writer_instructions.format(topic=state["topic"],
                                                           section_topic=state["section"].description,
                                                           number_of_queries=configurable.number_of_queries,
                                                           today=get_today_str())
    queries = await call_and_parse_json(writer_model, [SystemMessage(content=system_instructions),
                                                       HumanMessage(content="Generate search queries.")], Queries)
    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    source_str = await select_and_execute_search(get_config_value(configurable.search_api),
                                                 [q.search_query for q in state["search_queries"]],
                                                 get_search_params(get_config_value(configurable.search_api),
                                                                   configurable.search_api_config or {}))
    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}


async def write_section(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(model=get_config_value(configurable.writer_model),
                                   model_provider=get_config_value(configurable.writer_provider))

    inputs = section_writer_inputs.format(topic=state["topic"], section_name=state["section"].name,
                                          section_topic=state["section"].description, context=state["source_str"],
                                          section_content=state["section"].content)
    content = await writer_model.ainvoke(
        [SystemMessage(content=section_writer_instructions), HumanMessage(content=inputs)])
    state["section"].content = content.content

    # 评分逻辑
    reflection_model = init_chat_model(model=get_config_value(configurable.planner_model),
                                       model_provider=get_config_value(configurable.planner_provider))
    feedback = await call_and_parse_json(reflection_model, [SystemMessage(
        content=section_grader_instructions.format(topic=state["topic"], section_topic=state["section"].description,
                                                   section=state["section"].content,
                                                   number_of_follow_up_queries=configurable.number_of_queries)),
                                                            HumanMessage(content="Grade the section.")], Feedback)

    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        update = {"completed_sections": [state["section"]]}
        if configurable.include_source_str: update["source_str"] = state["source_str"]
        return Command(update=update, goto=END)
    return Command(update={"search_queries": feedback.follow_up_queries, "section": state["section"]},
                   goto="search_web")


async def write_final_sections(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(model=get_config_value(configurable.writer_model),
                                   model_provider=get_config_value(configurable.writer_provider))
    system_instructions = final_section_writer_instructions.format(topic=state["topic"],
                                                                   section_name=state["section"].name,
                                                                   section_topic=state["section"].description,
                                                                   context=state["report_sections_from_research"])
    content = await writer_model.ainvoke(
        [SystemMessage(content=system_instructions), HumanMessage(content="Generate section.")])
    state["section"].content = content.content
    return {"completed_sections": [state["section"]]}


def gather_completed_sections(state: ReportState):
    return {"report_sections_from_research": format_sections(state["completed_sections"])}


def compile_final_report(state: ReportState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    sections = state["sections"]
    completed = {s.name: s.content for s in state["completed_sections"]}
    for s in sections: s.content = completed[s.name]
    final = "\n\n".join([s.content for s in sections])
    return {"final_report": final, "source_str": state["source_str"]} if configurable.include_source_str else {
        "final_report": final}


def initiate_final_section_writing(state: ReportState):
    return [Send("write_final_sections", {"topic": state["topic"], "section": s,
                                          "report_sections_from_research": state["report_sections_from_research"]}) for
            s in state["sections"] if not s.research]


section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()