"""深度研究智能体的图状态定义与数据结构。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出模型
###################
class ConductResearch(BaseModel):
    """调用该工具对指定主题开展研究。"""
    research_topic: str = Field(
        description="需要研究的主题。应为单一主题，且描述需足够详细（至少一段文字）。",
    )

class ResearchComplete(BaseModel):
    """调用该工具表示研究已完成。"""

class Summary(BaseModel):
    """包含关键发现的研究摘要。"""

    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """用于向用户澄清问题的模型。"""

    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="向用户询问的、用于明确报告范围的问题。",
    )
    verification: str = Field(
        description="在用户提供必要信息后，确认即将开始研究的提示信息。",
    )

class ResearchQuestion(BaseModel):
    """用于指导研究的研究问题与大纲。"""

    research_brief: str = Field(
        description="用于指导整个研究过程的研究主题说明。",
    )


###################
# 状态定义
###################

def override_reducer(current_value, new_value):
    """用于覆盖状态中值的合并函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

class AgentInputState(MessagesState):
    """输入状态，仅包含消息。"""

class AgentState(MessagesState):
    """主智能体状态，包含消息与研究相关数据。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """管理研究任务的主管节点状态。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """执行具体研究的单个研究员状态。"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """单个研究员的输出状态。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []