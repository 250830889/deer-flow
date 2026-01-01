# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.prompts.planner_model import StepType

from .nodes import (
    background_investigation_node,
    coder_node,
    coordinator_node,
    human_feedback_node,
    planner_node,
    reporter_node,
    research_team_node,
    researcher_node,
)
from .types import State


def continue_to_running_research_team(state: State):
    """
    决定下一步应该执行哪个节点的条件路由函数
    
    该函数根据当前计划的状态决定流程应该转向哪个执行节点：
    - 如果没有计划或计划步骤，返回planner进行规划
    - 如果所有步骤都已完成，返回planner进行重新规划
    - 找到第一个未完成的步骤，根据其类型返回相应的执行节点
    
    Args:
        state (State): 当前工作流状态，包含current_plan等信息
        
    Returns:
        str: 下一个要执行的节点名称 ("planner", "researcher", "coder")
    """
    # 获取当前计划，如果没有计划或计划步骤，返回planner节点
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return "planner"

    # 检查所有步骤是否都已完成（execution_res不为空表示已完成）
    # 如果全部完成，返回planner节点进行下一轮规划
    if all(step.execution_res for step in current_plan.steps):
        return "planner"

    # 查找第一个未完成的步骤（execution_res为空的步骤）
    incomplete_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            incomplete_step = step
            break

    # 如果没有找到未完成的步骤（理论上不会发生，因为前面已检查过）
    # 返回planner节点
    if not incomplete_step:
        return "planner"

    # 根据未完成步骤的类型决定下一个执行节点
    if incomplete_step.step_type == StepType.RESEARCH:
        return "researcher"  # 研究类型步骤分配给researcher节点
    if incomplete_step.step_type == StepType.PROCESSING:
        return "coder"       # 处理类型步骤分配给coder节点
    return "planner"         # 其他类型返回planner节点


def _build_base_graph():
    """
    构建并返回包含所有节点和边的基础状态图
    
    该函数创建一个完整的AI研究工作流图，包含以下主要阶段：
    1. 协调阶段 (coordinator) - 入口节点，负责初始化工作流程
    2. 背景调查阶段 (background_investigator) - 收集背景信息
    3. 规划阶段 (planner) - 制定详细的执行计划
    4. 研究团队阶段 (research_team) - 协调研究工作，根据条件路由到不同执行节点
    5. 研究执行阶段 (researcher) - 执行具体的研究任务
    6. 代码执行阶段 (coder) - 执行数据处理和分析任务
    7. 报告生成阶段 (reporter) - 汇总结果并生成最终报告
    8. 人工反馈阶段 (human_feedback) - 处理人工干预和反馈
    
    Returns:
        StateGraph: 配置完整的LangGraph状态图对象，包含所有节点和连接关系
    """
    # 创建状态图构建器，使用State作为状态类型
    builder = StateGraph(State)
    
    # 设置工作流的起始点：从开始节点连接到协调器
    builder.add_edge(START, "coordinator")
    
    # 添加各个功能节点，每个节点对应一个特定的处理函数
    builder.add_node("coordinator", coordinator_node)                    # 协调器节点：工作流程入口
    builder.add_node("background_investigator", background_investigation_node)  # 背景调查节点：收集背景信息
    builder.add_node("planner", planner_node)                          # 规划器节点：制定执行计划
    builder.add_node("reporter", reporter_node)                        # 报告器节点：生成最终报告
    builder.add_node("research_team", research_team_node)              # 研究团队节点：协调研究工作
    builder.add_node("researcher", researcher_node)                    # 研究员节点：执行研究任务
    builder.add_node("coder", coder_node)                              # 程序员节点：执行数据处理
    builder.add_node("human_feedback", human_feedback_node)              # 人工反馈节点：处理人工干预
    
    # 设置固定的节点连接关系
    builder.add_edge("background_investigator", "planner")  # 背景调查完成后进入规划阶段
    
    # 设置条件路由：研究团队节点根据状态决定下一步
    # 使用continue_to_running_research_team函数进行动态路由决策
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,  # 条件路由函数
        ["planner", "researcher", "coder"],  # 可能的目标节点列表
    )
    
    # 设置工作流的终点：报告生成后结束流程
    builder.add_edge("reporter", END)
    
    return builder


def build_graph_with_memory():
    # """
    # 构建并返回带有记忆功能的智能体工作流图
    
    # 该函数创建一个具有持久化记忆能力的AI研究工作流图，主要特点：
    # - 使用MemorySaver提供内存级别的状态持久化
    # - 能够保存和恢复对话历史及执行状态
    # - 支持工作流的断点续传和状态恢复
    # - 当前使用内存存储，后续计划支持SQLite/PostgreSQL等持久化存储
    
    # 记忆功能的优势：
    # - 保持跨会话的上下文信息
    # - 支持工作流的中断和恢复
    # - 记录完整的执行历史轨迹
    # - 为多轮对话提供状态保持
    
    # Returns:
    #     CompiledGraph: 编译后的LangGraph图对象，配置了记忆检查点功能
        
    # Note:
    #     当前使用内存级别的记忆存储（MemorySaver），未来版本将支持
    #     SQLite和PostgreSQL等持久化数据库，以提供更可靠的状态保存
    # """
    # 使用 SQLite 持久化存储对话历史和工作流状态
    # SqliteSaver 提供基于文件的持久化检查点功能
    # 对话历史会保存到 checkpoints.db 文件中，服务器重启后仍然保留
    memory = SqliteSaver.from_conn_string("checkpoints.db")

    # 构建基础状态图（包含所有节点和连接关系）
    builder = _build_base_graph()
    
    # 编译图对象并配置记忆检查点
    # checkpointer参数启用状态保存和恢复功能
    return builder.compile(checkpointer=memory)


def build_graph():
    """构建并返回无记忆的代理工作流图
    
    该函数负责创建一个LangGraph工作流图实例，用于定义代理系统中各个节点之间的
    连接关系和数据流转路径。此函数构建的图不包含记忆功能，适用于每次执行都是
    独立场景的应用。
    
    Returns:
        CompiledGraph: 编译后的工作流图对象，可直接用于执行工作流
        
    Example:
        >>> graph = build_graph()
        >>> result = graph.invoke({"research_topic": "AI研究"})
        >>> print(result.get("plan"))"""
    # 构建基础状态图
    builder = _build_base_graph()  # 调用内部函数创建图的基本结构，包括节点和边
    # 编译并返回可执行的工作流图
    return builder.compile()  # 将图结构编译为可执行的工作流实例


graph = build_graph()