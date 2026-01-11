# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


"""
工作流状态类型定义模块

本模块定义了DeerFlow项目中用于工作流状态管理的核心数据结构。
State类扩展了LangGraph的MessagesState，添加了研究工作流所需的各种字段，
包括研究主题、计划、观察结果、资源文件等。

主要组件:
- State: 扩展的工作流状态类，包含所有工作流节点共享的状态信息
"""

from dataclasses import field

from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.rag import Resource


class State(MessagesState):
    """
    工作流状态类 - 扩展MessagesState，添加工作流特定字段
    
    这个类是整个研究工作流的核心状态容器，负责在工作流各节点之间传递数据。
    它继承自LangGraph的MessagesState，添加了研究主题、计划、观察结果等字段。
    
    主要功能:
    1. 存储用户输入的研究主题和澄清后的主题
    2. 跟踪研究计划及其迭代次数
    3. 收集研究过程中的观察结果和资源文件
    4. 管理澄清功能的开关和历史记录
    5. 控制工作流的导航和流程
    
    Attributes:
        locale: 语言环境设置，默认为"en-US"
        research_topic: 原始研究主题
        clarified_research_topic: 经过澄清后的完整研究主题
        observations: 研究过程中收集的观察结果列表
        resources: 研究过程中使用的资源文件列表
        plan_iterations: 计划迭代次数
        current_plan: 当前研究计划，可以是Plan对象或字符串
        final_report: 最终生成的报告内容
        auto_accepted_plan: 是否自动接受计划标志
        enable_background_investigation: 是否启用背景调查
        background_investigation_results: 背景调查结果
        enable_clarification: 是否启用澄清功能
        clarification_rounds: 已进行的澄清轮次
        clarification_history: 澄清历史记录列表
        is_clarification_complete: 澄清是否完成标志
        max_clarification_rounds: 最大澄清轮次数
        goto: 工作流下一个节点名称
    """

    # 运行时变量
    locale: str = "en-US"  # 语言环境设置，用于本地化显示和提示
    research_topic: str = ""  # 用户输入的原始研究主题
    clarified_research_topic: str = (
        ""  # 完整/最终澄清后的研究主题，包含所有澄清轮次的结果
    )
    observations: list[str] = []  # 研究过程中收集的观察结果列表
    resources: list[Resource] = []  # 研究过程中使用的资源文件列表
    plan_iterations: int = 0  # 计划迭代次数，用于跟踪计划修改次数
    current_plan: Plan | str = None  # 当前研究计划，可以是Plan对象或字符串形式
    final_report: str = ""  # 最终生成的报告内容
    auto_accepted_plan: bool = False  # 是否自动接受计划标志，用于跳过人工确认
    enable_background_investigation: bool = True  # 是否启用背景调查功能
    background_investigation_results: str = None  # 背景调查结果内容

    # 澄清状态跟踪（默认禁用）
    enable_clarification: bool = (
        False  # 启用/禁用澄清功能（默认值：False）
    )
    clarification_rounds: int = 0  # 已进行的澄清轮次计数
    clarification_history: list[str] = field(default_factory=list)  # 澄清历史记录列表
    is_clarification_complete: bool = False  # 澄清是否完成标志
    max_clarification_rounds: int = (
        3  # 最大澄清轮次数（默认值：3，仅在enable_clarification=True时使用）
    )

    # 工作流控制
    goto: str = "planner"  # 默认下一个节点，用于工作流导航
