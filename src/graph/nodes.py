# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
节点模块：定义AI研究工作流中的各种节点函数

本模块包含了AI研究工作流中的所有节点函数，这些节点构成了一个完整的
研究流程，从用户查询的澄清、背景调查、计划制定、执行研究到最终报告生成。

主要节点包括：
- coordinator_node: 协调器节点，处理用户交互和需求澄清
- background_investigation_node: 背景调查节点，进行初步研究
- planner_node: 规划器节点，制定详细研究计划
- human_feedback_node: 人工反馈节点，处理用户对计划的审核
- researcher_node: 研究员节点，执行研究任务
- coder_node: 编码器节点，执行代码分析任务
- reporter_node: 报告器节点，生成最终研究报告

每个节点都遵循统一的接口规范，接收State和RunnableConfig参数，
返回Command对象以控制工作流的流向。
"""

import json
import logging
import os
import time
from functools import partial
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command, interrupt

from src.agents import create_agent
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from src.observability import get_current_run_id, trace_store
from src.observability.utils import extract_response_usage, serialize_messages
from src.prompts.planner_model import Plan
from src.prompts.template import apply_prompt_template
from src.tools import (
    crawl_tool,
    get_retriever_tool,
    get_web_search_tool,
    python_repl_tool,
)
from src.tools.search import LoggedTavilySearch
from src.utils.context_manager import ContextManager, validate_message_content
from src.utils.json_utils import repair_json_output, sanitize_tool_response

from ..config import SELECTED_SEARCH_ENGINE, SearchEngine
from .types import State
from .utils import (
    build_clarified_topic_from_history,
    get_message_content,
    is_user_message,
    reconstruct_clarification_history,
)

# 配置日志记录器，用于跟踪节点执行过程
logger = logging.getLogger(__name__)


@tool
def handoff_after_clarification(
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
    research_topic: Annotated[
        str, "The clarified research topic based on all clarification rounds."
    ],
):
    """澄清完成后交接给规划器。
    
    该工具在澄清轮次完成后使用，将所有澄清历史传递给规划器进行分析。
    参数包括用户语言区域和基于所有澄清轮次的明确研究主题。
    """
    return


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """交接给规划器代理以制定计划。
    
    该工具不返回任何内容，仅作为LLM信号，表示需要交接给规划器代理。
    参数包括研究任务主题和用户语言区域。
    """
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


@tool
def handoff_after_clarification(
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
    research_topic: Annotated[
        str, "The clarified research topic based on all clarification rounds."
    ],
):
    """澄清完成后交接给规划器。
    
    该工具在澄清轮次完成后使用，将所有澄清历史传递给规划器进行分析。
    参数包括用户语言区域和基于所有澄清轮次的明确研究主题。
    """
    return


def needs_clarification(state: dict) -> bool:
    """
    根据当前状态检查是否需要澄清。
    
    这是一个集中化的逻辑，用于确定何时继续澄清过程。
    
    Args:
        state: 当前状态字典，包含澄清相关配置
        
    Returns:
        bool: 如果需要澄清返回True，否则返回False
    """
    # 检查是否启用了澄清功能
    if not state.get("enable_clarification", False):
        return False

    # 获取澄清相关状态
    clarification_rounds = state.get("clarification_rounds", 0)
    is_clarification_complete = state.get("is_clarification_complete", False)
    max_clarification_rounds = state.get("max_clarification_rounds", 3)

    # 需要澄清的条件：已启用 + 有澄清轮次 + 未完成 + 未超过最大轮次
    # 使用 <= 是因为在询问第N个问题后，仍需等待第N个回答
    return (
        clarification_rounds > 0
        and not is_clarification_complete
        and clarification_rounds <= max_clarification_rounds
    )


def preserve_state_meta_fields(state: State) -> dict:
    """
    提取应在状态转换过程中保留的元/配置字段。
    
    这些字段对于工作流连续性至关重要，应明确包含在所有Command.update字典中，
    以防止它们恢复为默认值。
    
    Args:
        state: 当前状态对象
        
    Returns:
        dict: 需要保留的元字段字典
    """
    return {
        "locale": state.get("locale", "en-US"),
        "research_topic": state.get("research_topic", ""),
        "clarified_research_topic": state.get("clarified_research_topic", ""),
        "clarification_history": state.get("clarification_history", []),
        "enable_clarification": state.get("enable_clarification", False),
        "max_clarification_rounds": state.get("max_clarification_rounds", 3),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "resources": state.get("resources", []),
    }


def validate_and_fix_plan(plan: dict, enforce_web_search: bool = False) -> dict:
    """
    验证并修复计划，确保其满足要求。
    
    此函数执行两个主要验证和修复任务：
    1. 修复缺失的step_type字段（解决Issue #650）
    2. 根据需要强制执行网络搜索要求
    
    Args:
        plan: 要验证的计划字典
        enforce_web_search: 如果为True，确保至少有一个步骤设置了need_search=true
        
    Returns:
        dict: 验证/修复后的计划字典
    """
    # 检查计划是否为字典类型
    if not isinstance(plan, dict):
        return plan

    # 获取计划中的步骤列表
    steps = plan.get("steps", [])

    # ============================================================
    # 第一部分：修复缺失的step_type字段（解决Issue #650）
    # ============================================================
    for idx, step in enumerate(steps):
        # 跳过非字典类型的步骤
        if not isinstance(step, dict):
            continue
        
        # 检查step_type是否缺失或为空
        if "step_type" not in step or not step.get("step_type"):
            # 根据need_search值推断step_type
            inferred_type = "research" if step.get("need_search", False) else "processing"
            step["step_type"] = inferred_type
            logger.info(
                f"Repaired missing step_type for step {idx} ({step.get('title', 'Untitled')}): "
                f"inferred as '{inferred_type}' based on need_search={step.get('need_search', False)}"
            )

    # ============================================================
    # 第二部分：强制执行网络搜索要求
    # ============================================================
# 假设有以下 steps 列表：
# steps = [    {"title": "步骤1",     "need_search": False},    {"title": "步骤2"},  # 没有     need_search 键    {"title": "步骤3",     "need_search": True},    {"title": "步骤4",     "need_search": False}]
# 执行顺序如下：
# 生成器表达式开始工作，检查第一个步骤：
# step.get("need_search", False) 返回 False
# any() 收到 False，继续检查下一个
# 检查第二个步骤：
# step.get("need_search", False) 返回默认值 False（因为键不存在）
# any() 收到 False，继续检查下一个
# 检查第三个步骤：
# step.get("need_search", False) 返回 True
# any() 收到 True，立即返回 True，不再检查剩余步骤
# 最终 has_search_step 被赋值为 True
# 性能优势
# 这种实现方式有几个性能优势：
# 惰性求值：生成器表达式不会一次性创建所有值的列表，节省内存
# 短路求值：any() 函数在找到第一个 True 值后立即返回，避免不必要的计算
# 高效遍历：对于大型列表，这种方式比先创建完整列表再检查更高效
# 等价写法对比
# 这行代码等价于以下几种写法，但性能和简洁度不同：



# # 写法1：使用列表推导式（内存效率较低）has_search_step = any([step.get("need_search", False) for step in steps])# 写法2：使用传统循环（更冗长）has_search_step = Falsefor step in steps:    if step.get("need_search",     False):        has_search_step = True        break# 写法3：使用filter和next（稍复杂）has_search_step = next(filter(lambda s: s.get("need_search", False), steps), None) is not None
# 原始写法结合了生成器表达式的内存效率和 any() 函数的短路求值特性，是最优雅和高效的实现方式。
    if enforce_web_search:
        # 检查是否有任何步骤设置了need_search=true
        has_search_step = any(step.get("need_search", False) for step in steps)

        if not has_search_step and steps:
            # 确保第一个研究步骤启用了网络搜索
            for idx, step in enumerate(steps):
                if step.get("step_type") == "research":
                    step["need_search"] = True
                    logger.info(f"在索引 {idx} 处的研究步骤上强制启用网络搜索")
                    break
            else:
                # 备选方案：如果没有研究步骤，将第一个步骤转换为启用了网络搜索的研究步骤
                # 这确保至少有一个步骤会按要求执行网络搜索
                steps[0]["step_type"] = "research"
                steps[0]["need_search"] = True
                logger.info(
                    "将第一个步骤转换为研究步骤并强制启用网络搜索"
                )
        elif not has_search_step and not steps:
            # 如果不存在步骤，则添加一个默认的研究步骤
            logger.warning("计划没有步骤。添加默认研究步骤。")
            plan["steps"] = [
                {
                    "need_search": True,
                    "title": "Initial Research",
                    "description": "Gather information about the topic",
                    "step_type": "research",
                }
            ]

    return plan


def background_investigation_node(state: State, config: RunnableConfig):
    """
    背景调查节点，负责对研究主题进行初步的背景信息收集
    
    这个节点是AI研究工作流中的第一步，用于收集与研究主题相关的背景信息。
    它会根据配置的搜索引擎（Tavily或其他）执行搜索，并处理不同格式的搜索结果。
    搜索结果会被格式化并存储在状态中，供后续的规划节点使用。
    
    Args:
        state: 当前工作流状态，包含研究主题和已澄清的研究主题
        config: 可运行配置对象，包含搜索结果数量限制等配置
        
    Returns:
        dict: 包含格式化后的背景调查结果的字典，键为"background_investigation_results"
              值为JSON字符串格式的搜索结果列表
              
    Note:
        - 优先使用澄清后的研究主题，如果不存在则使用原始研究主题
        - 支持多种搜索引擎，通过SELECTED_SEARCH_ENGINE配置选择
        - 对Tavily搜索引擎的特殊处理包括多种响应格式的兼容
        - 所有结果都使用JSON格式存储，确保中文字符的正确显示
    """
    # 记录节点开始运行的日志
    logger.info("background investigation node is running.")
    # 从配置中提取可配置参数
    configurable = Configuration.from_runnable_config(config)
    # 获取研究查询，优先使用澄清后的主题，否则使用原始主题
    query = state.get("clarified_research_topic") or state.get("research_topic")
    # 初始化背景调查结果列表
    background_investigation_results = []
    
    # 根据选择的搜索引擎执行不同的搜索逻辑
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        # 使用Tavily搜索引擎进行搜索
        searched_content = LoggedTavilySearch(
            max_results=configurable.max_search_results
        ).invoke(query)
        # 检查搜索内容是否为元组，如果是则需要解包
        if isinstance(searched_content, tuple):
            searched_content = searched_content[0]
        
        # 处理字符串JSON响应（来自修复后的Tavily工具的新格式）
        if isinstance(searched_content, str):
            try:
                # 尝试解析JSON字符串
                parsed = json.loads(searched_content)
                if isinstance(parsed, dict) and "error" in parsed:
                    # 处理错误响应
                    logger.error(f"Tavily search error: {parsed['error']}")
                    background_investigation_results = []
                elif isinstance(parsed, list):
                    # 处理列表格式的响应
                    background_investigation_results = [
                        f"## {elem.get('title', 'Untitled')}\n\n{elem.get('content', 'No content')}" 
                        for elem in parsed
                    ]
                else:
                    # 处理意外格式的响应
                    logger.error(f"Unexpected Tavily response format: {searched_content}")
                    background_investigation_results = []
            except json.JSONDecodeError:
                # 处理JSON解析错误
                logger.error(f"Failed to parse Tavily response as JSON: {searched_content}")
                background_investigation_results = []
        # 处理旧版列表格式
        elif isinstance(searched_content, list):
            background_investigation_results = [
                f"## {elem['title']}\n\n{elem['content']}" for elem in searched_content
            ]
            # 直接返回旧格式结果（注意：这里直接返回，不执行最后的JSON转换）
            return {
                "background_investigation_results": "\n\n".join(
                    background_investigation_results
                )
            }
        else:
            # 处理格式错误的响应
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
            background_investigation_results = []
    else:
        # 使用其他搜索引擎进行搜索
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)
    
    # 返回JSON格式的背景调查结果，确保中文字符正确显示
    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii=False
        )
    }


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """
    规划器节点，生成完整的研究计划。
    
    该节点负责根据用户查询和背景调查结果生成详细的研究计划。
    它支持澄清模式和背景调查模式，并能够根据配置选择不同的LLM模型。
    生成的计划将经过验证和修复，确保满足网络搜索等要求。
    
    Args:
        state: 当前工作流状态，包含研究主题、澄清历史等信息
        config: 可运行配置对象，包含LLM配置和其他参数
        
    Returns:
        Command[Literal["human_feedback", "reporter"]]: 
            包含状态更新和路由指令的命令对象，决定下一步工作流向：
            - "human_feedback": 转到人工反馈节点，等待用户审核计划
            - "reporter": 直接转到报告节点，当计划被认为已足够完善时
    """
    # 记录规划器开始运行，包括语言环境信息
    logger.info("Planner generating full plan with locale: %s", state.get("locale", "en-US"))
    # 从配置中提取可配置参数
    configurable = Configuration.from_runnable_config(config)
    # 获取当前计划迭代次数，如果不存在则初始化为0
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0

    # 澄清功能：使用澄清后的研究主题（完整历史）
    if state.get("enable_clarification", False) and state.get(
        "clarified_research_topic"
    ):
        # 修改状态以使用澄清后的研究主题，而非完整对话历史
        modified_state = state.copy()
        modified_state["messages"] = [
            {"role": "user", "content": state["clarified_research_topic"]}
        ]
        modified_state["research_topic"] = state["clarified_research_topic"]
        # 应用规划器提示模板，使用澄清后的主题
        messages = apply_prompt_template("planner", modified_state, configurable, state.get("locale", "en-US"))

        logger.info(
            f"Clarification mode: Using clarified research topic: {state['clarified_research_topic']}"
        )
    else:
        # 普通模式：使用完整对话历史
        messages = apply_prompt_template("planner", state, configurable, state.get("locale", "en-US"))

    # 如果启用了背景调查且有背景调查结果，则将其添加到消息中
    if state.get("enable_background_investigation") and state.get(
        "background_investigation_results"
    ):
        messages += [
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    + "\n"
                ),
            }
        ]

    # 根据配置选择合适的LLM模型
    if configurable.enable_deep_thinking:
        # 使用推理模型进行深度思考
        llm = get_llm_by_type("reasoning")
    elif AGENT_LLM_MAP["planner"] == "basic":
        # 使用基础模型，并配置结构化输出
        llm = get_llm_by_type("basic").with_structured_output(
            Plan,
            method="json_mode",
        )
    else:
        # 使用配置中指定的规划器模型
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # 如果计划迭代次数超过最大限制，则直接转到报告节点
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(
            update=preserve_state_meta_fields(state),
            goto="reporter"
        )

    # 调用LLM生成计划
    run_id = get_current_run_id()
    start_time = time.perf_counter()
    if run_id:
        trace_store.add_event(
            run_id,
            "agent_input",
            payload={
                "agent": "planner",
                "messages": serialize_messages(messages),
                "plan_iterations": plan_iterations,
            },
            agent="planner",
            node="planner",
        )

    full_response = ""
    token_usage = None
    if AGENT_LLM_MAP["planner"] == "basic" and not configurable.enable_deep_thinking:
        # 对于基础模型，使用invoke方法获取结构化响应
        response = llm.invoke(messages)
        full_response = response.model_dump_json(indent=4, exclude_none=True)
        token_usage = extract_response_usage(response)
    else:
        # 对于其他模型，使用流式响应
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content
    
    # 记录调试信息和规划器响应
    if run_id:
        duration_ms = (time.perf_counter() - start_time) * 1000
        trace_store.add_event(
            run_id,
            "agent_output",
            payload={
                "agent": "planner",
                "output": full_response,
            },
            agent="planner",
            node="planner",
            duration_ms=duration_ms,
            token_usage=token_usage,
        )

    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    # 尝试解析JSON格式的计划
    try:
        curr_plan = json.loads(repair_json_output(full_response))
    except json.JSONDecodeError:
        # 如果JSON解析失败，记录警告并根据迭代次数决定下一步
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            # 如果已有迭代记录，转到报告节点
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter"
            )
        else:
            # 如果是第一次失败，结束工作流
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__"
            )

    # 验证并修复计划，确保满足网络搜索要求
    if isinstance(curr_plan, dict):
        curr_plan = validate_and_fix_plan(curr_plan, configurable.enforce_web_search)

    # 检查计划是否包含足够的上下文
    if isinstance(curr_plan, dict) and curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        # 验证计划模型并直接转到报告节点
        new_plan = Plan.model_validate(curr_plan)
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
                **preserve_state_meta_fields(state),
            },
            goto="reporter",
        )
    
    # 如果计划需要进一步反馈，转到人工反馈节点
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
            **preserve_state_meta_fields(state),
        },
        goto="human_feedback",
    )


def human_feedback_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "research_team", "reporter", "__end__"]]:
    """人工反馈节点：处理用户对研究计划的审核和反馈。
    
    该节点是工作流中的关键决策点，允许用户审核由规划器生成的研究计划。
    用户可以选择接受计划、请求修改或提供其他反馈。该节点还处理计划验证
    和格式化，确保计划符合系统要求。
    
    Args:
        state (State): 当前工作流状态，包含当前计划、计划迭代次数等信息
        config (RunnableConfig): 可运行配置对象，包含运行时参数和设置
        
    Returns:
        Command[Literal["planner", "research_team", "reporter", "__end__"]]: 
            包含状态更新和路由指令的命令对象，决定下一步工作流向：
            - "planner": 返回规划器节点，重新生成或修改计划
            - "research_team": 转到研究团队节点，开始执行研究计划
            - "reporter": 转到报告节点，生成最终报告
            - "__end__": 结束工作流
            
    Raises:
        json.JSONDecodeError: 当计划JSON格式无效时抛出
        
    Note:
        - 支持自动接受计划模式（通过auto_accepted_plan标志控制）
        - 用户反馈格式必须为"[ACCEPTED]"或"[EDIT_PLAN]"
        - 计划迭代次数有限制，超过限制将终止工作流
        - 支持多语言处理，通过locale参数指定语言环境
    """
    # 获取当前计划
    current_plan = state.get("current_plan", "")
    # 检查计划是否自动接受
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    
    # 如果不是自动接受模式，需要获取用户反馈
    if not auto_accepted_plan:
        # 中断工作流，等待用户反馈
        feedback = interrupt("Please Review the Plan.")

        # 处理None或空反馈
        if not feedback:
            logger.warning(f"收到空或None反馈: {feedback}. 返回规划器生成新计划.")
            return Command(
                update=preserve_state_meta_fields(state),
                goto="planner"
            )

        # 标准化反馈字符串
        feedback_normalized = str(feedback).strip().upper()

        # 如果反馈不是接受计划，返回规划器节点
        if feedback_normalized.startswith("[EDIT_PLAN]"):
            logger.info(f"用户请求编辑计划: {feedback}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="feedback"),
                    ],
                    **preserve_state_meta_fields(state),
                },
                goto="planner",
            )
        elif feedback_normalized.startswith("[ACCEPTED]"):
            logger.info("计划已被用户接受.")
        else:
            logger.warning(f"不支持的反馈格式: {feedback}. 请使用'[ACCEPTED]'接受或'[EDIT_PLAN]'编辑.")
            return Command(
                update=preserve_state_meta_fields(state),
                goto="planner"
            )

    # 如果计划被接受，执行以下逻辑
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    goto = "research_team"
    try:
        # 修复JSON输出格式
        current_plan = repair_json_output(current_plan)
        # 增加计划迭代次数
        plan_iterations += 1
        # 解析计划
        new_plan = json.loads(current_plan)
        # 验证并修复计划，确保满足网络搜索要求
        configurable = Configuration.from_runnable_config(config)
        new_plan = validate_and_fix_plan(new_plan, configurable.enforce_web_search)
    except json.JSONDecodeError:
        logger.warning("规划器响应不是有效的JSON")
        if plan_iterations > 1:  # plan_iterations在此检查前已增加
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter"
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__"
            )

    # 构建更新字典，安全处理locale
    update_dict = {
        "current_plan": Plan.model_validate(new_plan),
        "plan_iterations": plan_iterations,
        **preserve_state_meta_fields(state),
    }
    
    # 只有当new_plan提供有效locale值时才覆盖，否则使用保留的locale
    if new_plan.get("locale"):
        update_dict["locale"] = new_plan["locale"]
    
    return Command(
        update=update_dict,
        goto=goto,
    )

def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "coordinator", "__end__"]]:
    """协调器节点，负责与用户沟通并处理澄清过程。
    
    该节点是工作流的入口点，根据配置决定是否启用澄清模式，
    在澄清模式下与用户进行多轮对话以明确研究需求，
    然后将澄清后的研究主题传递给规划器节点。
    
    Args:
        state: 当前工作流状态，包含用户消息、研究主题等信息
        config: 运行配置，包含模型参数、系统设置等
        
    Returns:
        Command: 包含状态更新和下一步路由指令的命令对象
    """
    logger.info("协调器正在对话.")
    configurable = Configuration.from_runnable_config(config)
    
    # 检查是否启用澄清模式
    enable_clarification = state.get("enable_clarification", False)
    initial_topic = state.get("research_topic", "")
    clarified_topic = initial_topic
    
    # ============================================================
    # 分支1: 禁用澄清模式 (传统模式)
    # ============================================================
    if not enable_clarification:
        # 使用普通提示，明确指示跳过澄清
        messages = apply_prompt_template("coordinator", state, locale=state.get("locale", "en-US"))
        messages.append(
            {
                "role": "system",
                "content": "关键: 澄清功能已禁用. 您必须立即调用handoff_to_planner工具，使用用户的原始查询. 不要提问或提及需要更多信息.",
            }
        )

        # 仅绑定handoff_to_planner工具
        tools = [handoff_to_planner]
        run_id = get_current_run_id()
        start_time = time.perf_counter()
        if run_id:
            trace_store.add_event(
                run_id,
                "agent_input",
                payload={
                    "agent": "coordinator",
                    "messages": serialize_messages(messages),
                    "mode": "direct_handoff",
                },
                agent="coordinator",
                node="coordinator",
            )
        response = (
            get_llm_by_type(AGENT_LLM_MAP["coordinator"])
            .bind_tools(tools)
            .invoke(messages)
        )
        if run_id:
            duration_ms = (time.perf_counter() - start_time) * 1000
            trace_store.add_event(
                run_id,
                "agent_output",
                payload={
                    "agent": "coordinator",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                },
                agent="coordinator",
                node="coordinator",
                duration_ms=duration_ms,
                token_usage=extract_response_usage(response),
            )

        goto = "__end__"
        locale = state.get("locale", "en-US")
        logger.info(f"协调器语言环境: {locale}")
        research_topic = state.get("research_topic", "")

        # 处理传统模式的工具调用
        if response.tool_calls:
            try:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    if tool_name == "handoff_to_planner":
                        logger.info("交接给规划器")
                        goto = "planner"

                        # 提取research_topic（如果提供）
                        if tool_args.get("research_topic"):
                            research_topic = tool_args.get("research_topic")
                        break

            except Exception as e:
                logger.error(f"处理工具调用时出错: {e}")
                goto = "planner"

    # ============================================================
    # 分支2: 启用澄清模式 (新功能)
    # ============================================================
    else:
        # 加载澄清状态
        clarification_rounds = state.get("clarification_rounds", 0)
        clarification_history = list(state.get("clarification_history", []) or [])
        clarification_history = [item for item in clarification_history if item]
        max_clarification_rounds = state.get("max_clarification_rounds", 3)

        # 准备协调器的消息
        state_messages = list(state.get("messages", []))
        messages = apply_prompt_template("coordinator", state, locale=state.get("locale", "en-US"))

        # 重建澄清历史
        clarification_history = reconstruct_clarification_history(
            state_messages, clarification_history, initial_topic
        )
        clarified_topic, clarification_history = build_clarified_topic_from_history(
            clarification_history
        )
        logger.debug("澄清历史已重建: %s", clarification_history)

        if clarification_history:
            initial_topic = clarification_history[0]
            latest_user_content = clarification_history[-1]
        else:
            latest_user_content = ""

        # 为第一轮添加澄清状态
        if clarification_rounds == 0:
            messages.append(
                {
                    "role": "system",
                    "content": "澄清模式已启用. 请遵循您指令中的'澄清流程'指南.",
                }
            )

        current_response = latest_user_content or "无响应"
        logger.info(
            "澄清轮次 %s/%s | 主题: %s | 当前用户响应: %s",
            clarification_rounds,
            max_clarification_rounds,
            clarified_topic or initial_topic,
            current_response,
        )

        clarification_context = f"""继续澄清 (轮次 {clarification_rounds}/{max_clarification_rounds}):
            用户最新响应: {current_response}
            询问剩余的缺失维度. 不要重复问题或开始新话题."""

        messages.append({"role": "system", "content": clarification_context})

        # 绑定两个澄清工具 - 让LLM选择合适的工具
        tools = [handoff_to_planner, handoff_after_clarification]
        run_id = get_current_run_id()
        start_time = time.perf_counter()
        if run_id:
            trace_store.add_event(
                run_id,
                "agent_input",
                payload={
                    "agent": "coordinator",
                    "messages": serialize_messages(messages),
                    "mode": "clarification",
                    "round": clarification_rounds,
                },
                agent="coordinator",
                node="coordinator",
            )

        # 检查是否已达到最大轮次
        if clarification_rounds >= max_clarification_rounds:
            # 达到最大轮次 - 通过添加系统指令强制交接
            logger.warning(
                f"达到最大澄清轮次 ({max_clarification_rounds}). 强制交接给规划器. 使用准备好的澄清主题: {clarified_topic}"
            )
            # 添加系统指令强制交接 - 让LLM选择正确的工具
            messages.append(
                {
                    "role": "system",
                    "content": f"达到最大轮次. 您必须调用handoff_after_clarification (不是handoff_to_planner)，根据用户语言使用适当的locale，research_topic='{clarified_topic}'. 不要再问任何问题.",
                }
            )

        response = (
            get_llm_by_type(AGENT_LLM_MAP["coordinator"])
            .bind_tools(tools)
            .invoke(messages)
        )
        if run_id:
            duration_ms = (time.perf_counter() - start_time) * 1000
            trace_store.add_event(
                run_id,
                "agent_output",
                payload={
                    "agent": "coordinator",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                },
                agent="coordinator",
                node="coordinator",
                duration_ms=duration_ms,
                token_usage=extract_response_usage(response),
            )
        logger.debug(f"当前状态消息: {state['messages']}")

        # 初始化响应处理变量
        goto = "__end__"
        locale = state.get("locale", "en-US")
        research_topic = (
            clarification_history[0]
            if clarification_history
            else state.get("research_topic", "")
        )
        if not clarified_topic:
            clarified_topic = research_topic

        # --- 处理LLM响应 ---
        # 没有工具调用 - LLM正在询问澄清问题
        if not response.tool_calls and response.content:
            # 检查是否已达到最大轮次 - 如果是，强制交接给规划器
            if clarification_rounds >= max_clarification_rounds:
                logger.warning(
                    f"达到最大澄清轮次 ({max_clarification_rounds}). "
                    "LLM未调用交接工具，强制交接给规划器."
                )
                goto = "planner"
                # 继续到最终部分而不是提前返回
            else:
                # 继续澄清过程
                clarification_rounds += 1
                # 不要将LLM响应添加到clarification_history - 只添加用户响应
                logger.info(
                    f"澄清响应: {clarification_rounds}/{max_clarification_rounds}: {response.content}"
                )

                # 将协调器的问题附加到消息中
                updated_messages = list(state_messages)
                if response.content:
                    updated_messages.append(
                        HumanMessage(content=response.content, name="coordinator")
                    )

                return Command(
                    update={
                        "messages": updated_messages,
                        "locale": locale,
                        "research_topic": research_topic,
                        "resources": configurable.resources,
                        "clarification_rounds": clarification_rounds,
                        "clarification_history": clarification_history,
                        "clarified_research_topic": clarified_topic,
                        "is_clarification_complete": False,
                        "goto": goto,
                        "__interrupt__": [("coordinator", response.content)],
                    },
                    goto=goto,
                )
        else:
            # LLM调用了工具（交接）或没有内容 - 澄清完成
            if response.tool_calls:
                logger.info(
                    f"在{clarification_rounds}轮后澄清完成. LLM调用了交接工具."
                )
            else:
                logger.warning("LLM响应没有内容也没有工具调用.")
            # goto将在最终部分基于工具调用设置

    # ============================================================
    # 最终: 构建并返回命令
    # ============================================================
    messages = list(state.get("messages", []) or [])
    if response.content:
        messages.append(HumanMessage(content=response.content, name="coordinator"))

    # 处理两个分支（传统和澄清）的工具调用
    if response.tool_calls:
        try:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                if tool_name in ["handoff_to_planner", "handoff_after_clarification"]:
                    logger.info("交接给规划器")
                    goto = "planner"

                    if not enable_clarification and tool_args.get("research_topic"):
                        research_topic = tool_args["research_topic"]

                    if enable_clarification:
                        logger.info(
                            "使用准备好的澄清主题: %s",
                            clarified_topic or research_topic,
                        )
                    else:
                        logger.info(
                            "使用研究主题进行交接: %s", research_topic
                        )
                    break

        except Exception as e:
            logger.error(f"处理工具调用时出错: {e}")
            goto = "planner"
    else:
        # 没有检测到工具调用 - 回退到规划器而不是结束
        logger.warning(
            "LLM没有调用任何工具. 这可能表明模型的工具调用存在问题. "
            "回退到规划器以确保研究继续进行."
        )
        # 记录完整响应以便调试
        logger.debug(f"协调器响应内容: {response.content}")
        logger.debug(f"协调器响应对象: {response}")
        # 回退到规划器以确保工作流继续
        goto = "planner"

    # 如果启用背景调查，应用background_investigation路由（统一逻辑）
    if goto == "planner" and state.get("enable_background_investigation"):
        goto = "background_investigator"

    # 为状态变量设置默认值（以防在传统模式下未定义）
    if not enable_clarification:
        clarification_rounds = 0
        clarification_history = []

    clarified_research_topic_value = clarified_topic or research_topic

    # clarified_research_topic: 包含所有澄清轮次的完整澄清主题
    return Command(
        update={
            "messages": messages,
            "locale": locale,
            "research_topic": research_topic,
            "clarified_research_topic": clarified_research_topic_value,
            "resources": configurable.resources,
            "clarification_rounds": clarification_rounds,
            "clarification_history": clarification_history,
            "is_clarification_complete": goto != "coordinator",
            "goto": goto,
        },
        goto=goto,
    )



def reporter_node(state: State, config: RunnableConfig):
    """
    报告节点 - 生成最终研究报告
    
    这个节点是研究工作流的最后一步，负责将所有研究步骤的结果整合成一份
    完整、结构化的研究报告。它会根据收集到的观察结果和研究发现，
    生成一份格式化的报告，包含关键点、概述、详细分析和引用。
    
    主要功能:
    1. 应用报告模板，确保报告格式一致
    2. 整合所有研究步骤的观察结果
    3. 使用上下文压缩处理大量信息
    4. 生成包含引用和表格的格式化报告
    
    Args:
        state: 当前工作流状态，包含计划、观察结果和研究主题
        config: 可运行配置对象，包含系统配置和报告样式
        
    Returns:
        dict: 包含最终报告内容的字典，键为"final_report"
        
    Note:
        - 报告包含关键点、概述、详细分析和引用部分
        - 支持使用Markdown表格展示数据和比较信息
        - 引用格式为链接引用格式，放在报告末尾
        - 会自动压缩长上下文以适应LLM的令牌限制
    """
    logger.info("报告器开始生成最终报告")
    
    # 从配置中提取可配置参数
    configurable = Configuration.from_runnable_config(config)
    current_plan = state.get("current_plan")
    
    # 准备基础输入，包含研究任务的主题和描述
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    
    # 应用报告模板，确保报告格式一致
    invoke_messages = apply_prompt_template("reporter", input_, configurable, input_.get("locale", "en-US"))
    observations = state.get("observations", [])

    # 添加关于新报告格式、引用风格和表格使用的提醒
    invoke_messages.append(
        HumanMessage(
            content="重要提示：根据提示中的格式构建报告。请记住包含：\n\n1. 关键点 - 最重要的发现列表\n2. 概述 - 主题的简要介绍\n3. 详细分析 - 按逻辑部分组织\n4. 调查说明（可选）- 用于更全面的报告\n5. 关键引用 - 在末尾列出所有参考文献\n\n对于引用，不要在文本中包含内联引用。而是将所有引用放在末尾的\"关键引用\"部分，使用格式：`- [来源标题](URL)`。在每个引用之间包含空行以提高可读性。\n\n优先使用MARKDOWN表格展示数据和比较。在展示比较数据、统计、特性或选项时使用表格。使用清晰的标题和对齐的列构建表格。表格格式示例：\n\n| 特性 | 描述 | 优点 | 缺点 |\n|---------|-------------|------|------|\n| 特性1 | 描述1 | 优点1 | 缺点1 |\n| 特性2 | 描述2 | 优点2 | 缺点2 |",
            name="system",
        )
    )

    # 将观察结果转换为消息格式
    observation_messages = []
    for observation in observations:
        observation_messages.append(
            HumanMessage(
                content=f"以下是研究任务的一些观察结果：\n\n{observation}",
                name="observation",
            )
        )

    # 上下文压缩，处理大量信息
    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP["reporter"])
    compressed_state = ContextManager(llm_token_limit).compress_messages(
        {"messages": observation_messages}
    )
    invoke_messages += compressed_state.get("messages", [])

    # 记录调试信息
    logger.debug(f"当前调用消息: {invoke_messages}")
    
    # 调用LLM生成报告
    run_id = get_current_run_id()
    start_time = time.perf_counter()
    if run_id:
        trace_store.add_event(
            run_id,
            "agent_input",
            payload={
                "agent": "reporter",
                "messages": serialize_messages(invoke_messages),
            },
            agent="reporter",
            node="reporter",
        )

    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    response_content = response.content

    if run_id:
        duration_ms = (time.perf_counter() - start_time) * 1000
        trace_store.add_event(
            run_id,
            "agent_output",
            payload={
                "agent": "reporter",
                "output": response_content,
            },
            agent="reporter",
            node="reporter",
            duration_ms=duration_ms,
            token_usage=extract_response_usage(response),
        )
    logger.info(f"报告器响应: {response_content}")

    # 返回最终报告
    return {"final_report": response_content}


def research_team_node(state: State):
    """
    研究团队节点 - 协调研究和编码代理的协作
    
    这个节点是研究工作流中的协调中心，负责管理和调度研究团队中的
    不同代理（研究代理和编码代理）来执行计划中的各个步骤。它会根据
    计划中的步骤类型，将任务分配给合适的代理，并跟踪执行进度。
    
    主要功能:
    1. 分析当前计划，确定下一步需要执行的步骤
    2. 根据步骤类型（研究或编码）选择合适的代理
    3. 协调代理之间的工作流程，确保按顺序执行步骤
    4. 跟踪整体研究进度，决定何时生成最终报告
    
    Args:
        state: 当前工作流状态，包含计划、观察结果和执行进度
        
    Returns:
        目前函数尚未实现完整，返回None
        
    Note:
        - 这是一个关键的控制节点，决定了整个研究流程的执行路径
        - 需要与researcher_node和coder_node紧密配合
        - 未来实现将包含更复杂的任务调度和错误处理逻辑
    """
    logger.info("研究团队正在协作执行任务")
    logger.debug("进入research_team_node - 协调研究和编码代理")
    
    # TODO: 实现研究团队节点的完整逻辑
    # 1. 分析当前计划状态
    # 2. 确定下一步执行的步骤
    # 3. 根据步骤类型选择合适的代理
    # 4. 调用代理并处理结果
    # 5. 更新状态并决定下一步路由
    
    pass


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """
    使用指定代理执行步骤的辅助函数
    
    这是AI研究工作流的核心执行引擎，负责智能调度和执行研究步骤。
    函数会自动查找当前计划中第一个未完成的步骤，准备执行上下文，
    调用相应的代理执行该步骤，并处理执行结果和错误情况。
    
    Args:
        state: 当前工作流状态，包含计划、观察结果等信息
        agent: 要调用的代理实例（如researcher或coder）
        agent_name: 代理名称字符串，用于日志记录和错误处理
        
    Returns:
        Command[Literal["research_team"]]: 返回一个命令对象，包含更新后的状态
        和路由到"research_team"的指令，使工作流继续执行下一步
        
    Raises:
        无显式抛出异常，所有错误都被捕获并转换为错误消息存储在状态中
        
    Note:
        - 函数会自动处理递归限制配置，可通过环境变量AGENT_RECURSION_LIMIT设置
        - 对于researcher代理，会添加特殊的引用格式要求和资源文件处理
        - 所有执行结果都会被清理和验证，确保格式一致性
        - 错误处理机制完善，会记录详细的错误信息和堆栈跟踪
    """
    # 记录调试信息：开始执行特定代理
    logger.debug(f"[_execute_agent_step] Starting execution for agent: {agent_name}")
    
    # 从状态中提取当前计划和观察结果
    current_plan = state.get("current_plan")
    plan_title = current_plan.title
    observations = state.get("observations", [])
    logger.debug(f"[_execute_agent_step] Plan title: {plan_title}, observations count: {len(observations)}")

    # 查找第一个未执行的步骤
    current_step = None
    completed_steps = []
    for idx, step in enumerate(current_plan.steps):
        # 检查步骤是否已执行（execution_res为空表示未执行）
        if not step.execution_res:
            current_step = step
            logger.debug(f"[_execute_agent_step] Found unexecuted step at index {idx}: {step.title}")
            break
        else:
            # 收集已完成的步骤，用于构建上下文
            completed_steps.append(step)

    # 如果没有找到未执行的步骤，返回到research_team
    if not current_step:
        logger.warning(f"[_execute_agent_step] No unexecuted step found in {len(current_plan.steps)} total steps")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="research_team"
        )

    # 记录当前执行的步骤信息
    logger.info(f"[_execute_agent_step] Executing step: {current_step.title}, agent: {agent_name}")
    logger.debug(f"[_execute_agent_step] Completed steps so far: {len(completed_steps)}")

    # 格式化已完成步骤的信息，作为代理执行的上下文
    completed_steps_info = ""
    if completed_steps:
        completed_steps_info = "# Completed Research Steps\n\n"
        for i, step in enumerate(completed_steps):
            completed_steps_info += f"## Completed Step {i + 1}: {step.title}\n\n"
            completed_steps_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

    # 准备代理输入消息，包含研究主题、已完成步骤和当前步骤信息
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# Research Topic\n\n{plan_title}\n\n{completed_steps_info}# Current Step\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
            )
        ]
    }

    # 为researcher代理添加引用提醒和资源文件信息
    if agent_name == "researcher":
        # 如果状态中有资源文件，添加资源文件信息
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        # 添加引用格式要求，确保研究输出格式一致
        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )

    # 设置代理递归限制，防止无限递归
    default_recursion_limit = 25
    try:
        # 尝试从环境变量获取递归限制值
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)

        # 验证递归限制值是否为正数
        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        # 处理环境变量值无效的情况
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    # 记录代理输入信息，用于调试
    logger.info(f"Agent input: {agent_input}")
    
    # 在调用代理前验证消息内容
    try:
        validated_messages = validate_message_content(agent_input["messages"])
        agent_input["messages"] = validated_messages
    except Exception as validation_error:
        logger.error(f"Error validating agent input messages: {validation_error}")

    run_id = get_current_run_id()
    start_time = time.perf_counter()
    if run_id:
        trace_store.add_event(
            run_id,
            "agent_input",
            payload={
                "agent": agent_name,
                "step_title": current_step.title,
                "step_type": current_step.step_type,
                "messages": serialize_messages(agent_input.get("messages", [])),
            },
            agent=agent_name,
            node=agent_name,
            step=current_step.title,
        )
    
    # 调用代理执行步骤
    try:
        result = await agent.ainvoke(
            input=agent_input, config={"recursion_limit": recursion_limit}
        )
    except Exception as e:
        # 处理代理执行错误
        import traceback

        # 获取详细的错误堆栈信息
        error_traceback = traceback.format_exc()
        error_message = f"Error executing {agent_name} agent for step '{current_step.title}': {str(e)}"
        logger.exception(error_message)
        logger.error(f"Full traceback:\n{error_traceback}")
        
        # 针对内容相关错误的增强诊断
        if "Field required" in str(e) and "content" in str(e):
            logger.error(f"Message content validation error detected")
            for i, msg in enumerate(agent_input.get('messages', [])):
                logger.error(f"Message {i}: type={type(msg).__name__}, "
                            f"has_content={hasattr(msg, 'content')}, "
                            f"content_type={type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}, "
                            f"content_len={len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0}")

        # 创建详细的错误消息
        detailed_error = f"[ERROR] {agent_name.capitalize()} Agent Error\n\nStep: {current_step.title}\n\nError Details:\n{str(e)}\n\nPlease check the logs for more information."
        current_step.execution_res = detailed_error

        if run_id:
            duration_ms = (time.perf_counter() - start_time) * 1000
            trace_store.add_event(
                run_id,
                "agent_error",
                payload={
                    "agent": agent_name,
                    "step_title": current_step.title,
                    "error": str(e),
                },
                agent=agent_name,
                node=agent_name,
                step=current_step.title,
                duration_ms=duration_ms,
            )

        # 返回包含错误信息的命令
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=detailed_error,
                        name=agent_name,
                    )
                ],
                "observations": observations + [detailed_error],
                **preserve_state_meta_fields(state),
            },
            goto="research_team",
        )

    # 处理代理执行结果
    response_content = result["messages"][-1].content
    
    # 清理响应内容，移除多余标记并在需要时截断
    response_content = sanitize_tool_response(str(response_content))
    
    # 记录代理的完整响应，用于调试
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # 更新步骤的执行结果
    current_step.execution_res = response_content
    # 记录步骤执行完成的成功日志
    logger.info(f"Step '{current_step.title}' execution completed by {agent_name}")

    if run_id:
        duration_ms = (time.perf_counter() - start_time) * 1000
        trace_store.add_event(
            run_id,
            "agent_output",
            payload={
                "agent": agent_name,
                "step_title": current_step.title,
                "output": response_content,
            },
            agent=agent_name,
            node=agent_name,
            step=current_step.title,
            duration_ms=duration_ms,
        )

    # 返回包含更新状态的命令，路由到research_team
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": observations + [response_content],
            **preserve_state_meta_fields(state),
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """
    设置具有适当工具的代理并执行步骤的辅助函数
    
    这个函数处理researcher_node和coder_node的通用逻辑，负责：
    1. 根据代理类型配置MCP服务器和工具
    2. 创建具有适当工具的代理或使用默认代理
    3. 在当前步骤上执行代理
    
    该函数是代理工作流中的关键组件，实现了动态工具加载和代理配置，
    支持通过MCP（Model Context Protocol）服务器扩展代理能力。
    
    Args:
        state: 当前工作流状态，包含计划、观察结果等信息
        config: 可运行配置对象，包含系统配置和MCP设置
        agent_type: 代理类型字符串（"researcher"或"coder"）
        default_tools: 要添加到代理的默认工具列表
        
    Returns:
        Command[Literal["research_team"]]: 返回一个命令对象，包含更新后的状态
        和路由到"research_team"的指令，使工作流继续执行下一步
        
    Note:
        - 函数会根据配置动态加载MCP服务器提供的工具
        - 如果没有配置MCP服务器，则使用默认工具集
        - 所有代理都会配置令牌限制和消息压缩机制
        - 支持在工具执行前中断，用于调试和控制
    """
    # 从运行配置中提取配置信息
    configurable = Configuration.from_runnable_config(config)
    # 初始化MCP服务器和启用工具的字典
    mcp_servers = {}
    enabled_tools = {}

    # 为当前代理类型提取MCP服务器配置
    if configurable.mcp_settings:
        # 遍历所有配置的MCP服务器
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            # 检查服务器是否启用了工具且当前代理类型在允许列表中
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                # 提取服务器连接配置（仅保留必要的连接参数）
                mcp_servers[server_name] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env", "headers")
                }
                # 记录启用的工具及其所属服务器
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_name

    # 如果有可用的MCP工具，创建并执行带有MCP工具的代理
    if mcp_servers:
        # 创建多服务器MCP客户端，连接到所有配置的服务器
        client = MultiServerMCPClient(mcp_servers)
        # 复制默认工具列表，避免修改原始列表
        loaded_tools = default_tools[:]
        # 从所有MCP服务器获取可用工具
        all_tools = await client.get_tools()
        # 筛选并添加启用的工具
        for tool in all_tools:
            if tool.name in enabled_tools:
                # 为工具添加来源服务器信息，增强工具描述
                tool.description = (
                    f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                )
                loaded_tools.append(tool)

        # 获取代理类型对应的LLM令牌限制
        llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_type])
        # 创建消息压缩的预处理钩子，用于处理长上下文
        pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)
        # 创建带有MCP工具的代理实例
        agent = create_agent(
            agent_type,
            agent_type,
            loaded_tools,
            agent_type,
            pre_model_hook,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )
        # 执行代理步骤
        return await _execute_agent_step(state, agent, agent_type)
    else:
        # 如果没有配置MCP服务器，使用默认工具
        # 获取代理类型对应的LLM令牌限制
        llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_type])
        # 创建消息压缩的预处理钩子，用于处理长上下文
        pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)
        # 创建带有默认工具的代理实例
        agent = create_agent(
            agent_type,
            agent_type,
            default_tools,
            agent_type,
            pre_model_hook,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )
        # 执行代理步骤
        return await _execute_agent_step(state, agent, agent_type)


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """
    研究节点 - 负责执行研究相关的任务
    
    这个节点是研究团队中的核心组件，专门负责处理需要信息收集、
    网络搜索和资源检索的研究任务。它使用多种工具来获取和分析信息。
    
    主要功能:
    1. 配置研究专用工具集，包括网络搜索和网页爬取工具
    2. 根据可用资源动态添加检索工具
    3. 调用通用代理执行逻辑来执行研究步骤
    
    Args:
        state: 当前工作流状态，包含计划、观察结果和资源信息
        config: 可运行配置对象，包含系统配置和搜索参数
        
    Returns:
        Command[Literal["research_team"]]: 返回一个命令对象，包含研究步骤的执行结果
        和路由到"research_team"的指令，使工作流继续执行下一步
        
    Note:
        - 研究节点会优先使用检索工具访问已有资源
        - 网络搜索工具支持配置最大搜索结果数量
        - 所有研究活动都会记录详细日志用于调试
    """
    logger.info("研究节点开始执行研究任务")
    logger.debug(f"[researcher_node] 启动研究代理")
    
    # 从配置中提取可配置参数
    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"[researcher_node] 最大搜索结果数: {configurable.max_search_results}")
    
    # 创建基础工具集：网络搜索和网页爬取工具
    tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
    
    # 尝试获取检索工具，用于访问已有资源
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        logger.debug(f"[researcher_node] 将检索工具添加到工具列表")
        tools.insert(0, retriever_tool)
    
    # 记录工具配置信息
    logger.info(f"[researcher_node] 研究工具总数: {len(tools)}")
    logger.debug(f"[researcher_node] 研究工具列表: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]}")
    
    # 调用通用代理执行逻辑，传入研究专用工具集
    return await _setup_and_execute_agent_step(
        state,
        config,
        "researcher",
        tools,
    )


async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """
    编码节点 - 负责执行代码相关的任务
    
    这个节点是研究团队中处理代码分析、生成和执行的核心组件。
    它专门负责需要编程能力和代码执行的任务，使用Python REPL工具
    来执行代码并获取结果。
    
    主要功能:
    1. 配置编码专用工具集，包括Python REPL执行环境
    2. 调用通用代理执行逻辑来执行编码步骤
    3. 处理代码执行结果并返回给工作流
    
    Args:
        state: 当前工作流状态，包含计划、观察结果和代码相关任务
        config: 可运行配置对象，包含系统配置和执行参数
        
    Returns:
        Command[Literal["research_team"]]: 返回一个命令对象，包含编码步骤的执行结果
        和路由到"research_team"的指令，使工作流继续执行下一步
        
    Note:
        - 编码节点使用Python REPL工具执行代码
        - 支持复杂的代码分析和生成任务
        - 所有代码执行活动都会记录详细日志用于调试
    """
    logger.info("编码节点开始执行代码任务")
    logger.debug(f"[coder_node] 启动编码代理，配置python_repl_tool")
    
    # 调用通用代理执行逻辑，传入编码专用工具集（Python REPL工具）
    return await _setup_and_execute_agent_step(
        state,
        config,
        "coder",
        [python_repl_tool],
    )
