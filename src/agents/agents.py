# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from typing import List, Optional


from langgraph.prebuilt import create_react_agent

from src.agents.tool_interceptor import wrap_tools_with_interceptor
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prompts import apply_prompt_template

logger = logging.getLogger(__name__)


# Create agents using configured LLM types
# 使用配置的LLM类型创建代理
def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list,#✅必须是列表类型✅ 示例：[search_tool, calc_tool]、[] 
    prompt_template: str,
    pre_model_hook: callable = None,
    interrupt_before_tools: Optional[List[str]] = None,
):
    """Factory function to create agents with consistent configuration.
    工厂函数：创建具有统一配置的智能代理

    Args:
        agent_name: Name of the agent
        agent_name: 代理名称 - 用于标识和日志记录
        agent_type: Type of agent (researcher, coder, etc.)
        agent_type: 代理类型 - 决定使用哪种LLM配置（如researcher、coder等）
        tools: List of tools available to the agent
        tools: 工具列表 - 代理可以调用的工具集合
        prompt_template: Name of the prompt template to use
        prompt_template: 提示模板名称 - 用于生成代理的系统提示
        pre_model_hook: Optional hook to preprocess state before model invocation
        pre_model_hook: 预处理钩子（可选）- 在模型调用前对状态进行预处理
        interrupt_before_tools: Optional list of tool names to interrupt before execution
        interrupt_before_tools: 工具中断列表（可选）- 指定需要中断的工具名称

    Returns:
        A configured agent graph
        返回配置完成的ReAct代理图
    """
    # 记录代理创建开始信息
    logger.debug(
        f"Creating agent '{agent_name}' of type '{agent_type}' "
        f"with {len(tools)} tools and template '{prompt_template}'"
    )
    
    # Wrap tools with interrupt logic if specified
    # 工具处理：如果需要中断特定工具，则包装工具
    processed_tools = tools
    if interrupt_before_tools:
        # 记录需要中断的工具信息
        logger.info(
            f"Creating agent '{agent_name}' with tool-specific interrupts: {interrupt_before_tools}"
        )
        logger.debug(f"Wrapping {len(tools)} tools for agent '{agent_name}'")
        # 使用拦截器包装工具，实现执行前中断
        processed_tools = wrap_tools_with_interceptor(
            tools, interrupt_before_tools, agent_name
        )
        logger.debug(f"Agent '{agent_name}' tool wrapping completed")
    else:
        logger.debug(f"Agent '{agent_name}' has no interrupt-before-tools configured")

    # 验证代理类型是否存在配置映射中
    if agent_type not in AGENT_LLM_MAP:
        # 代理类型未找到，使用默认配置并记录警告
        logger.warning(
            f"Agent type '{agent_type}' not found in AGENT_LLM_MAP. "
            f"Falling back to default LLM type 'basic' for agent '{agent_name}'. "
            "This may indicate a configuration issue."
        )
    # 获取LLM类型，不存在则使用默认的"basic"
    llm_type = AGENT_LLM_MAP.get(agent_type, "basic")
    logger.debug(f"Agent '{agent_name}' using LLM type: {llm_type}")
    
    # 创建ReAct代理
    logger.debug(f"Creating ReAct agent '{agent_name}'")
    agent = create_react_agent(
        name=agent_name,
        model=get_llm_by_type(llm_type),  # 根据类型获取对应的LLM模型
        tools=processed_tools,  # 使用处理后的工具列表
        prompt=lambda state: apply_prompt_template(  # 动态生成提示
            prompt_template, state, locale=state.get("locale", "en-US")
        ),
        pre_model_hook=pre_model_hook,  # 预处理钩子函数
    )
    # 记录代理创建完成
    logger.info(f"Agent '{agent_name}' created successfully")
    
    return agent
