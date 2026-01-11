# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
DeerFlow工作流模块

此模块提供了DeerFlow智能体系统的核心工作流功能，包括：
1. 工作流图的构建和初始化
2. 异步工作流执行函数
3. 调试日志配置
4. 澄清流程处理

主要功能：
- 构建并初始化DeerFlow工作流图
- 提供异步执行智能体工作流的接口
- 支持多轮澄清对话
- 支持调试模式，提供详细的执行日志

使用示例：
```python
# 基本用法
result = await run_agent_workflow_async(
    user_input="请帮我分析最新的AI技术趋势",
    debug=True,
    max_plan_iterations=2
)

# 启用澄清功能
result = await run_agent_workflow_async(
    user_input="帮我写个程序",
    enable_clarification=True,
    max_clarification_rounds=3
)
```
"""

# 标准库导入
import logging

# 项目内部导入
from src.config.configuration import get_recursion_limit  # 获取递归限制配置
from src.graph import build_graph  # 构建工作流图
from src.graph.utils import build_clarified_topic_from_history  # 从历史记录构建澄清主题

# 配置日志记录器，设置默认日志级别为INFO，并指定日志格式
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别为INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 日志格式：时间-模块名-级别-消息
)


def enable_debug_logging():
    """
    启用调试级别日志记录
    
    此函数将src模块的日志级别设置为DEBUG，以便在执行过程中获取更详细的信息。
    调试日志对于排查问题和理解系统内部工作流程非常有用。
    
    使用场景：
    - 开发阶段需要详细了解系统执行流程
    - 生产环境中排查复杂问题
    - 性能分析和优化
    
    注意事项：
    - 调试日志会产生大量输出，可能影响性能
    - 生产环境中应谨慎使用，避免日志文件过大
    - 建议在完成调试后恢复到INFO级别
    """
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
graph = build_graph()


async def run_agent_workflow_async(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
    enable_clarification: bool | None = None,
    max_clarification_rounds: int | None = None,
    initial_state: dict | None = None,
):
    """
    异步运行智能体工作流
    
    此函数是DeerFlow系统的核心入口点，负责异步执行智能体工作流以处理用户输入。
    工作流包括计划制定、执行、结果生成等阶段，并支持多轮澄清对话以获取更准确的需求。
    
    参数:
        user_input (str): 用户的查询或请求内容，不能为空
        debug (bool): 是否启用调试模式，启用后将显示详细执行日志，默认为False
        max_plan_iterations (int): 最大计划迭代次数，用于复杂任务的多轮规划，默认为1
        max_step_num (int): 单个计划中的最大步骤数，限制执行复杂度，默认为3
        enable_background_investigation (bool): 是否在计划前执行网络搜索以增强上下文，默认为True
        enable_clarification (bool | None): 是否启用澄清功能
            - None: 使用State类的默认值（False）
            - True/False: 覆盖默认值
        max_clarification_rounds (int | None): 最大澄清轮次数，默认为3（当enable_clarification为True时）
        initial_state (dict | None): 初始状态字典，用于递归调用时的状态传递，默认为None
    
    返回:
        dict: 工作流完成后的最终状态字典，包含执行结果、消息历史等信息
    
    异常:
        ValueError: 当用户输入为空时抛出
    
    使用示例:
        ```python
        # 基本用法
        result = await run_agent_workflow_async(
            user_input="分析最新的AI技术趋势",
            debug=True,
            max_plan_iterations=2
        )
        
        # 启用澄清功能
        result = await run_agent_workflow_async(
            user_input="帮我写个程序",
            enable_clarification=True,
            max_clarification_rounds=3
        )
        ```
    
    注意事项:
        - 用户输入不能为空，否则将抛出ValueError异常
        - 启用调试模式会产生大量日志，可能影响性能
        - 澄清功能需要用户交互，不适合自动化场景
        - 递归调用时必须提供initial_state参数以保持上下文
    """
    # 验证用户输入不能为空
    if not user_input:
        raise ValueError("Input could not be empty")

    # 如果启用调试模式，则设置调试级别日志
    if debug:
        enable_debug_logging()

    # 记录工作流开始，包含用户输入信息
    logger.info(f"Starting async workflow with user input: {user_input}")

    # 使用提供的初始状态或创建新的初始状态
    if initial_state is None:
        # 创建新的初始状态字典
        initial_state = {
            # 运行时变量
            "messages": [{"role": "user", "content": user_input}],  # 消息历史，初始只包含用户输入
            "auto_accepted_plan": True,  # 自动接受计划，无需用户确认
            "enable_background_investigation": enable_background_investigation,  # 是否启用背景调查
        }
        # 设置研究主题为用户输入
        initial_state["research_topic"] = user_input
        initial_state["clarified_research_topic"] = user_input

        # 只有在明确提供时才设置澄清参数
        # 如果为None，将使用State类的默认值（enable_clarification=False）
        if enable_clarification is not None:
            initial_state["enable_clarification"] = enable_clarification

        if max_clarification_rounds is not None:
            initial_state["max_clarification_rounds"] = max_clarification_rounds

    # 配置工作流执行参数
    config = {
        "configurable": {
            "thread_id": "default",  # 线程ID，用于维护对话上下文
            "max_plan_iterations": max_plan_iterations,  # 最大计划迭代次数
            "max_step_num": max_step_num,  # 最大执行步数
            # MCP（Model Context Protocol）服务器配置
            "mcp_settings": {
                "servers": {
                    "mcp-github-trending": {
                        "transport": "stdio",  # 传输方式
                        "command": "uvx",  # 执行命令
                        "args": ["mcp-github-trending"],  # 命令参数
                        "enabled_tools": ["get_github_trending_repositories"],  # 启用的工具
                        "add_to_agents": ["researcher"],  # 应用到的代理
                    }
                }
            },
        },
        "recursion_limit": get_recursion_limit(default=100),  # 递归限制，防止无限循环
    }
    
    # 初始化消息计数器和最终状态
    last_message_cnt = 0
    final_state = None
    
    # 异步流式执行工作流
    async for s in graph.astream(
        input=initial_state, config=config, stream_mode="values"
    ):
        try:
            # 更新最终状态
            final_state = s
            
            # 处理包含消息的状态
            if isinstance(s, dict) and "messages" in s:
                # 如果消息数量没有增加，跳过处理
                if len(s["messages"]) <= last_message_cnt:
                    continue
                    
                # 更新消息计数并处理最新消息
                last_message_cnt = len(s["messages"])
                message = s["messages"][-1]
                
                # 根据消息类型进行不同处理
                if isinstance(message, tuple):
                    # 元组类型消息直接打印
                    print(message)
                else:
                    # 其他类型消息使用pretty_print方法格式化输出
                    message.pretty_print()
            else:
                # 非消息状态直接打印
                print(f"Output: {s}")
        except Exception as e:
            # 记录流输出处理错误
            logger.error(f"Error processing stream output: {e}")
            print(f"Error processing output: {str(e)}")

    # 检查是否需要澄清，使用集中式逻辑判断
    if final_state and isinstance(final_state, dict):
        from src.graph.nodes import needs_clarification

        # 如果需要澄清
        if needs_clarification(final_state):
            # 等待用户输入
            print()
            clarification_rounds = final_state.get("clarification_rounds", 0)  # 当前澄清轮次
            max_clarification_rounds = final_state.get("max_clarification_rounds", 3)  # 最大澄清轮次
            
            # 提示用户输入澄清信息
            user_response = input(
                f"Your response ({clarification_rounds}/{max_clarification_rounds}): "
            ).strip()

            # 如果用户输入为空，结束澄清流程
            if not user_response:
                logger.warning("Empty response, ending clarification")
                return final_state

            # 使用用户响应继续工作流
            current_state = final_state.copy()
            # 添加用户响应到消息历史
            current_state["messages"] = final_state["messages"] + [
                {"role": "user", "content": user_response}
            ]
            
            # 保留澄清相关的状态字段
            for key in (
                "clarification_history",
                "clarification_rounds",
                "clarified_research_topic",
                "research_topic",
                "locale",
                "enable_clarification",
                "max_clarification_rounds",
            ):
                if key in final_state:
                    current_state[key] = final_state[key]

            # 递归调用自身，继续处理澄清后的工作流
            return await run_agent_workflow_async(
                user_input=user_response,
                max_plan_iterations=max_plan_iterations,
                max_step_num=max_step_num,
                enable_background_investigation=enable_background_investigation,
                enable_clarification=enable_clarification,
                max_clarification_rounds=max_clarification_rounds,
                initial_state=current_state,
            )

    # 记录工作流成功完成
    logger.info("Async workflow completed successfully")


# 脚本入口点：当直接运行此文件时执行的代码
if __name__ == "__main__":
    import asyncio  # 导入异步IO库，用于运行异步函数

    # 定义主函数，封装异步工作流的调用
    async def main():
        """
        主函数：演示如何使用run_agent_workflow_async函数
        
        此函数展示了DeerFlow系统的基本使用方法，包括：
        1. 创建用户输入
        2. 调用异步工作流函数
        3. 处理可能的异常
        
        注意：这是一个示例函数，实际使用时可以根据需要修改参数
        """
        try:
            # 示例用户查询，可以根据需要修改
            user_query = "帮我分析一下最新的AI技术趋势"
            
            # 调用异步工作流函数处理用户查询
            # 这里使用默认参数，可以根据需要调整
            await run_agent_workflow_async(
                user_input=user_query,
                debug=True,  # 启用调试模式以查看详细日志
                max_plan_iterations=2,  # 允许最多2轮计划迭代
                max_step_num=5  # 每个计划最多5个步骤
            )
        except Exception as e:
            # 捕获并记录主函数中的异常
            logger.error(f"Error in main function: {e}")
            print(f"An error occurred: {str(e)}")

    # 运行主函数
    # asyncio.run()用于在同步环境中运行异步函数
    asyncio.run(main())
