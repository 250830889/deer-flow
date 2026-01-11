# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
DeerFlow项目入口脚本

本模块是DeerFlow多智能体协作系统的命令行入口点，提供两种运行模式：
1. 交互式模式：提供内置问题选择和自定义问题输入
2. 命令行模式：直接通过命令行参数处理查询

主要功能：
- 提供简洁的API接口调用多智能体工作流
- 支持多种运行参数配置（调试模式、计划迭代次数、步骤限制等）
- 支持中英文双语交互界面
- 提供内置问题库和自定义问题输入

使用示例：
    # 交互式模式
    python main.py --interactive
    
    # 命令行模式
    python main.py "什么是量子计算？"
    
    # 带参数的命令行模式
    python main.py "分析AI趋势" --debug --max_plan_iterations 2
"""

# 标准库导入
import argparse  # 命令行参数解析库，用于处理用户输入的运行参数
import asyncio  # 异步I/O库，用于执行异步工作流

# 第三方库导入
from InquirerPy import inquirer  # 交互式命令行界面库，用于创建用户友好的交互体验

# 项目内部导入
from src.config.questions import BUILT_IN_QUESTIONS, BUILT_IN_QUESTIONS_ZH_CN  # 内置问题库（英文和中文）
from src.workflow import run_agent_workflow_async  # 异步工作流执行函数


def ask(
    question,
    debug=False,
    max_plan_iterations=1,
    max_step_num=3,
    enable_background_investigation=True,
    enable_clarification=False,
    max_clarification_rounds=None,
):
    """运行智能体工作流以处理用户问题。

    这是DeerFlow项目的核心入口函数，用于启动多智能体协作系统来回答用户问题。
    该函数封装了异步工作流的同步调用，简化了用户交互接口。

    Args:
        question (str): 用户的查询或请求，可以是任何需要智能体系统处理的问题
        debug (bool, optional): 是否启用调试级别日志记录。默认为False。
            启用后会输出更详细的执行过程信息，便于问题排查
        max_plan_iterations (int, optional): 计划迭代的最大次数。默认为1。
            控制智能体系统对问题计划的优化次数，更多迭代可能产生更精细的执行计划
        max_step_num (int, optional): 计划中的最大步骤数。默认为3。
            限制单个计划包含的执行步骤数量，防止计划过于复杂
        enable_background_investigation (bool, optional): 是否在规划前执行网络搜索以增强上下文。默认为True。
            启用后系统会先进行背景信息收集，提高回答质量和准确性
        enable_clarification (bool, optional): 是否启用多轮澄清功能。默认为False。
            启用后系统会对模糊问题进行主动澄清，以获取更准确的需求理解
        max_clarification_rounds (int, optional): 澄清轮次的最大数量。默认为None，使用State默认值3。
            限制系统与用户的澄清交互次数，防止无限循环

    Returns:
        None: 函数不直接返回结果，而是通过异步工作流处理并输出最终答案

    Raises:
        可能抛出与异步执行相关的异常，具体取决于run_agent_workflow_async的实现

    Example:
        >>> # 基本用法
        >>> ask("什么是量子计算？")
        
        >>> # 启用调试和澄清功能
        >>> ask("分析一下最新的AI趋势", debug=True, enable_clarification=True)
        
        >>> # 自定义执行参数
        >>> ask("比较Python和JavaScript", max_plan_iterations=2, max_step_num=5)

    Note:
        该函数是同步接口，内部通过asyncio.run调用异步工作流。
        对于需要在现有异步事件循环中调用的情况，应直接使用run_agent_workflow_async函数。
    """
    # 使用asyncio.run将异步工作流包装为同步调用
    # 这使得函数可以从同步代码中直接调用，无需处理事件循环
    asyncio.run(
        run_agent_workflow_async(
            user_input=question,  # 用户输入的问题
            debug=debug,  # 调试标志
            max_plan_iterations=max_plan_iterations,  # 计划迭代次数限制
            max_step_num=max_step_num,  # 计划步骤数限制
            enable_background_investigation=enable_background_investigation,  # 背景调查开关
            enable_clarification=enable_clarification,  # 澄清功能开关
            max_clarification_rounds=max_clarification_rounds,  # 澄清轮次限制
        )
    )


def main(
    debug=False,
    max_plan_iterations=1,
    max_step_num=3,
    enable_background_investigation=True,
    enable_clarification=False,
    max_clarification_rounds=None,
):
    """
    交互式模式主函数
    
    提供交互式用户界面，允许用户选择语言、从内置问题库中选择问题或输入自定义问题。
    该函数处理用户交互逻辑，并将最终选择的问题传递给ask函数处理。
    
    Args:
        debug (bool, optional): 是否启用调试级别日志记录。默认为False。
            启用后会输出更详细的执行过程信息，便于问题排查
        max_plan_iterations (int, optional): 计划迭代的最大次数。默认为1。
            控制智能体系统对问题计划的优化次数
        max_step_num (int, optional): 计划中的最大步骤数。默认为3。
            限制单个计划包含的执行步骤数量
        enable_background_investigation (bool, optional): 是否在规划前执行网络搜索以增强上下文。默认为True。
            启用后系统会先进行背景信息收集，提高回答质量和准确性
        enable_clarification (bool, optional): 是否启用多轮澄清功能。默认为False。
            启用后系统会对模糊问题进行主动澄清
        max_clarification_rounds (int, optional): 澄清轮次的最大数量。默认为None，使用State默认值3。
            限制系统与用户的澄清交互次数
    
    Returns:
        None: 函数不返回值，通过ask函数处理用户问题并输出结果
    
    Raises:
        可能抛出与用户交互或工作流执行相关的异常
    
    Note:
        该函数使用InquirerPy库创建交互式命令行界面，提供友好的用户体验。
        用户界面支持中英文双语，根据用户选择的语言显示相应的提示信息。
    """
    # 第一步：选择界面语言
    language = inquirer.select(
        message="Select language / 选择语言:",
        choices=["English", "中文"],
    ).execute()

    # 根据选择的语言加载对应的问题库
    questions = (
        BUILT_IN_QUESTIONS if language == "English" else BUILT_IN_QUESTIONS_ZH_CN
    )
    # 设置自定义问题选项的文本
    ask_own_option = (
        "[Ask my own question]" if language == "English" else "[自定义问题]"
    )

    # 第二步：选择问题（从内置问题库或自定义）
    initial_question = inquirer.select(
        message=(
            "What do you want to know?" if language == "English" else "您想了解什么?"
        ),
        choices=[ask_own_option] + questions,
    ).execute()

    # 如果用户选择自定义问题，则提供文本输入框
    if initial_question == ask_own_option:
        initial_question = inquirer.text(
            message=(
                "What do you want to know?"
                if language == "English"
                else "您想了解什么?"
            ),
        ).execute()

    # 将所有参数传递给ask函数，执行实际的工作流
    ask(
        question=initial_question,
        debug=debug,
        max_plan_iterations=max_plan_iterations,
        max_step_num=max_step_num,
        enable_background_investigation=enable_background_investigation,
        enable_clarification=enable_clarification,
        max_clarification_rounds=max_clarification_rounds,
    )


if __name__ == "__main__":
    """
    脚本入口点
    
    当直接运行main.py文件时，此代码块将被执行。
    它负责解析命令行参数，并根据参数决定运行模式（交互式或命令行模式）。
    """
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Run the Deer")
    # 位置参数：查询内容，允许为空（空时将通过交互式输入）
    parser.add_argument("query", nargs="*", help="The query to process")
    # 可选参数：启用交互式模式
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with built-in questions",
    )
    # 可选参数：设置最大计划迭代次数
    parser.add_argument(
        "--max_plan_iterations",
        type=int,
        default=1,
        help="Maximum number of plan iterations (default: 1)",
    )
    # 可选参数：设置计划中最大步骤数
    parser.add_argument(
        "--max_step_num",
        type=int,
        default=3,
        help="Maximum number of steps in a plan (default: 3)",
    )
    # 可选参数：启用调试模式
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # 可选参数：禁用背景调查功能
    parser.add_argument(
        "--no-background-investigation",
        action="store_false",
        dest="enable_background_investigation",
        help="Disable background investigation before planning",
    )
    # 可选参数：启用澄清功能
    parser.add_argument(
        "--enable-clarification",
        action="store_true",
        dest="enable_clarification",
        help="Enable multi-turn clarification for vague questions (default: disabled)",
    )
    # 可选参数：设置最大澄清轮次
    parser.add_argument(
        "--max-clarification-rounds",
        type=int,
        dest="max_clarification_rounds",
        help="Maximum number of clarification rounds (default: 3)",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 根据参数选择运行模式
    if args.interactive:
        # 交互式模式：调用main函数处理用户交互
        main(
            debug=args.debug,
            max_plan_iterations=args.max_plan_iterations,
            max_step_num=args.max_step_num,
            enable_background_investigation=args.enable_background_investigation,
            enable_clarification=args.enable_clarification,
            max_clarification_rounds=args.max_clarification_rounds,
        )
    else:
        # 命令行模式：直接处理查询或提示用户输入
        if args.query:
            # 如果命令行提供了查询参数，则合并为单个字符串
            user_query = " ".join(args.query)
        else:
            # 如果没有提供查询参数，则提示用户输入
            # 循环直到用户提供非空输入
            while True:
                user_query = input("Enter your query: ")
                if user_query is not None and user_query != "":
                    break

        # 使用提供的参数运行智能体工作流
        ask(
            question=user_query,
            debug=args.debug,
            max_plan_iterations=args.max_plan_iterations,
            max_step_num=args.max_step_num,
            enable_background_investigation=args.enable_background_investigation,
            enable_clarification=args.enable_clarification,
            max_clarification_rounds=args.max_clarification_rounds,
        )
