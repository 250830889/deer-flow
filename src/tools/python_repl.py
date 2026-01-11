# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## Python REPL工具模块
## 提供用于执行Python代码的工具，支持数据分析和计算

import logging                              ## 日志记录模块
import os                                   ## 操作系统接口模块，用于读取环境变量
from typing import Annotated, Optional      ## 类型注解模块

from langchain_core.tools import tool       ## LangChain工具装饰器
from langchain_experimental.utilities import PythonREPL  ## Python REPL执行工具

from .decorators import log_io              ## 导入日志记录装饰器


## 检查Python REPL工具是否启用
## 首先检查环境变量ENABLE_PYTHON_REPL，支持多种启用值
## 返回值：
##   布尔值，表示REPL工具是否启用
def _is_python_repl_enabled() -> bool:
    """Check if Python REPL tool is enabled from configuration."""
    ## 检查环境变量
    env_enabled = os.getenv("ENABLE_PYTHON_REPL", "false").lower()
    ## 支持多种启用值：true、1、yes、on
    if env_enabled in ("true", "1", "yes", "on"):
        return True
    return False


## 初始化REPL和日志记录器
## 如果REPL工具启用，则创建PythonREPL实例，否则为None
repl: Optional[PythonREPL] = PythonREPL() if _is_python_repl_enabled() else None
logger = logging.getLogger(__name__)        ## 创建日志记录器


@tool                                       ## 标记为LangChain工具
@log_io                                     ## 添加输入输出日志记录
## Python REPL执行工具函数
## 用于执行Python代码进行数据分析或计算
## 参数：
##   code: 要执行的Python代码字符串
## 返回值：
##   执行结果字符串，如果有错误则返回错误信息
def python_repl_tool(
    code: Annotated[
        str, "The python code to execute to do further analysis or calculation."
    ],
):
    """Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""

    ## 检查工具是否启用
    if not _is_python_repl_enabled():
        error_msg = "Python REPL tool is disabled. Please enable it in environment configuration."
        logger.warning(error_msg)  ## 记录警告日志
        return f"Tool disabled: {error_msg}"

    ## 验证输入类型
    if not isinstance(code, str):
        error_msg = f"Invalid input: code must be a string, got {type(code)}"
        logger.error(error_msg)  ## 记录错误日志
        return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"

    logger.info("Executing Python code")  ## 记录执行日志
    try:
        ## 执行Python代码
        result = repl.run(code)
        ## 检查结果是否包含错误信息
        if isinstance(result, str) and ("Error" in result or "Exception" in result):
            logger.error(result)  ## 记录错误日志
            return f"Error executing code:\n```python\n{code}\n```\nError: {result}"
        logger.info("Code execution successful")  ## 记录成功日志
    except BaseException as e:
        ## 捕获所有异常
        error_msg = repr(e)
        logger.error(error_msg)  ## 记录异常日志
        return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"

    ## 格式化执行结果
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str
