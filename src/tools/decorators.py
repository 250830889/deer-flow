# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 工具装饰器模块
## 提供用于工具函数和类的日志记录装饰器

import functools              ## 函数工具模块，用于装饰器实现
import logging                ## 日志记录模块
from typing import Any, Callable, Type, TypeVar  ## 类型注解模块

logger = logging.getLogger(__name__)  ## 创建日志记录器

T = TypeVar("T")  ## 类型变量，用于泛型函数定义


## 日志记录装饰器
## 用于记录工具函数的输入参数和输出结果
## 参数：
##   func: 要装饰的工具函数
## 返回值：
##   包装后的函数，带有输入输出日志记录功能
def log_io(func: Callable) -> Callable:
    """
    A decorator that logs the input parameters and output of a tool function.

    Args:
        func: The tool function to be decorated

    Returns:
        The wrapped function with input/output logging
    """

    @functools.wraps(func)  ## 保留原函数的元信息
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ## 记录输入参数
        func_name = func.__name__  ## 获取函数名
        ## 格式化参数列表，将位置参数和关键字参数转换为字符串
        params = ", ".join(
            [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
        )
        logger.info(f"Tool {func_name} called with parameters: {params}")  ## 记录调用日志

        ## 执行原函数
        result = func(*args, **kwargs)

        ## 记录输出结果
        logger.info(f"Tool {func_name} returned: {result}")  ## 记录返回值日志

        return result

    return wrapper


## 日志记录工具混入类
## 为任何工具类添加日志记录功能
class LoggedToolMixin:
    """A mixin class that adds logging functionality to any tool."""

    ## 工具操作日志记录辅助方法
    ## 记录工具方法的调用信息
    ## 参数：
    ##   method_name: 调用的方法名
    ##   *args: 位置参数
    ##   **kwargs: 关键字参数
    def _log_operation(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Helper method to log tool operations."""
        ## 获取工具名称（移除Logged前缀）
        tool_name = self.__class__.__name__.replace("Logged", "")
        ## 格式化参数列表
        params = ", ".join(
            [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
        )
        ## 记录方法调用日志
        logger.debug(f"Tool {tool_name}.{method_name} called with parameters: {params}")

    ## 重写_run方法，添加日志记录
    ## 当工具执行_run方法时，自动记录日志
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Override _run method to add logging."""
        ## 记录_run方法调用
        self._log_operation("_run", *args, **kwargs)
        ## 调用父类的_run方法
        result = super()._run(*args, **kwargs)
        ## 记录返回结果
        logger.debug(
            f"Tool {self.__class__.__name__.replace('Logged', '')} returned: {result}"
        )
        return result


## 带日志功能的工具类工厂函数
## 创建任何工具类的带日志版本
## 参数：
##   base_tool_class: 原始工具类
## 返回值：
##   一个新类，继承自LoggedToolMixin和原始工具类，带有日志记录功能
def create_logged_tool(base_tool_class: Type[T]) -> Type[T]:
    """
    Factory function to create a logged version of any tool class.

    Args:
        base_tool_class: The original tool class to be enhanced with logging

    Returns:
        A new class that inherits from both LoggedToolMixin and the base tool class
    """

    ## 创建新的工具类，继承自LoggedToolMixin和原始工具类
    class LoggedTool(LoggedToolMixin, base_tool_class):
        pass

    ## 为新类设置更具描述性的名称
    LoggedTool.__name__ = f"Logged{base_tool_class.__name__}"
    return LoggedTool
