# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
日志清理工具模块，用于防止日志注入攻击。

本模块提供了在记录日志之前清理用户控制输入的功能，以防止攻击者通过以下方式伪造日志条目：
- 换行注入 (\n)
- HTML注入（针对HTML日志）
- 可能被误解的特殊字符序列

日志注入攻击是一种安全漏洞，攻击者可以通过在输入中插入特殊字符（如换行符、控制字符等）
来伪造日志条目，误导日志解析系统或隐藏恶意活动。本模块提供的工具可以有效防止这类攻击。

主要功能：
1. 清理用户输入中的危险字符
2. 截断过长的输入以防止日志洪水攻击
3. 为不同类型的日志内容提供专门的清理函数
4. 提供安全的日志消息模板功能

使用示例：
```python
from src.utils.log_sanitizer import sanitize_log_input, create_safe_log_message

# 清理用户输入
user_input = "恶意输入\n[ERROR] 伪造错误"
safe_input = sanitize_log_input(user_input)

# 创建安全的日志消息
log_msg = create_safe_log_message(
    "用户 {user_id} 执行了 {action} 操作",
    user_id="admin\n[INFO]",
    action="删除"
)
```

注意事项：
1. 所有用户控制的输入在记录日志前都应该进行清理
2. 不同类型的日志内容有不同的长度限制
3. 清理过程会转义特殊字符，但不会改变原始语义
4. 对于极长的输入，会进行截断处理
"""

# 导入标准库模块
import re  # 正则表达式模块，用于模式匹配和字符串替换

# 导入类型注解模块
from typing import Any, Optional  # Any表示任意类型，Optional表示可选类型


def sanitize_log_input(value: Any, max_length: int = 500) -> str:
    """
    清理用户控制的输入，使其可以安全地记录到日志中。

    此函数通过将危险字符（换行符、制表符、回车符等）替换为它们的转义表示，
    来防止日志注入攻击。这是防止攻击者通过特殊字符伪造日志条目的核心函数。

    日志注入攻击示例：
    - 输入: "用户操作\n[ERROR] 系统被攻击"
    - 不安全的日志: "用户操作
                       [ERROR] 系统被攻击"  # 伪造了错误日志
    - 安全的日志: "用户操作\\n[ERROR] 系统被攻击"  # 转义了换行符

    Args:
        value (Any): 需要清理的输入值，可以是任意类型
        max_length (int): 输出字符串的最大长度，如果超过则截断。默认为500个字符

    Returns:
        str: 清理后的字符串，可以安全地用于日志记录

    Raises:
        无异常抛出，所有输入都会被转换为字符串并处理

    Examples:
        >>> sanitize_log_input("普通文本")
        '普通文本'

        >>> sanitize_log_input("恶意输入\\n[INFO] 伪造条目")
        '恶意输入\\\\n[INFO] 伪造条目'

        >>> sanitize_log_input("制表符\\t这里")
        '制表符\\\\t这里'

        >>> sanitize_log_input(None)
        'None'

        >>> long_text = "a" * 1000
        >>> result = sanitize_log_input(long_text, max_length=100)
        >>> len(result) <= 100
        True

    Note:
        1. 此函数会首先将输入转换为字符串，然后进行清理
        2. 清理过程包括转义特殊字符和移除控制字符
        3. 如果输入超过最大长度，会在末尾添加"..."表示截断
        4. 此函数是所有其他清理函数的基础，其他函数都是对此函数的封装
    """
    if value is None:
        return "None"

    # 将输入转换为字符串
    string_value = str(value)

    # 将危险字符替换为它们的转义表示
    # 替换顺序很重要：首先转义反斜杠，以避免双重转义
    replacements = {
        "\\": "\\\\",  # 反斜杠（必须首先处理）
        "\n": "\\n",   # 换行符 - 防止创建新的日志条目
        "\r": "\\r",   # 回车符
        "\t": "\\t",   # 制表符
        "\x00": "\\0",  # 空字符
        "\x1b": "\\x1b",  # 转义字符（用于ANSI序列）
    }

    # 逐个替换危险字符
    for char, replacement in replacements.items():
        string_value = string_value.replace(char, replacement)

    # 移除其他控制字符（ASCII 0-31，除了已处理的字符）
    # 这些字符在日志中很少有用，且可能被利用进行攻击
    string_value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", string_value)

    # 如果字符串过长则截断（防止日志洪水攻击）
    if len(string_value) > max_length:
        string_value = string_value[: max_length - 3] + "..."

    return string_value


def sanitize_thread_id(thread_id: Any) -> str:
    """
    清理线程ID以便安全地记录到日志中。

    线程ID应该是字母数字、连字符和下划线的组合，但为了防御性编程，
    我们仍然进行完整的清理处理。

    Args:
        thread_id (Any): 需要清理的线程ID

    Returns:
        str: 清理后的线程ID，最大长度限制为100个字符

    Note:
        虽然线程ID通常是系统生成的，相对安全，但在某些情况下可能来自用户输入，
        因此仍然需要进行清理以防止日志注入。
    """
    return sanitize_log_input(thread_id, max_length=100)


def sanitize_user_content(content: Any) -> str:
    """
    清理用户提供的消息内容以便安全地记录到日志中。

    用户消息可以是任意长度，因此我们更激进地截断内容，以防止日志洪水攻击。

    Args:
        content (Any): 需要清理的用户内容

    Returns:
        str: 清理后的用户内容，最大长度限制为200个字符

    Note:
        用户内容是最需要谨慎处理的输入类型，因为它完全由用户控制，
        可能包含恶意构造的字符串。因此我们设置了相对较短的长度限制。
    """
    return sanitize_log_input(content, max_length=200)


def sanitize_agent_name(agent_name: Any) -> str:
    """
    清理代理名称以便安全地记录到日志中。

    代理名称应该是简单的标识符，但为了防御性编程，我们仍然进行完整的清理处理。

    Args:
        agent_name (Any): 需要清理的代理名称

    Returns:
        str: 清理后的代理名称，最大长度限制为100个字符

    Note:
        虽然代理名称通常是预定义的，但在某些动态配置或插件系统中，
        可能来自用户输入，因此仍然需要进行清理。
    """
    return sanitize_log_input(agent_name, max_length=100)


def sanitize_tool_name(tool_name: Any) -> str:
    """
    清理工具名称以便安全地记录到日志中。

    工具名称应该是简单的标识符，但为了防御性编程，我们仍然进行完整的清理处理。

    Args:
        tool_name (Any): 需要清理的工具名称

    Returns:
        str: 清理后的工具名称，最大长度限制为100个字符

    Note:
        虽然工具名称通常是预定义的，但在某些动态配置或插件系统中，
        可能来自用户输入，因此仍然需要进行清理。
    """
    return sanitize_log_input(tool_name, max_length=100)


def sanitize_feedback(feedback: Any) -> str:
    """
    清理用户反馈以便安全地记录到日志中。

    反馈可能来自中断的任意文本，因此需要谨慎地进行清理。

    Args:
        feedback (Any): 需要清理的反馈内容

    Returns:
        str: 清理后的反馈内容，最大长度限制为150个字符（更激进的截断）

    Note:
        用户反馈通常来自中断或异常处理流程，可能包含敏感信息或恶意构造的内容。
        因此我们设置了中等长度的限制，既保留了足够的信息，又防止了日志洪水攻击。
    """
    return sanitize_log_input(feedback, max_length=150)


def create_safe_log_message(template: str, **kwargs) -> str:
    """
    通过清理所有值来创建安全的日志消息。

    此函数使用带有关键字参数的模板字符串，在替换之前清理每个值，
    以防止日志注入攻击。这是构建包含用户输入的日志消息的推荐方法。

    日志注入攻击示例：
    - 模板: "用户 {user_id} 执行了 {action} 操作"
    - 不安全的参数: user_id="admin\n[ERROR]", action="删除"
    - 不安全的日志: "用户 admin
                       [ERROR] 执行了 删除 操作"  # 伪造了错误日志
    - 安全的日志: "用户 admin\\n[ERROR] 执行了 删除 操作"  # 转义了换行符

    Args:
        template (str): 包含{key}占位符的模板字符串
        **kwargs: 要替换的键值对，所有值都会被清理

    Returns:
        str: 安全的日志消息，所有用户输入都已被清理

    Raises:
        KeyError: 如果模板中包含未提供的关键字参数
        ValueError: 如果模板格式不正确

    Examples:
        >>> msg = create_safe_log_message(
        ...     "[{thread_id}] 处理 {tool_name}",
        ...     thread_id="abc\\n[INFO]",
        ...     tool_name="我的工具"
        ... )
        >>> "[abc\\\\n[INFO]] 处理 我的工具" in msg
        True

        >>> # 处理多个参数
        >>> log_msg = create_safe_log_message(
        ...     "用户 {user} 在 {time} 使用 {tool} 执行了 {action}",
        ...     user="admin\\r[ATTACK]",
        ...     time="2023-01-01",
        ...     tool="search\\x1b[31m",
        ...     action="查询\\t数据"
        ... )
        >>> "admin\\\\r[ATTACK]" in log_msg
        True

    Note:
        1. 所有传递给此函数的值都会被sanitize_log_input函数清理
        2. 模板字符串本身不会被清理，应使用硬编码的可信字符串
        3. 此函数不会清理模板字符串本身，只清理替换的值
        4. 对于复杂的日志消息，建议使用此函数而不是手动字符串拼接
        5. 如果模板格式不正确或缺少参数，会抛出相应的异常
    """
    # 清理所有值
    safe_kwargs = {
        key: sanitize_log_input(value) for key, value in kwargs.items()
    }

    # 替换到模板中
    return template.format(**safe_kwargs)
