# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any

# 助手角色名称集合
ASSISTANT_SPEAKER_NAMES = {
    "coordinator",
    "planner",
    "researcher",
    "coder",
    "reporter",
    "background_investigator",
}


def get_message_content(message: Any) -> str:
    """从字典或LangChain消息中提取消息内容。"""
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def is_user_message(message: Any) -> bool:
    """判断消息是否来自最终用户。

    Args:
        message: 要检查的消息对象，可以是字典或具有特定属性的对象

    Returns:
        bool: 如果消息来自用户返回True，否则返回False
    """
    # 检查消息是否为字典类型
    if isinstance(message, dict):
        # 获取消息的角色并转换为小写
        role = (message.get("role") or "").lower()
        # 如果角色是"user"或"human"，则认为是用户消息
        if role in {"user", "human"}:
            return True
        # 如果角色是"assistant"或"system"，则不是用户消息
        if role in {"assistant", "system"}:
            return False
        
        # 获取消息的名称并转换为小写
        name = (message.get("name") or "").lower()
        # 如果名称存在于助手名称列表中，则不是用户消息
        if name and name in ASSISTANT_SPEAKER_NAMES:
            return False
        # 如果角色为空且名称不在助手名称列表中，则认为是用户消息
        return role == "" and name not in ASSISTANT_SPEAKER_NAMES

    # 获取消息对象的类型属性并转换为小写
    message_type = (getattr(message, "type", "") or "").lower()
    # 获取消息对象的名称属性并转换为小写
    name = (getattr(message, "name", "") or "").lower()
    
    # 如果消息类型为"human"且名称不在助手名称列表中，则认为是用户消息
    if message_type == "human":
        return not (name and name in ASSISTANT_SPEAKER_NAMES)

    # 获取消息对象的role属性
    role_attr = getattr(message, "role", None)
    # 如果role属性是字符串且为"user"或"human"，则认为是用户消息
    if isinstance(role_attr, str) and role_attr.lower() in {"user", "human"}:
        return True

    # 获取消息对象的additional_kwargs中的role属性
    additional_role = getattr(message, "additional_kwargs", {}).get("role")
    # 如果additional_role是字符串且为"user"或"human"，则认为是用户消息
    if isinstance(additional_role, str) and additional_role.lower() in {
        "user",
        "human",
    }:
        return True

    # 如果以上条件都不满足，则不是用户消息
    return False


def get_latest_user_message(messages: list[Any]) -> tuple[Any, str]:
    """返回最新的用户消息及其内容。

    Args:
        messages: 消息列表，按时间顺序排列（旧消息在前，新消息在后）

    Returns:
        返回一个元组，包含：
        - 最新的用户消息对象（如果没有找到则返回None）
        - 消息的内容字符串（如果没有找到则返回空字符串）
    """
    # 从后向前遍历消息列表，以找到最新的用户消息
    # 使用reversed()函数反转列表，使最新消息排在前面
    # messages or [] 确保即使messages为None也不会出错
    for message in reversed(messages or []):
        # 使用is_user_message函数检查消息是否来自用户
        if is_user_message(message):
            # 提取消息内容
            content = get_message_content(message)
            # 如果内容不为空，返回消息对象和内容
            if content:
                return message, content
    
    # 如果没有找到有效的用户消息，返回None和空字符串
    return None, ""


def build_clarified_topic_from_history(
    clarification_history: list[str],
) -> tuple[str, list[str]]:
    """从有序的澄清历史记录构建澄清后的主题字符串。

    Args:
        clarification_history: 按时间顺序排列的澄清历史记录列表

    Returns:
        返回一个元组，包含：
        - 澄清后的主题字符串（将历史记录合并为单一字符串）
        - 过滤后的澄清历史记录列表（移除空项）
    """
    # 过滤掉澄清历史中的空项，创建一个新的序列
    sequence = [item for item in clarification_history if item]
    
    # 如果过滤后的序列为空，返回空字符串和空列表
    if not sequence:
        return "", []
    
    # 如果序列只有一个元素，直接返回该元素和序列本身
    if len(sequence) == 1:
        return sequence[0], sequence
    
    # 解构序列：第一个元素作为头部(head)，其余元素作为尾部(tail)
    head, *tail = sequence
    
    # 构建澄清后的字符串：头部内容 + " - " + 尾部内容用逗号和空格连接
    # 例如：["研究AI", "特别是深度学习", "关注图像识别"] 
    # 会变成："研究AI - 特别是深度学习, 关注图像识别"
    clarified_string = f"{head} - {', '.join(tail)}"
    
    # 返回构建的澄清字符串和过滤后的序列
    return clarified_string, sequence


def reconstruct_clarification_history(
    messages: list[Any],
    fallback_history: list[str] | None = None,
    base_topic: str = "",
) -> list[str]:
    """从用户消息重建澄清历史，提供回退机制。

    Args:
        messages: 按时间顺序排列的对话消息列表
        fallback_history: 当没有找到用户消息时使用的可选现有历史记录
        base_topic: 当没有用户消息可用时使用的可选主题

    Returns:
        返回一个清理后的澄清历史，包含唯一的连续用户内容
    """
    # 初始化序列列表，用于存储澄清历史记录
    sequence: list[str] = []
    
    # 遍历消息列表，处理每条消息
    for message in messages or []:
        # 跳过非用户消息，只处理用户发送的消息
        if not is_user_message(message):
            continue
        
        # 提取消息内容
        content = get_message_content(message)
        # 如果内容为空，跳过此消息
        if not content:
            continue
        
        # 检查序列中最后一条消息是否与当前内容相同
        # 避免重复内容被添加到历史记录中
        if sequence and sequence[-1] == content:
            continue
        
        # 将内容添加到序列中
        sequence.append(content)

    # 如果成功从消息中提取了内容，返回该序列
    if sequence:
        return sequence

    # 如果没有从消息中提取到内容，尝试使用回退历史记录
    # 过滤掉空字符串项
    fallback = [item for item in (fallback_history or []) if item]
    # 如果回退历史记录不为空，返回该记录
    if fallback:
        return fallback

    # 如果既没有消息内容也没有回退历史记录，尝试使用基础主题
    base_topic = (base_topic or "").strip()
    # 如果基础主题不为空，返回包含基础主题的列表
    return [base_topic] if base_topic else []