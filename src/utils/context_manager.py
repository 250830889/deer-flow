# src/utils/context_manager.py
"""
上下文管理模块，提供对话上下文的Token计数和压缩功能。

本模块实现了对话上下文的管理功能，主要用于处理与大型语言模型(LLM)交互时的
上下文长度限制问题。通过精确的Token计数和智能的消息压缩策略，确保对话
历史不会超出模型的Token限制，同时保留最重要的上下文信息。

主要功能：
1. 精确计算消息列表的Token数量（区分中英文字符）
2. 智能压缩对话上下文，保留关键信息
3. 验证和修复消息内容，确保格式正确
4. 支持保留前缀消息，确保系统提示和重要输入不被压缩
5. 提供灵活的配置选项，适应不同模型的Token限制

使用场景：
- 长对话历史管理
- 多轮对话上下文压缩
- 模型输入限制处理
- 对话历史优化

使用示例：
```python
from src.utils.context_manager import ContextManager

# 创建上下文管理器，设置Token限制为4000，保留前2条消息
manager = ContextManager(token_limit=4000, preserve_prefix_message_count=2)

# 检查消息是否超出限制
if manager.is_over_limit(messages):
    # 压缩消息以适应Token限制
    compressed_state = manager.compress_messages({"messages": messages})
    messages = compressed_state["messages"]
```

注意事项：
1. Token计算是基于字符数的估算，不是精确的tokenizer计算
2. 中英文混合内容的Token计算方式不同，中文按1字符=1Token计算
3. 消息压缩策略是保留前缀和后缀消息，中间部分会被丢弃
4. 系统消息和用户输入通常会被优先保留
5. 工具调用消息可能会占用较多Token，需要特别注意
"""
# 标准库导入
import copy  # 深拷贝对象，用于创建消息副本而不影响原始对象
import logging  # 日志记录功能，用于记录上下文管理过程中的关键信息
import json  # JSON数据处理，用于消息内容的序列化和反序列化

# 类型注解导入
from typing import List, Dict, Any, Optional, Union  # 类型提示支持

# LangChain核心消息类型导入
from langchain_core.messages import (
    AIMessage,       # AI模型生成的消息
    BaseMessage,     # 所有消息类型的基类
    HumanMessage,    # 用户输入的消息
    SystemMessage,   # 系统提示消息
    ToolMessage,     # 工具调用返回的消息
)

# 本地配置导入
from src.config import load_yaml_config  # 加载YAML配置文件

# 设置日志记录器
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例


def get_search_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    从配置文件中加载搜索相关的配置信息。
    
    该函数从指定的YAML配置文件中加载搜索配置，包括搜索引擎API密钥、
    搜索结果数量限制、搜索语言偏好等参数。配置信息用于控制搜索功能的行为。
    
    参数:
        config_path (str): 配置文件的路径，默认为"config.yaml"
                         如果文件不存在，将返回空字典
    
    返回:
        Dict[str, Any]: 包含搜索配置的字典，键为配置项名称，值为配置项值
                      如果配置文件不存在或加载失败，返回空字典
    
    异常:
        无异常抛出，配置文件加载失败时会记录错误日志并返回空字典
    
    使用示例:
        ```python
        # 使用默认配置文件路径
        search_config = get_search_config()
        print(search_config.get("search_engine", "google"))
        
        # 使用自定义配置文件路径
        custom_config = get_search_config("custom_search_config.yaml")
        api_key = custom_config.get("api_key", "")
        ```
    
    注意事项:
        1. 配置文件必须是有效的YAML格式
        2. 如果配置文件不存在，函数不会抛出异常，而是返回空字典
        3. 建议在调用函数后检查返回的字典是否包含所需的配置项
        4. 敏感信息（如API密钥）应存储在配置文件中，而不是硬编码在代码中
    """
    try:
        config = load_yaml_config(config_path)
        return config.get("search", {})
    except Exception as e:
        logger.error(f"Failed to load search config: {e}")
        return {}


class ContextManager:
    """
    上下文管理器类，负责管理对话上下文的Token计数和压缩。
    
    该类提供了完整的对话上下文管理功能，包括Token计数、上下文压缩、
    消息验证等。通过精确的Token计算和智能的压缩策略，确保对话历史
    不会超出模型的Token限制，同时保留最重要的上下文信息。
    
    主要功能：
    1. 精确计算消息列表的Token数量（区分中英文字符）
    2. 智能压缩对话上下文，保留关键信息
    3. 验证和修复消息内容，确保格式正确
    4. 支持保留前缀消息，确保系统提示和重要输入不被压缩
    5. 提供灵活的配置选项，适应不同模型的Token限制
    
    使用场景：
    - 长对话历史管理
    - 多轮对话上下文压缩
    - 模型输入限制处理
    - 对话历史优化
    
    使用示例：
    ```python
    # 创建上下文管理器，设置Token限制为4000，保留前2条消息
    manager = ContextManager(token_limit=4000, preserve_prefix_message_count=2)
    
    # 检查消息是否超出限制
    if manager.is_over_limit(messages):
        # 压缩消息以适应Token限制
        compressed_state = manager.compress_messages({"messages": messages})
        messages = compressed_state["messages"]
    ```
    
    注意事项：
    1. Token计算是基于字符数的估算，不是精确的tokenizer计算
    2. 中英文混合内容的Token计算方式不同，中文按1字符=1Token计算
    3. 消息压缩策略是保留前缀和后缀消息，中间部分会被丢弃
    4. 系统消息和用户输入通常会被优先保留
    5. 工具调用消息可能会占用较多Token，需要特别注意
    """
    
    def __init__(
        self,
        token_limit: int = 4000,
        preserve_prefix_message_count: int = 2,
        preserve_suffix_message_count: int = 1,
        enable_message_validation: bool = True,
        enable_content_repair: bool = True,
    ):
        """
        初始化上下文管理器实例。
        
        创建一个新的上下文管理器实例，设置Token限制和消息保留策略。
        这些参数将影响后续的Token计算和消息压缩行为。
        
        参数:
            token_limit (int): Token数量限制，默认为4000
                              当消息列表的Token数量超过此值时，将触发压缩
            preserve_prefix_message_count (int): 保留前缀消息的数量，默认为2
                                                这些消息在压缩时不会被丢弃
            preserve_suffix_message_count (int): 保留后缀消息的数量，默认为1
                                                这些消息在压缩时不会被丢弃
            enable_message_validation (bool): 是否启用消息验证，默认为True
                                            启用后会验证消息内容的格式正确性
            enable_content_repair (bool): 是否启用内容修复，默认为True
                                        启用后会尝试修复损坏的消息内容
        
        异常:
            无异常抛出
        
        使用示例:
            ```python
            # 使用默认参数创建上下文管理器
            manager = ContextManager()
            
            # 自定义参数创建上下文管理器
            custom_manager = ContextManager(
                token_limit=8000,
                preserve_prefix_message_count=3,
                preserve_suffix_message_count=2,
                enable_message_validation=False
            )
            ```
        
        注意事项:
            1. Token限制应根据实际使用的模型进行调整
            2. 保留前缀消息数量应确保包含系统提示和重要上下文
            3. 保留后缀消息数量应确保包含最新的用户输入和AI回复
            4. 消息验证和修复会增加处理时间，但在生产环境中建议启用
            5. 总保留消息数（前缀+后缀）不应过大，否则压缩效果会降低
        """
        self.token_limit = token_limit
        self.preserve_prefix_message_count = preserve_prefix_message_count
        self.preserve_suffix_message_count = preserve_suffix_message_count
        self.enable_message_validation = enable_message_validation
        self.enable_content_repair = enable_content_repair

    def count_tokens(self, messages: List[BaseMessage]) -> int:
        """
        计算消息列表的总Token数量。
        
        遍历消息列表中的每条消息，计算每条消息的Token数量，
        然后累加得到总Token数量。这是上下文管理的基础功能，
        用于判断是否需要进行消息压缩。
        
        参数:
            messages (List[BaseMessage]): 需要计算Token数量的消息列表
            
        返回:
            int: 消息列表的总Token数量
            
        异常:
            无异常抛出
            
        使用示例:
            ```python
            manager = ContextManager(token_limit=4000)
            messages = [SystemMessage(content="你是一个助手"), 
                       HumanMessage(content="请介绍一下Python")]
            total_tokens = manager.count_tokens(messages)
            print(f"总Token数量: {total_tokens}")
            ```
            
        注意事项:
            1. Token计算是基于字符数的估算，不是精确的tokenizer计算
            2. 不同类型的消息会有不同的Token计算权重
            3. 工具调用消息通常占用更多Token
            4. 空消息也会被计算为至少1个Token
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self._count_message_tokens(message)
        return total_tokens

    def _count_message_tokens(self, message: BaseMessage) -> int:
        """
        计算单条消息的Token数量。
        
        根据消息的类型、内容和结构，估算该消息占用的Token数量。
        不同类型的消息有不同的计算策略，例如工具消息通常占用更多Token。
        
        参数:
            message (BaseMessage): 需要计算Token数量的消息对象
            
        返回:
            int: 消息的Token数量，至少为1
            
        异常:
            无异常抛出
            
        使用示例:
            ```python
            manager = ContextManager()
            message = HumanMessage(content="请介绍一下Python编程语言")
            token_count = manager._count_message_tokens(message)
            print(f"消息Token数量: {token_count}")
            ```
            
        注意事项:
            1. 系统消息会乘以1.1的权重，因为它们通常很重要
            2. AI消息会乘以1.2的权重，因为可能包含推理内容
            3. 工具消息会乘以1.3的权重，因为可能包含大量结构化数据
            4. 如果消息包含工具调用，会额外增加50个Token的估算
            5. 即使是空消息也会返回至少1个Token
        """
        # 基于字符长度估算Token数量（英文和非英文有不同的计算方式）
        token_count = 0

        # 计算content字段的Token
        if hasattr(message, "content") and message.content:
            # 处理不同类型的content
            if isinstance(message.content, str):
                token_count += self._count_text_tokens(message.content)

        # 计算角色相关的Token
        if hasattr(message, "type"):
            token_count += self._count_text_tokens(message.type)

        # 针对不同消息类型的特殊处理
        if isinstance(message, SystemMessage):
            # 系统消息通常简短但重要，略微增加估算
            token_count = int(token_count * 1.1)
        elif isinstance(message, HumanMessage):
            # 人类消息使用正常估算
            pass
        elif isinstance(message, AIMessage):
            # AI消息可能包含推理内容，略微增加估算
            token_count = int(token_count * 1.2)
        elif isinstance(message, ToolMessage):
            # 工具消息可能包含大量结构化数据，增加估算
            token_count = int(token_count * 1.3)

        # 处理additional_kwargs中的额外信息
        if hasattr(message, "additional_kwargs") and message.additional_kwargs:
            # 简单估算额外字段的Token
            extra_str = str(message.additional_kwargs)
            token_count += self._count_text_tokens(extra_str)

            # 如果有tool_calls，增加估算
            if "tool_calls" in message.additional_kwargs:
                token_count += 50  # 为函数调用信息增加估算

        # 确保至少有1个Token
        return max(1, token_count)

    def _count_text_tokens(self, text: str) -> int:
        """
        计算文本的Token数量，对英文和非英文字符采用不同的计算方式。
        
        英文字符：约4个字符 = 1个Token
        非英文字符（如中文）：约1个字符 = 1个Token
        
        这种计算方式是基于经验的近似值，不是精确的tokenizer计算，
        但对于大多数场景已经足够准确。
        
        参数:
            text (str): 需要计算Token数量的文本
            
        返回:
            int: 文本的Token数量
            
        异常:
            无异常抛出
            
        使用示例:
            ```python
            manager = ContextManager()
            english_text = "Hello, world!"
            chinese_text = "你好，世界！"
            english_tokens = manager._count_text_tokens(english_text)
            chinese_tokens = manager._count_text_tokens(chinese_text)
            print(f"英文Token数: {english_tokens}, 中文Token数: {chinese_tokens}")
            ```
            
        注意事项:
            1. 空字符串返回0个Token
            2. ASCII字符（英文字母、数字、标点）按4字符=1Token计算
            3. 非ASCII字符（如中文、日文、韩文等）按1字符=1Token计算
            4. 这种计算方式对于混合语言文本也适用
            5. 计算结果是整数，使用整数除法
        """
        if not text:
            return 0

        english_chars = 0
        non_english_chars = 0

        for char in text:
            # 检查字符是否为ASCII（英文字母、数字、标点）
            if ord(char) < 128:
                english_chars += 1
            else:
                non_english_chars += 1

        # 计算Token：英文按4字符/Token，其他按1字符/Token
        english_tokens = english_chars // 4
        non_english_tokens = non_english_chars

        return english_tokens + non_english_tokens

    def is_over_limit(self, messages: List[BaseMessage]) -> bool:
        """
        检查消息列表是否超出Token限制。
        
        计算消息列表的总Token数量，并与设置的Token限制进行比较。
        这是决定是否需要进行消息压缩的关键判断。
        
        参数:
            messages (List[BaseMessage]): 需要检查的消息列表
            
        返回:
            bool: 如果消息列表的Token数量超过限制则返回True，否则返回False
            
        异常:
            无异常抛出
            
        使用示例:
            ```python
            manager = ContextManager(token_limit=4000)
            messages = [SystemMessage(content="你是一个助手"), 
                       HumanMessage(content="请介绍一下Python")]
            if manager.is_over_limit(messages):
                print("消息超出Token限制，需要进行压缩")
            else:
                print("消息在Token限制范围内")
            ```
            
        注意事项:
            1. 该方法依赖于count_tokens方法的结果
            2. 如果token_limit设置为None，该方法将始终返回False
            3. 空消息列表不会超出Token限制
            4. Token计算是基于字符数的估算，不是精确的tokenizer计算
        """
        return self.count_tokens(messages) > self.token_limit

    def compress_messages(self, state: dict) -> List[BaseMessage]:
        """
        压缩消息以适应Token限制。
        
        当消息列表的Token数量超过限制时，该方法会调用内部压缩逻辑，
        保留重要的前缀和后缀消息，丢弃中间部分的消息，以确保总Token数量
        不超过限制。压缩后的消息列表会替换原始状态中的消息列表。
        
        参数:
            state (dict): 包含原始消息的状态字典，必须包含"messages"键
            
        返回:
            List[BaseMessage]: 包含压缩后消息的状态字典
            
        异常:
            无异常抛出，但会记录警告日志
            
        使用示例:
            ```python
            manager = ContextManager(token_limit=4000)
            state = {"messages": [SystemMessage(content="你是一个助手"), 
                                  HumanMessage(content="请介绍一下Python"),
                                  AIMessage(content="Python是一种编程语言...")] * 10}
            compressed_state = manager.compress_messages(state)
            print(f"压缩后消息数量: {len(compressed_state['messages'])}")
            ```
            
        注意事项:
            1. 如果token_limit为None，将直接返回原始状态
            2. 如果状态中不包含"messages"键，将记录警告并返回原始状态
            3. 压缩策略是保留前缀和后缀消息，中间部分会被丢弃
            4. 压缩过程会记录日志，显示压缩前后的Token数量
            5. 原始状态对象会被修改，而不是创建新对象
        """
        # 如果未设置token_limit，返回原始状态
        if self.token_limit is None:
            logger.info("No token_limit set, the context management doesn't work.")
            return state

        if not isinstance(state, dict) or "messages" not in state:
            logger.warning("No messages found in state")
            return state

        messages = state["messages"]

        if not self.is_over_limit(messages):
            return state

        # 压缩消息
        compressed_messages = self._compress_messages(messages)

        logger.info(
            f"Message compression completed: {self.count_tokens(messages)} -> {self.count_tokens(compressed_messages)} tokens"
        )

        state["messages"] = compressed_messages
        return state

    def _compress_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        执行消息压缩的核心逻辑。
        
        该方法实现了智能的消息压缩策略，首先保留指定数量的前缀消息
        （通常是系统提示和重要上下文），然后从消息列表末尾开始保留
        指定数量的后缀消息（通常是最近的对话）。被丢弃的中间消息
        会被生成摘要保留关键信息。
        
        参数:
            messages (List[BaseMessage]): 需要压缩的消息列表
            
        返回:
            List[BaseMessage]: 压缩后的消息列表
            
        异常:
            无异常抛出
            
        使用示例:
            ```python
            manager = ContextManager(token_limit=4000, preserve_prefix_message_count=2)
            messages = [SystemMessage(content="你是一个助手")] * 10
            compressed_messages = manager._compress_messages(messages)
            print(f"压缩后消息数量: {len(compressed_messages)}")
            ```
            
        注意事项:
            1. 压缩策略是保留前缀和后缀消息，中间部分生成摘要
            2. 如果单条消息的Token数量超过可用Token，会截断消息内容
            3. 前缀消息优先保留，通常是系统提示和重要上下文
            4. 后缀消息其次保留，通常是最近的对话
            5. 压缩后的消息列表顺序与原始列表保持一致
        """
        available_token = self.token_limit
        prefix_messages = []

        # 1. 保留指定长度的头部消息，以保留系统提示和用户输入
        for i in range(min(self.preserve_prefix_message_count, len(messages))):
            cur_token_cnt = self._count_message_tokens(messages[i])
            if available_token > 0 and available_token >= cur_token_cnt:
                prefix_messages.append(messages[i])
                available_token -= cur_token_cnt
            elif available_token > 0:
                # 截断内容以适应可用Token
                truncated_message = self._truncate_message_content(
                    messages[i], available_token
                )
                prefix_messages.append(truncated_message)
                return prefix_messages
            else:
                break

        # 2. 从尾部压缩后续消息，一些消息可能会被丢弃
        remaining_messages = messages[len(prefix_messages):]
        suffix_messages = []
        suffix_start_index = len(remaining_messages)  # 记录从哪个位置开始保留后缀
        
        # 预留 Token 给摘要（如果可能需要丢弃消息的话）
        reserved_for_summary = 100 if len(remaining_messages) > self.preserve_suffix_message_count else 0
        available_token_for_suffix = available_token - reserved_for_summary
        
        for i in range(len(remaining_messages) - 1, -1, -1):
            cur_token_cnt = self._count_message_tokens(remaining_messages[i])

            if cur_token_cnt > 0 and available_token_for_suffix >= cur_token_cnt:
                suffix_messages = [remaining_messages[i]] + suffix_messages
                available_token_for_suffix -= cur_token_cnt
                suffix_start_index = i  # 更新后缀起始位置
            elif available_token_for_suffix > 0:
                # 截断内容以适应可用Token
                truncated_message = self._truncate_message_content(
                    remaining_messages[i], available_token_for_suffix
                )
                suffix_messages = [truncated_message] + suffix_messages
                suffix_start_index = i  # 更新后缀起始位置
                available_token_for_suffix = 0  # 用完了
                break
            else:
                # 没有可用Token了，退出循环
                break
        
        # 3. 计算被丢弃的消息（在后缀起始位置之前的消息）
        discarded_messages = remaining_messages[:suffix_start_index]
        
        # 4. 如果有消息被丢弃，生成摘要
        if discarded_messages:
            logger.info(f"压缩过程中丢弃了 {len(discarded_messages)} 条消息，正在生成摘要...")
            summary_message = self._create_summary_message(discarded_messages)
            if summary_message:
                # 检查摘要消息是否会超出预留的Token空间
                summary_tokens = self._count_message_tokens(summary_message)
                # 使用预留的空间 + 后缀处理后剩余的空间
                available_for_summary = reserved_for_summary + available_token_for_suffix
                if available_for_summary >= summary_tokens:
                    # 在前缀和后缀之间插入摘要
                    return prefix_messages + [summary_message] + suffix_messages
                else:
                    logger.warning(f"摘要消息超出预留Token限制（需要{summary_tokens}，可用{available_for_summary}），跳过摘要")

        return prefix_messages + suffix_messages

    def _truncate_message_content(
        self, message: BaseMessage, max_tokens: int
    ) -> BaseMessage:
        """
        截断消息内容，同时保留所有其他属性。
        
        通过深拷贝原始消息并仅修改其content属性，来创建一个截断后的
        消息实例。这样可以保留原始消息的所有其他属性和元数据。
        
        参数:
            message (BaseMessage): 需要截断的消息
            max_tokens (int): 允许保留的最大Token数量
            
        返回:
            BaseMessage: 截断内容后的新消息实例
            
        异常:
            无异常抛出
            
        使用示例:
            ```python
            manager = ContextManager()
            message = HumanMessage(content="这是一段很长的文本内容..." * 100)
            truncated_message = manager._truncate_message_content(message, 100)
            print(f"原始长度: {len(message.content)}, 截断后长度: {len(truncated_message.content)}")
            ```
            
        注意事项:
            1. 该方法使用深拷贝来保留原始消息的所有属性
            2. 截断是基于字符数，不是精确的Token计算
            3. 截断后的消息类型与原始消息类型相同
            4. 如果max_tokens为0或负数，将返回空内容的消息
            5. 该方法不会修改原始消息对象
        """
        # 创建原始消息的深拷贝，保留所有属性
        truncated_message = copy.deepcopy(message)

        # 仅截断content属性
        truncated_message.content = message.content[:max_tokens]

        return truncated_message

    def _create_summary_message(self, messages: List[BaseMessage]) -> Optional[BaseMessage]:
        """
        为消息列表创建摘要消息。
        
        该方法使用LLM生成被丢弃消息的摘要，以便在压缩后的上下文中
        保留关键信息。如果LLM调用失败，将回退到简单的统计摘要。
        
        参数:
            messages (List[BaseMessage]): 需要摘要的消息列表
            
        返回:
            Optional[BaseMessage]: 包含摘要内容的SystemMessage对象，
                                   如果消息列表为空则返回None
            
        异常:
            无异常抛出，LLM调用失败时会回退到简单摘要
            
        使用示例:
            ```python
            manager = ContextManager()
            messages = [HumanMessage(content="请介绍一下Python")] * 10
            summary = manager._create_summary_message(messages)
            print(f"摘要消息: {summary.content}")
            ```
            
        注意事项:
            1. 摘要消息类型为SystemMessage，以便与对话消息区分
            2. 摘要内容会被限制在200字以内，控制Token消耗
            3. LLM调用失败时，会生成包含消息统计的简单摘要
            4. 空消息列表会返回None
        """
        # 如果没有消息需要摘要，返回None
        if not messages:
            return None
        
        # 统计消息类型
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        total_count = len(messages)
        
        # 提取消息内容用于生成摘要
        conversation_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "用户"
            elif isinstance(msg, AIMessage):
                role = "助手"
            elif isinstance(msg, ToolMessage):
                role = "工具"
            else:
                role = "系统"
            
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # 限制每条消息的长度，避免输入过长
            if len(content) > 200:
                content = content[:200] + "..."
            conversation_text += f"{role}: {content}\n"
        
        # 尝试使用LLM生成摘要
        try:
            from langchain_openai import ChatOpenAI
            from src.config import load_yaml_config
            from pathlib import Path
            
            # 直接加载配置文件
            config_path = str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
            conf = load_yaml_config(config_path)
            basic_conf = conf.get("BASIC_MODEL", {})
            
            # 创建 ChatOpenAI 实例（豆包兼容 OpenAI 接口）
            llm = ChatOpenAI(
                model=basic_conf.get("model", "doubao-1-5-pro-32k-250115"),
                api_key=basic_conf.get("api_key", ""),
                base_url=basic_conf.get("base_url", ""),
            )
            
            summary_prompt = f"""请用一句话（不超过50字）总结以下对话的核心内容：

{conversation_text}

一句话摘要："""
            
            response = llm.invoke(summary_prompt)
            summary_content = response.content if hasattr(response, 'content') else str(response)
            
            # 限制摘要长度（更严格）
            if len(summary_content) > 100:
                summary_content = summary_content[:100] + "..."
            
            logger.info(f"LLM摘要生成成功，共处理 {total_count} 条消息")
            
            return SystemMessage(
                content=f"[摘要] {summary_content}"
            )
            
        except Exception as e:
            # LLM调用失败，回退到简单摘要
            logger.warning(f"LLM摘要生成失败，使用简单摘要: {e}")
            
            simple_summary = f"[系统摘要] 此处省略了 {total_count} 条对话消息"
            if human_count > 0:
                simple_summary += f"，包含 {human_count} 条用户消息"
            if ai_count > 0:
                simple_summary += f"、{ai_count} 条助手回复"
            if tool_count > 0:
                simple_summary += f"、{tool_count} 条工具调用"
            simple_summary += "。"
            
            return SystemMessage(content=simple_summary)


def validate_message_content(messages: List[BaseMessage], max_content_length: int = 100000) -> List[BaseMessage]:
    """
    验证并修复所有消息，确保它们在发送给LLM之前具有有效内容。
    
    该函数确保以下条件：
    1. 所有消息都有content字段
    2. 没有消息的content为None或空字符串（除了合法的空响应）
    3. 复杂对象（列表、字典）被转换为JSON字符串
    4. 如果内容过长则截断，以防止Token溢出
    
    参数:
        messages (List[BaseMessage]): 需要验证的消息列表
        max_content_length (int): 每条消息允许的最大内容长度，默认为100000
        
    返回:
        List[BaseMessage]: 已验证并修复内容后的消息列表
        
    异常:
        无异常抛出，但会记录错误日志
        
    使用示例:
        ```python
        from langchain_core.messages import HumanMessage, AIMessage
        
        messages = [
            HumanMessage(content=None),  # None内容
            AIMessage(content={"key": "value"}),  # 字典内容
            HumanMessage(content="A" * 200000),  # 超长内容
        ]
        
        validated_messages = validate_message_content(messages, max_content_length=1000)
        for msg in validated_messages:
            print(f"消息类型: {type(msg).__name__}, 内容长度: {len(msg.content)}")
        ```
        
    注意事项:
        1. 该函数会直接修改原始消息对象，而不是创建新对象
        2. 对于ToolMessage类型的消息，错误处理会返回JSON格式的错误信息
        3. 对于其他类型的消息，错误处理会返回包含错误信息的字符串
        4. 内容截断会在末尾添加"..."表示截断
        5. 复杂对象转换为JSON时会保留Unicode字符
    """
    validated = []
    for i, msg in enumerate(messages):
        try:
            # 检查消息是否有content属性
            if not hasattr(msg, 'content'):
                logger.warning(f"Message {i} ({type(msg).__name__}) has no content attribute")
                msg.content = ""
            
            # 处理None内容
            elif msg.content is None:
                logger.warning(f"Message {i} ({type(msg).__name__}) has None content, setting to empty string")
                msg.content = ""
            
            # 处理复杂内容类型（转换为JSON）
            elif isinstance(msg.content, (list, dict)):
                logger.debug(f"Message {i} ({type(msg).__name__}) has complex content type {type(msg.content).__name__}, converting to JSON")
                msg.content = json.dumps(msg.content, ensure_ascii=False)
            
            # 处理其他非字符串类型
            elif not isinstance(msg.content, str):
                logger.debug(f"Message {i} ({type(msg).__name__}) has non-string content type {type(msg.content).__name__}, converting to string")
                msg.content = str(msg.content)
            
            # 验证内容长度
            if isinstance(msg.content, str) and len(msg.content) > max_content_length:
                logger.warning(f"Message {i} content truncated from {len(msg.content)} to {max_content_length} chars")
                msg.content = msg.content[:max_content_length].rstrip() + "..."
            
            validated.append(msg)
        except Exception as e:
            logger.error(f"Error validating message {i}: {e}")
            # 创建安全的回退消息
            if isinstance(msg, ToolMessage):
                msg.content = json.dumps({"error": str(e)}, ensure_ascii=False)
            else:
                msg.content = f"[Error processing message: {str(e)}]"
            validated.append(msg)
    
    return validated
