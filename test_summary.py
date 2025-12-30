# 测试 context_manager 的摘要功能
# 文件位置: d:\AI\deer-flow\test_summary.py

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.context_manager import ContextManager

def test_summary_feature():
    """测试摘要功能"""
    print("=" * 50)
    print("测试 ContextManager 摘要功能")
    print("=" * 50)
    
    # 创建一个 Token 限制较低的管理器（便于触发压缩）
    manager = ContextManager(
        token_limit=250,  # 设置合适的限制
        preserve_prefix_message_count=1,  # 只保留系统提示
        preserve_suffix_message_count=1   # 保留最后一条用户消息
    )
    
    # 创建一个较长的对话历史
    messages = [
        SystemMessage(content="你是一个专业的Python编程助手，擅长解答编程问题。"),
        HumanMessage(content="请介绍一下Python的基础语法。"),
        AIMessage(content="Python是一种简洁易读的编程语言。基础语法包括：1. 变量定义不需要声明类型；2. 使用冒号和缩进来表示代码块；3. 支持多种数据类型如列表、字典、元组等。"),
        HumanMessage(content="那Python的循环语句怎么写？"),
        AIMessage(content="Python支持for循环和while循环。for循环通常用于遍历序列，例如：for i in range(10): print(i)。while循环用于条件判断，例如：while count < 10: count += 1。"),
        HumanMessage(content="函数怎么定义？"),
        AIMessage(content="使用def关键字定义函数。例如：def my_function(param1, param2): return param1 + param2。Python支持默认参数、关键字参数、可变参数等多种参数形式。"),
        HumanMessage(content="面向对象编程呢？"),
        AIMessage(content="Python是完全面向对象的语言。使用class关键字定义类，支持继承、封装、多态等特性。例如：class MyClass: def __init__(self): self.value = 0。"),
        HumanMessage(content="最后总结一下Python的特点。"),
    ]
    
    print(f"\n原始消息数量: {len(messages)}")
    print(f"原始Token数量: {manager.count_tokens(messages)}")
    print(f"Token限制: {manager.token_limit}")
    print(f"是否超出限制: {manager.is_over_limit(messages)}")
    
    # 测试压缩功能
    print("\n" + "-" * 50)
    print("开始压缩消息...")
    print("-" * 50)
    
    state = {"messages": messages}
    compressed_state = manager.compress_messages(state)
    compressed_messages = compressed_state["messages"]
    
    print(f"\n压缩后消息数量: {len(compressed_messages)}")
    print(f"压缩后Token数量: {manager.count_tokens(compressed_messages)}")
    
    # 打印压缩后的消息
    print("\n" + "=" * 50)
    print("压缩后的消息内容:")
    print("=" * 50)
    
    for i, msg in enumerate(compressed_messages):
        msg_type = type(msg).__name__
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"\n[{i+1}] {msg_type}:")
        print(f"    {content}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    test_summary_feature()
