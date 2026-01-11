# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
JSON工具模块

该模块提供了一系列用于处理JSON数据的实用函数，主要用于处理和修复
从语言模型或工具调用中获取的JSON数据。模块包含以下主要功能：

1. 参数清理：防止特殊字符导致的问题
2. JSON提取：从包含额外标记的内容中提取有效的JSON
3. JSON修复：修复不完整或格式错误的JSON
4. 响应清理：清理工具响应中的无效内容和垃圾字符

主要使用场景：
- 处理语言模型生成的JSON输出
- 修复量化模型产生的损坏JSON
- 清理工具调用响应中的无效内容
- 防止特殊字符导致的解析错误

示例用法：
```python
from src.utils.json_utils import repair_json_output, sanitize_tool_response

# 修复模型输出的JSON
broken_json = '{"name": "test", "value": 123'  # 缺少闭合括号
fixed_json = repair_json_output(broken_json)

# 清理工具响应
response = sanitize_tool_response(tool_output, max_length=10000)
```

注意事项：
- JSON修复功能依赖于json_repair库
- 清理过程可能会截断过长的内容
- 某些极端损坏的JSON可能无法完全修复
"""

# 标准库导入
import json  # JSON数据编码和解码
import logging  # 日志记录功能
import re  # 正则表达式操作
from typing import Any  # 类型注解支持

# 第三方库导入
import json_repair  # JSON修复库，用于修复损坏的JSON数据

# 模块级日志记录器
logger = logging.getLogger(__name__)


def sanitize_args(args: Any) -> str:
    """
    清理工具调用参数，防止特殊字符导致的问题。
    
    该函数将参数中的方括号和花括号替换为HTML实体编码，防止这些字符
    在某些上下文中被错误解析。这对于确保工具调用参数的正确传递
    和解析非常重要，特别是在处理可能包含这些特殊字符的JSON数据时。
    
    参数:
        args (Any): 工具调用参数，可以是任何类型，但只有字符串类型会被处理
        
    返回:
        str: 清理后的参数字符串，非字符串输入将返回空字符串
        
    异常:
        无异常抛出
        
    使用示例:
        ```python
        # 清理包含特殊字符的参数
        raw_args = '{"key": "[value]"}'
        clean_args = sanitize_args(raw_args)
        print(clean_args)  # 输出: &#123;"key": "&#91;value&#93;"&#125;
        ```
        
    注意事项:
        1. 只有字符串类型的输入会被处理，其他类型将返回空字符串
        2. 该函数不会验证参数的有效性，仅进行字符替换
        3. 替换后的字符串可能需要在使用前进行解码
        4. 此函数主要用于防止特殊字符在某些上下文中被错误解析
    """
    if not isinstance(args, str):
        return ""
    else:
        # 将特殊字符替换为HTML实体编码
        return (
            args.replace("[", "&#91;")    # 左方括号
            .replace("]", "&#93;")        # 右方括号
            .replace("{", "&#123;")       # 左花括号
            .replace("}", "&#125;")       # 右花括号
        )


def _extract_json_from_content(content: str) -> str:
    """
    从可能包含额外标记的内容中提取有效的JSON。
    
    该函数通过分析字符串中的括号匹配情况，找到最后一个有效的JSON
    闭合括号并在此处截断内容。它能够正确处理对象{}和数组[]两种
    JSON结构，并考虑字符串中的转义字符和引号，确保不会错误地
    将字符串内的括号计入匹配计数。
    
    参数:
        content (str): 可能包含JSON的字符串，JSON后可能跟着额外的文本
        
    返回:
        str: 提取出的潜在JSON字符串，如果找不到有效JSON则返回原始内容
        
    异常:
        无异常抛出
        
    使用示例:
        ```python
        # 从包含额外文本的内容中提取JSON
        content = '{"name": "test", "value": 123} 这是额外的文本'
        json_part = _extract_json_from_content(content)
        print(json_part)  # 输出: {"name": "test", "value": 123}
        
        # 处理嵌套结构
        nested = '{"data": {"items": [1, 2, 3]}} 更多内容'
        json_part = _extract_json_from_content(nested)
        print(json_part)  # 输出: {"data": {"items": [1, 2, 3]}}
        ```
        
    注意事项:
        1. 该函数只处理完整的JSON对象或数组，不处理JSON片段
        2. 函数会保留原始内容的缩进和格式
        3. 如果JSON格式错误（如括号不匹配），函数可能无法正确提取
        4. 该函数不会验证提取的内容是否为有效JSON，只检查括号匹配
        5. 提取过程会记录调试日志，显示截断前后的字符数差异
    """
    content = content.strip()
    
    # 尝试找到一个完整的JSON对象或数组
    # 查找最后一个可能是有效JSON的闭合括号
    
    # 跟踪计数器和是否已看到开始括号
    brace_count = 0          # 花括号计数器
    bracket_count = 0        # 方括号计数器
    seen_opening_brace = False   # 是否已看到开始花括号
    seen_opening_bracket = False  # 是否已看到开始方括号
    in_string = False        # 是否在字符串内
    escape_next = False      # 下一个字符是否被转义
    last_valid_end = -1      # 最后一个有效结束位置
    
    for i, char in enumerate(content):
        # 处理转义字符
        if escape_next:
            escape_next = False
            continue
        
        # 检测转义序列
        if char == '\\':
            escape_next = True
            continue
        
        # 检测字符串边界（忽略转义的引号）
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        # 跳过字符串内的字符
        if in_string:
            continue
        
        # 处理花括号
        if char == '{':
            brace_count += 1
            seen_opening_brace = True
        elif char == '}':
            brace_count -= 1
            # 只有在开始于花括号且达到平衡状态时才标记为有效结束
            if brace_count == 0 and seen_opening_brace:
                last_valid_end = i
        # 处理方括号
        elif char == '[':
            bracket_count += 1
            seen_opening_bracket = True
        elif char == ']':
            bracket_count -= 1
            # 只有在开始于方括号且达到平衡状态时才标记为有效结束
            if bracket_count == 0 and seen_opening_bracket:
                last_valid_end = i
    
    # 如果找到有效结束位置，截断内容
    if last_valid_end > 0:
        truncated = content[:last_valid_end + 1]
        if truncated != content:
            logger.debug(f"Truncated content from {len(content)} to {len(truncated)} chars")
        return truncated
    
    return content


def _extract_code_block_content(content: str) -> str:
    if not isinstance(content, str):
        return ""

    match = re.search(r"```(?:json|ts)?\s*(.*?)```", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    for marker in ("```json", "```ts"):
        if marker in content:
            return content.split(marker, 1)[1].strip()

    return content


def _looks_like_json_payload(content: str) -> bool:
    if not content:
        return False

    stripped = content.lstrip()
    if stripped.startswith(("{", "[", "\"")):
        return True
    if "```json" in content or "```ts" in content:
        return True
    if ("{" in content and "}" in content) or ("[" in content and "]" in content):
        return True

    return False


def _truncate_response(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def repair_json_output(json_str: str) -> str:
    """
    尝试修复损坏的JSON字符串，使其成为有效的JSON格式。
    
    该函数首先尝试使用json_repair库自动修复JSON，如果失败则尝试
    从内容中提取有效的JSON部分。该函数能够处理常见的JSON格式错误，
    如缺少引号、多余逗号、括号不匹配等问题。修复过程会记录详细的
    调试信息，帮助诊断JSON损坏的原因。
    
    参数:
        json_str (str): 可能损坏的JSON字符串
        
    返回:
        str: 修复后的JSON字符串，如果无法修复则返回原始字符串
        
    异常:
        无异常抛出
        
    使用示例:
        ```python
        # 修复缺少引号的JSON
        bad_json = '{name: "test", value: 123}'  # name缺少引号
        fixed_json = repair_json_output(bad_json)
        print(fixed_json)  # 输出: {"name": "test", "value": 123}
        
        # 修复多余逗号的JSON
        bad_json = '{"items": [1, 2, 3,], "count": 3,}'  # 有多余逗号
        fixed_json = repair_json_output(bad_json)
        print(fixed_json)  # 输出: {"items": [1, 2, 3], "count": 3}
        
        # 处理JSON后跟额外文本的情况
        mixed = '{"result": "success"} 这不是JSON部分'
        fixed_json = repair_json_output(mixed)
        print(fixed_json)  # 输出: {"result": "success"}
        ```
        
    注意事项:
        1. 该函数依赖于json_repair库，需要确保该库已正确安装
        2. 修复过程可能会改变原始JSON的格式（如空格、缩进等）
        3. 对于严重损坏的JSON，修复可能不完整或失败
        4. 函数不会验证修复后的JSON语义正确性，只保证语法有效
        5. 修复失败时会记录警告日志，并尝试提取有效JSON部分
        6. 如果原始内容完全不是JSON格式，函数将返回原始字符串
    """
    if json_str is None:
        return ""
    if not isinstance(json_str, str):
        json_str = str(json_str)

    if not json_str.strip():
        return ""

    candidate = _extract_code_block_content(json_str).strip()
    if not _looks_like_json_payload(candidate):
        return json_str

    candidate = _extract_json_from_content(candidate)

    try:
        # Attempt to repair JSON with json_repair.
        logger.debug("Attempting to repair JSON with json_repair")
        repaired = json_repair.repair_json(candidate)
        parsed = json.loads(repaired)
        if isinstance(parsed, str) and not candidate.lstrip().startswith('\"'):
            return json_str
        logger.debug("Successfully repaired JSON")
        return json.dumps(parsed, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")

    # Fallback to extracting JSON from content.
    logger.debug("Attempting to extract JSON from content")
    return _extract_json_from_content(candidate)

def sanitize_tool_response(response: str, max_length: int = 50000) -> str:
    """
    清理工具调用响应内容，确保返回有效的JSON格式。
    
    该函数用于处理工具调用的响应内容，去除可能的解释性文本，
    修复JSON格式错误，并确保返回的内容是有效的JSON。它首先尝试
    从响应中提取JSON部分，然后尝试修复任何JSON格式错误，
    最后验证修复后的内容是否为有效的JSON格式。
    
    参数:
        response (str): 工具调用返回的原始响应字符串，可能包含
                       JSON数据和额外的解释性文本
        
    返回:
        str: 清理后的JSON字符串，如果无法提取或修复有效JSON，
             则返回原始响应字符串
        
    异常:
        无异常抛出
        
    使用示例:
        ```python
        # 处理包含解释性文本的响应
        raw_response = '以下是查询结果：{"users": ["Alice", "Bob"], "count": 2}'
        clean_json = sanitize_tool_response(raw_response)
        print(clean_json)  # 输出: {"users": ["Alice", "Bob"], "count": 2}
        
        # 处理格式错误的JSON响应
        bad_response = '{result: "success", data: [1, 2, 3],}'  # 缺少引号和多余逗号
        clean_json = sanitize_tool_response(bad_response)
        print(clean_json)  # 输出: {"result": "success", "data": [1, 2, 3]}
        
        # 处理纯文本响应（非JSON）
        text_response = '操作已完成，但没有返回结构化数据。'
        clean_json = sanitize_tool_response(text_response)
        print(clean_json)  # 输出: 操作已完成，但没有返回结构化数据。
        ```
        
    注意事项:
        1. 该函数会保留原始响应中的所有JSON数据，但会移除周围的文本
        2. 函数依赖于json_repair库来修复损坏的JSON格式
        3. 如果响应中包含多个JSON对象，函数只会提取最后一个有效的JSON
        4. 清理过程会记录调试日志，可用于诊断问题
        5. 如果响应完全不是JSON格式，函数将返回原始字符串而不做修改
        6. 该函数不会验证JSON内容的语义正确性，只保证语法有效
    """
    if response is None:
        return ""
    if not isinstance(response, str):
        response = str(response)

    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", response)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""

    candidate = _extract_code_block_content(cleaned).strip()
    if _looks_like_json_payload(candidate):
        json_part = _extract_json_from_content(candidate)
        repaired_json = repair_json_output(json_part)
        try:
            parsed = json.loads(repaired_json)
            if isinstance(parsed, str) and not candidate.lstrip().startswith('\"'):
                return _truncate_response(cleaned, max_length)
            return _truncate_response(repaired_json, max_length)
        except json.JSONDecodeError:
            logger.debug("Could not extract valid JSON from tool response")
            return _truncate_response(cleaned, max_length)

    return _truncate_response(cleaned, max_length)

