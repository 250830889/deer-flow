# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 聊天请求数据模型模块
## 该文件定义了DeerFlow API服务器使用的各种请求数据模型
## 所有模型基于Pydantic构建，确保数据验证和类型安全

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from src.config.report_style import ReportStyle
from src.rag.retriever import Resource


## 内容项模型
## 表示聊天消息中的单个内容项，可以是文本或图片
class ContentItem(BaseModel):
    ## 内容类型（text, image等）
    type: str = Field(..., description="The type of content (text, image, etc.)")
    ## 文本内容（当type为'text'时）
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    ## 图片URL（当type为'image'时）
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


## 聊天消息模型
## 表示用户和助手之间的单条聊天消息
class ChatMessage(BaseModel):
    ## 消息发送者角色（user或assistant）
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    ## 消息内容，可以是字符串或内容项列表
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


## 聊天请求模型
## 表示完整的聊天请求，包含消息历史、资源配置和各种参数
class ChatRequest(BaseModel):
    ## 用户和助手之间的消息历史
    messages: Optional[List[ChatMessage]] = Field(
        [], description="History of messages between the user and the assistant"
    )
    ## 用于研究的资源列表
    resources: Optional[List[Resource]] = Field(
        [], description="Resources to be used for the research"
    )
    ## 是否启用调试日志
    debug: Optional[bool] = Field(False, description="Whether to enable debug logging")
    ## 特定的对话标识符
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier"
    )
    ## 对话的语言区域设置（例如：en-US, zh-CN）
    locale: Optional[str] = Field(
        "en-US", description="Language locale for the conversation (e.g., en-US, zh-CN)"
    )
    ## 最大计划迭代次数
    max_plan_iterations: Optional[int] = Field(
        1, description="The maximum number of plan iterations"
    )
    ## 计划中的最大步骤数
    max_step_num: Optional[int] = Field(
        3, description="The maximum number of steps in a plan"
    )
    ## 最大搜索结果数
    max_search_results: Optional[int] = Field(
        3, description="The maximum number of search results"
    )
    ## 是否自动接受计划
    auto_accepted_plan: Optional[bool] = Field(
        False, description="Whether to automatically accept the plan"
    )
    ## 用户对计划的中断反馈
    interrupt_feedback: Optional[str] = Field(
        None, description="Interrupt feedback from the user on the plan"
    )
    ## 聊天请求的MCP设置
    mcp_settings: Optional[dict] = Field(
        None, description="MCP settings for the chat request"
    )
    ## 是否在计划前进行背景调查
    enable_background_investigation: Optional[bool] = Field(
        True, description="Whether to get background investigation before plan"
    )
    ## 报告的风格
    report_style: Optional[ReportStyle] = Field(
        ReportStyle.ACADEMIC, description="The style of the report"
    )
    ## 是否启用深度思考
    enable_deep_thinking: Optional[bool] = Field(
        False, description="Whether to enable deep thinking"
    )
    ## 是否启用多轮澄清
    enable_clarification: Optional[bool] = Field(
        None,
        description="Whether to enable multi-turn clarification (default: None, uses State default=False)",
    )
    ## 最大澄清轮数
    max_clarification_rounds: Optional[int] = Field(
        None,
        description="Maximum number of clarification rounds (default: None, uses State default=3)",
    )
    ## 在执行前需要中断的工具列表
    interrupt_before_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names to interrupt before execution (e.g., ['db_tool', 'api_tool'])\n",
    )


## 文本转语音请求模型
## 表示将文本转换为语音的请求
class TTSRequest(BaseModel):
    ## 要转换为语音的文本
    text: str = Field(..., description="The text to convert to speech")
    ## 使用的语音类型
    voice_type: Optional[str] = Field(
        "BV700_V2_streaming", description="The voice type to use"
    )
    ## 音频编码格式
    encoding: Optional[str] = Field("mp3", description="The audio encoding format")
    ## 语速比例
    speed_ratio: Optional[float] = Field(1.0, description="Speech speed ratio")
    ## 音量比例
    volume_ratio: Optional[float] = Field(1.0, description="Speech volume ratio")
    ## 音调比例
    pitch_ratio: Optional[float] = Field(1.0, description="Speech pitch ratio")
    ## 文本类型（plain或ssml）
    text_type: Optional[str] = Field("plain", description="Text type (plain or ssml)")
    ## 是否使用前端处理
    with_frontend: Optional[int] = Field(
        1, description="Whether to use frontend processing"
    )
    ## 前端类型
    frontend_type: Optional[str] = Field("unitTson", description="Frontend type")


## 生成播客请求模型
## 表示生成播客内容的请求
class GeneratePodcastRequest(BaseModel):
    ## 播客的内容
    content: str = Field(..., description="The content of the podcast")


## 生成PPT请求模型
## 表示生成PPT演示文稿的请求
class GeneratePPTRequest(BaseModel):
    ## PPT的内容
    content: str = Field(..., description="The content of the ppt")


## 生成散文请求模型
## 表示生成散文内容的请求
class GenerateProseRequest(BaseModel):
    ## 散文的提示词
    prompt: str = Field(..., description="The content of the prose")
    ## 散文写作器的选项
    option: str = Field(..., description="The option of the prose writer")
    ## 散文写作器的用户自定义命令
    command: Optional[str] = Field(
        "", description="The user custom command of the prose writer"
    )


## 增强提示词请求模型
## 表示增强原始提示词的请求
class EnhancePromptRequest(BaseModel):
    ## 要增强的原始提示词
    prompt: str = Field(..., description="The original prompt to enhance")
    ## 关于预期用途的附加上下文
    context: Optional[str] = Field(
        "", description="Additional context about the intended use"
    )
    ## 报告的风格
    report_style: Optional[str] = Field(
        "academic", description="The style of the report"
    )