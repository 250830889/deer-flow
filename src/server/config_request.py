# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 配置请求响应模型模块
## 该文件定义了DeerFlow API服务器配置相关的响应数据模型
## 用于向客户端返回服务器的配置信息

from pydantic import BaseModel, Field

from src.server.rag_request import RAGConfigResponse


## 配置响应模型
## 表示服务器配置的响应结构
## 包含RAG配置和已配置的模型信息
class ConfigResponse(BaseModel):
    """Response model for server config."""

    ## RAG（检索增强生成）的配置信息
    rag: RAGConfigResponse = Field(..., description="The config of the RAG")
    ## 已配置的模型列表，以字典形式表示，键为模型类型，值为模型名称列表
    models: dict[str, list[str]] = Field(..., description="The configured models")