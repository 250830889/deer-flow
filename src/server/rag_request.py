# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## RAG配置和资源请求响应模型模块
## 该文件定义了检索增强生成(RAG)系统相关的请求和响应数据模型
## 用于管理RAG配置、搜索和返回相关资源

from pydantic import BaseModel, Field

from src.rag.retriever import Resource


## RAG配置响应模型
## 表示RAG系统配置的响应结构
## 包含RAG提供商信息
class RAGConfigResponse(BaseModel):
    """Response model for RAG config."""

    ## RAG提供商，默认为ragflow
    provider: str | None = Field(
        None, description="The provider of the RAG, default is ragflow"
    )


## RAG资源请求模型
## 表示搜索RAG资源的请求结构
## 包含搜索查询关键词
class RAGResourceRequest(BaseModel):
    """Request model for RAG resource."""

    ## 需要搜索的资源查询关键词
    query: str | None = Field(
        None, description="The query of the resource need to be searched"
    )


## RAG资源响应模型
## 表示RAG资源搜索结果的响应结构
## 包含搜索到的资源列表
class RAGResourcesResponse(BaseModel):
    """Response model for RAG resources."""

    ## RAG系统的资源列表
    resources: list[Resource] = Field(..., description="The resources of the RAG")