# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## MCP服务器元数据请求响应模型模块
## 该文件定义了模型控制协议(MCP)服务器元数据相关的请求和响应数据模型
## 用于管理和获取MCP服务器的配置信息和可用工具列表

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


## MCP服务器元数据请求模型
## 表示获取MCP服务器元数据的请求结构
## 包含连接类型、命令、URL等配置信息
class MCPServerMetadataRequest(BaseModel):
    """Request model for MCP server metadata."""

    ## MCP服务器连接类型（stdio、sse或streamable_http）
    transport: str = Field(
        ...,
        description=(
            "The type of MCP server connection (stdio or sse or streamable_http)"
        ),
    )
    ## 要执行的命令（用于stdio类型）
    command: Optional[str] = Field(
        None, description="The command to execute (for stdio type)"
    )
    ## 命令参数（用于stdio类型）
    args: Optional[List[str]] = Field(
        None, description="Command arguments (for stdio type)"
    )
    ## SSE服务器的URL（用于sse类型）
    url: Optional[str] = Field(
        None, description="The URL of the SSE server (for sse type)"
    )
    ## 环境变量（用于stdio类型）
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (for stdio type)"
    )
    ## HTTP头信息（用于sse/streamable_http类型）
    headers: Optional[Dict[str, str]] = Field(
        None, description="HTTP headers (for sse/streamable_http type)"
    )
    ## 操作的可选自定义超时时间（秒）
    timeout_seconds: Optional[int] = Field(
        None, description="Optional custom timeout in seconds for the operation"
    )


## MCP服务器元数据响应模型
## 表示MCP服务器元数据的响应结构
## 包含服务器配置信息和可用工具列表
class MCPServerMetadataResponse(BaseModel):
    """Response model for MCP server metadata."""

    ## MCP服务器连接类型（stdio、sse或streamable_http）
    transport: str = Field(
        ...,
        description=(
            "The type of MCP server connection (stdio or sse or streamable_http)"
        ),
    )
    ## 要执行的命令（用于stdio类型）
    command: Optional[str] = Field(
        None, description="The command to execute (for stdio type)"
    )
    ## 命令参数（用于stdio类型）
    args: Optional[List[str]] = Field(
        None, description="Command arguments (for stdio type)"
    )
    ## SSE服务器的URL（用于sse类型）
    url: Optional[str] = Field(
        None, description="The URL of the SSE server (for sse type)"
    )
    ## 环境变量（用于stdio类型）
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (for stdio type)"
    )
    ## HTTP头信息（用于sse/streamable_http类型）
    headers: Optional[Dict[str, str]] = Field(
        None, description="HTTP headers (for sse/streamable_http type)"
    )
    ## 从MCP服务器获取的可用工具列表
    tools: List = Field(
        default_factory=list, description="Available tools from the MCP server"
    )