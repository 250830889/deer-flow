# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 本地文档检索工具模块
## 提供用于从本地知识库检索相关文档的工具

import logging                              ## 日志记录模块
from typing import List, Optional, Type     ## 类型注解模块

from langchain_core.callbacks import (      ## LangChain回调管理器
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool   ## LangChain基础工具类
from pydantic import BaseModel, Field       ## Pydantic模型和字段定义

from src.config.tools import SELECTED_RAG_PROVIDER  ## 配置的RAG提供商
from src.rag import Document, Resource, Retriever, build_retriever  ## RAG相关类和构建函数

logger = logging.getLogger(__name__)        ## 创建日志记录器


## 检索工具输入模型
## 定义本地检索工具的输入参数
class RetrieverInput(BaseModel):
    keywords: str = Field(description="search keywords to look up")  ## 搜索关键词


## 本地检索工具类
## 用于从带有rag:// URI前缀的文件中检索信息，优先级高于网络搜索或编写代码
class RetrieverTool(BaseTool):
    name: str = "local_search_tool"  ## 工具名称
    description: str = "Useful for retrieving information from the file with `rag://` uri prefix, it should be higher priority than the web search or writing code. Input should be a search keywords."  ## 工具描述
    args_schema: Type[BaseModel] = RetrieverInput  ## 输入参数模型

    retriever: Retriever = Field(default_factory=Retriever)  ## 检索器实例
    resources: list[Resource] = Field(default_factory=list)  ## 资源列表

    ## 同步运行检索
    ## 参数：
    ##   keywords: 搜索关键词
    ##   run_manager: 可选的回调管理器
    ## 返回值：
    ##   相关文档列表或未找到结果的提示信息
    def _run(
        self,
        keywords: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[Document]:
        ## 记录检索查询日志，附加资源信息
        logger.info(
            f"Retriever tool query: {keywords}", extra={"resources": self.resources}
        )
        ## 执行文档检索
        documents = self.retriever.query_relevant_documents(keywords, self.resources)
        ## 如果没有找到文档，返回提示信息
        if not documents:
            return "No results found from the local knowledge base."
        ## 将文档转换为字典列表返回
        return [doc.to_dict() for doc in documents]

    ## 异步运行检索
    ## 参数：
    ##   keywords: 搜索关键词
    ##   run_manager: 可选的异步回调管理器
    ## 返回值：
    ##   相关文档列表或未找到结果的提示信息
    async def _arun(
        self,
        keywords: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list[Document]:
        ## 调用同步方法的实现
        return self._run(keywords, run_manager.get_sync())


## 获取检索工具实例
## 根据提供的资源列表创建并返回RetrieverTool实例
## 参数：
##   resources: 资源列表
## 返回值：
##   RetrieverTool实例或None（如果资源为空或构建检索器失败）
def get_retriever_tool(resources: List[Resource]) -> RetrieverTool | None:
    ## 如果资源列表为空，返回None
    if not resources:
        return None
    ## 记录创建检索工具日志，包含选中的RAG提供商
    logger.info(f"create retriever tool: {SELECTED_RAG_PROVIDER}")
    ## 构建检索器实例
    retriever = build_retriever()

    ## 如果构建检索器失败，返回None
    if not retriever:
        return None
    ## 创建并返回RetrieverTool实例
    return RetrieverTool(retriever=retriever, resources=resources)
