# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 支持图片搜索的Tavily搜索工具模块
## 继承自TavilySearchResults，添加了图片搜索和结果处理功能

import json  ## JSON处理模块，用于格式化结果
import logging  ## 日志记录模块
from typing import Dict, List, Optional, Tuple, Union  ## 类型注解模块

from langchain.callbacks.manager import (  ## LangChain回调管理器
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain_community.tools.tavily_search.tool import TavilySearchResults  ## Tavily搜索结果工具基类
from pydantic import Field  ## Pydantic字段定义

from src.tools.tavily_search.tavily_search_api_wrapper import (
    EnhancedTavilySearchAPIWrapper,  ## 导入增强型Tavily搜索API包装器
)

logger = logging.getLogger(__name__)  ## 创建日志记录器


## 支持图片搜索的Tavily搜索工具类
## 继承自TavilySearchResults，添加了图片搜索和结果处理功能
## 可以返回包含图片的搜索结果
class TavilySearchWithImages(TavilySearchResults):  # type: ignore[override, override]
    """Tool that queries the Tavily Search API and gets back json.

    Setup:
        Install ``langchain-openai`` and ``tavily-python``, and set environment variable ``TAVILY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community tavily-python
            export TAVILY_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_tavily.tavily_search import TavilySearch

            tool = TavilySearch(
                max_results=5,
                include_answer=True,
                include_raw_content=True,
                include_images=True,
                include_image_descriptions=True,
                # search_depth="advanced",
                # include_domains = []
                # exclude_domains = []
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'query': 'who won the last french open'})

        .. code-block:: json

            {
                "url": "https://www.nytimes.com...",
                "content": "Novak Djokovic won the last French Open by beating Casper Ruud ..."
            }

    Invoke with tool call:

        .. code-block:: python

            tool.invoke({"args": {'query': 'who won the last french open'}, "type": "tool_call", "id": "foo", "name": "tavily"})

        .. code-block:: python

            ToolMessage(
                content='{ "url": "https://www.nytimes.com...", "content": "Novak Djokovic won the last French Open by beating Casper Ruud ..." }',
                artifact={
                    'query': 'who won the last french open',
                    'follow_up_questions': None,
                    'answer': 'Novak ...',
                    'images': [
                        'https://www.amny.com/wp-content/uploads/2023/06/AP23162622181176-1200x800.jpg',
                        ...
                        ],
                    'results': [
                        {
                            'title': 'Djokovic ...',
                            'url': 'https://www.nytimes.com...',
                            'content': "Novak...",
                            'score': 0.99505633,
                            'raw_content': 'Tennis\nNovak ...'
                        },
                        ...
                    ],
                    'response_time': 2.92
                },
                tool_call_id='1',
                name='tavily_search_results_json',
            )

    """  # noqa: E501

    include_image_descriptions: bool = False  ## 是否在响应中包含图片描述，默认为False
    """Include a image descriptions in the response.

    Default is False.
    """

    api_wrapper: EnhancedTavilySearchAPIWrapper = Field(
        default_factory=EnhancedTavilySearchAPIWrapper
    )  ## 使用增强型Tavily搜索API包装器，默认自动初始化

    ## 同步运行搜索工具
    ## 参数：
    ##   query: 搜索查询字符串
    ##   run_manager: 可选的回调管理器
    ## 返回值：
    ##   元组，包含清理后的搜索结果JSON字符串和原始结果字典
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool."""
        # TODO: remove try/except, should be handled by BaseTool
        try:
            ## 调用增强型API包装器的同步搜索方法
            raw_results = self.api_wrapper.raw_results(
                query,  ## 搜索查询
                self.max_results,  ## 最大结果数量
                self.search_depth,  ## 搜索深度
                self.include_domains,  ## 包含的域名列表
                self.exclude_domains,  ## 排除的域名列表
                self.include_answer,  ## 是否包含答案
                self.include_raw_content,  ## 是否包含原始内容
                self.include_images,  ## 是否包含图片
                self.include_image_descriptions,  ## 是否包含图片描述
            )
        except Exception as e:
            ## 处理异常，记录错误日志
            logger.error("Tavily search returned error: {}".format(e))
            ## 构建错误结果JSON
            error_result = json.dumps({"error": repr(e)}, ensure_ascii=False)
            return error_result, {}
        ## 清理搜索结果，添加图片结果
        cleaned_results = self.api_wrapper.clean_results_with_images(raw_results)
        ## 记录调试日志
        logger.debug(
            "sync: %s", json.dumps(cleaned_results, indent=2, ensure_ascii=False)
        )
        ## 将清理后的结果转换为JSON字符串
        result_json = json.dumps(cleaned_results, ensure_ascii=False)
        return result_json, raw_results

    ## 异步运行搜索工具
    ## 参数：
    ##   query: 搜索查询字符串
    ##   run_manager: 可选的异步回调管理器
    ## 返回值：
    ##   元组，包含清理后的搜索结果JSON字符串和原始结果字典
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool asynchronously."""
        try:
            ## 调用增强型API包装器的异步搜索方法
            raw_results = await self.api_wrapper.raw_results_async(
                query,  ## 搜索查询
                self.max_results,  ## 最大结果数量
                self.search_depth,  ## 搜索深度
                self.include_domains,  ## 包含的域名列表
                self.exclude_domains,  ## 排除的域名列表
                self.include_answer,  ## 是否包含答案
                self.include_raw_content,  ## 是否包含原始内容
                self.include_images,  ## 是否包含图片
                self.include_image_descriptions,  ## 是否包含图片描述
            )
        except Exception as e:
            ## 处理异常，记录错误日志
            logger.error("Tavily search returned error: {}".format(e))
            ## 构建错误结果JSON
            error_result = json.dumps({"error": repr(e)}, ensure_ascii=False)
            return error_result, {}
        ## 清理搜索结果，添加图片结果
        cleaned_results = self.api_wrapper.clean_results_with_images(raw_results)
        ## 记录调试日志
        logger.debug(
            "async: %s", json.dumps(cleaned_results, indent=2, ensure_ascii=False)
        )
        ## 将清理后的结果转换为JSON字符串
        result_json = json.dumps(cleaned_results, ensure_ascii=False)
        return result_json, raw_results
