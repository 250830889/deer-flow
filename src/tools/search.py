# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
搜索工具模块

此模块提供了多种搜索引擎的统一接口，支持配置不同的搜索引擎进行网络搜索。
支持的搜索引擎包括：Tavily、DuckDuckGo、Brave Search、ArXiv、Searx和Wikipedia。
通过配置文件可以灵活切换搜索引擎，并自定义搜索参数。

主要功能：
- 提供统一的搜索工具接口
- 支持多种搜索引擎配置
- 支持搜索结果日志记录
- 支持图片搜索和内容提取

使用示例：
    ```python
    # 获取配置的搜索工具
    search_tool = get_web_search_tool(max_search_results=10)
    
    # 执行搜索
    results = search_tool.run("Python编程教程")
    print(results)
    ```

注意事项：
- 使用某些搜索引擎（如Brave Search）需要设置相应的API密钥
- 搜索结果格式可能因搜索引擎而异
- 建议在生产环境中适当限制搜索频率以避免API配额耗尽
"""

import logging  ## 日志记录模块，用于记录搜索活动和错误信息
import os  ## 操作系统接口模块，用于获取环境变量
from typing import List, Optional  ## 类型注解模块，用于类型提示

## LangChain社区工具导入 - 各种搜索引擎工具
from langchain_community.tools import (
    BraveSearch,  ## Brave搜索引擎工具
    DuckDuckGoSearchResults,  ## DuckDuckGo搜索引擎工具
    SearxSearchRun,  ## Searx元搜索引擎工具
    WikipediaQueryRun,  ## Wikipedia查询工具
)
from langchain_community.tools.arxiv import ArxivQueryRun  ## ArXiv学术论文搜索引擎工具
## LangChain社区工具包装器导入 - 用于配置搜索引擎参数
from langchain_community.utilities import (
    ArxivAPIWrapper,  ## ArXiv API包装器
    BraveSearchWrapper,  ## Brave搜索API包装器
    SearxSearchWrapper,  ## Searx搜索API包装器
    WikipediaAPIWrapper,  ## Wikipedia API包装器
)

## 项目内部模块导入
from src.config import SELECTED_SEARCH_ENGINE, SearchEngine, load_yaml_config  ## 搜索引擎配置和枚举
from src.tools.decorators import create_logged_tool  ## 创建带日志记录的工具装饰器
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchWithImages,  ## Tavily搜索引擎工具（支持图片搜索）
)

## 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

## 创建带日志记录的搜索工具实例
## 这些工具包装了原始搜索引擎工具，添加了日志记录功能，便于跟踪搜索活动
LoggedTavilySearch = create_logged_tool(TavilySearchWithImages)  ## 带日志的Tavily搜索工具（支持图片搜索）
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)  ## 带日志的DuckDuckGo搜索工具
LoggedBraveSearch = create_logged_tool(BraveSearch)  ## 带日志的Brave搜索工具
LoggedArxivSearch = create_logged_tool(ArxivQueryRun)  ## 带日志的ArXiv学术论文搜索工具
LoggedSearxSearch = create_logged_tool(SearxSearchRun)  ## 带日志的Searx元搜索工具
LoggedWikipediaSearch = create_logged_tool(WikipediaQueryRun)  ## 带日志的Wikipedia查询工具


def get_search_config():
    """
    从配置文件中加载搜索引擎配置
    
    此函数从项目的配置文件(conf.yaml)中读取搜索引擎相关的配置参数，
    包括搜索引擎类型、搜索参数设置等。如果配置文件中没有找到相关配置，
    将返回空字典。
    
    返回:
        dict: 包含搜索引擎配置的字典，可能包含以下键：
            - include_domains: 包含的域名列表
            - exclude_domains: 排除的域名列表
            - include_answer: 是否包含搜索结果摘要
            - search_depth: 搜索深度(basic/advanced)
            - include_raw_content: 是否包含原始内容
            - include_images: 是否包含图片
            - include_image_descriptions: 是否包含图片描述
            - wikipedia_lang: Wikipedia语言设置
            - wikipedia_doc_content_chars_max: Wikipedia文档最大字符数
    
    使用示例:
        ```python
        config = get_search_config()
        search_depth = config.get("search_depth", "basic")
        ```
    
    注意事项:
        - 配置文件路径是相对于项目根目录的"conf.yaml"
        - 如果配置文件不存在或格式错误，可能会抛出异常
        - 建议在调用此函数前确保配置文件存在且格式正确
    """
    ## 加载YAML配置文件
    config = load_yaml_config("conf.yaml")
    ## 获取搜索引擎配置部分，如果不存在则返回空字典
    search_config = config.get("SEARCH_ENGINE", {})
    return search_config


def get_web_search_tool(max_search_results: int):
    """
    获取配置的网络搜索工具实例
    
    根据配置文件中指定的搜索引擎类型，返回相应的搜索工具实例。
    支持的搜索引擎包括：Tavily、DuckDuckGo、Brave Search、ArXiv、Searx和Wikipedia。
    每个搜索引擎都有特定的配置参数，可以从配置文件中读取。
    
    参数:
        max_search_results (int): 最大搜索结果数量，控制返回结果的数量上限
    
    返回:
        搜索工具实例，具体类型取决于配置的搜索引擎。所有返回的工具都带有日志记录功能。
    
    异常:
        ValueError: 当配置的搜索引擎不受支持时抛出
    
    使用示例:
        ```python
        # 获取配置的搜索工具，限制返回10个结果
        search_tool = get_web_search_tool(max_search_results=10)
        
        # 执行搜索
        results = search_tool.run("Python编程教程")
        ```
    
    注意事项:
        - 使用Brave Search需要设置环境变量BRAVE_SEARCH_API_KEY
        - 不同搜索引擎的返回结果格式可能不同
        - 某些搜索引擎可能有API调用频率限制
        - 搜索工具的具体参数可以通过配置文件进行调整
    """
    ## 获取搜索引擎配置
    search_config = get_search_config()

    ## 根据配置的搜索引擎类型创建相应的搜索工具
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        ## Tavily搜索引擎配置
        ## 从配置文件中获取Tavily搜索参数，如果未配置则使用默认值
        include_domains: Optional[List[str]] = search_config.get("include_domains", [])  ## 包含的域名列表
        exclude_domains: Optional[List[str]] = search_config.get("exclude_domains", [])  ## 排除的域名列表
        include_answer: bool = search_config.get("include_answer", False)  ## 是否包含搜索结果摘要
        search_depth: str = search_config.get("search_depth", "advanced")  ## 搜索深度：basic或advanced
        include_raw_content: bool = search_config.get("include_raw_content", True)  ## 是否包含原始内容
        include_images: bool = search_config.get("include_images", True)  ## 是否包含图片
        include_image_descriptions: bool = include_images and search_config.get(
            "include_image_descriptions", True
        )  ## 是否包含图片描述（仅在包含图片时有效）

        ## 记录Tavily搜索配置信息
        logger.info(
            f"Tavily search configuration loaded: include_domains={include_domains}, "
            f"exclude_domains={exclude_domains}, include_answer={include_answer}, "
            f"search_depth={search_depth}, include_raw_content={include_raw_content}, "
            f"include_images={include_images}, include_image_descriptions={include_image_descriptions}"
        )

        ## 创建并返回Tavily搜索工具实例
        return LoggedTavilySearch(
            name="web_search",
            max_results=max_search_results,  ## 设置最大返回结果数量
            include_answer=include_answer,  ## 是否包含搜索结果摘要
            search_depth=search_depth,  ## 设置搜索深度
            include_raw_content=include_raw_content,  ## 是否包含原始内容
            include_images=include_images,  ## 是否包含图片
            include_image_descriptions=include_image_descriptions,  ## 是否包含图片描述
            include_domains=include_domains,  ## 设置包含的域名列表
            exclude_domains=exclude_domains,  ## 设置排除的域名列表
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        ## DuckDuckGo搜索引擎配置
        return LoggedDuckDuckGoSearch(
            name="web_search",
            num_results=max_search_results,  ## 设置返回结果数量
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        ## Brave搜索引擎配置，需要API密钥
        return LoggedBraveSearch(
            name="web_search",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),  ## 从环境变量获取API密钥
                search_kwargs={"count": max_search_results},  ## 设置搜索结果数量
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        ## ArXiv学术论文搜索引擎配置
        return LoggedArxivSearch(
            name="web_search",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,  ## 设置返回结果数量
                load_max_docs=max_search_results,  ## 设置加载文档数量
                load_all_available_meta=True,  ## 加载所有可用元数据
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.SEARX.value:
        ## Searx元搜索引擎配置
        return LoggedSearxSearch(
            name="web_search",
            wrapper=SearxSearchWrapper(
                k=max_search_results,  ## 设置返回结果数量
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.WIKIPEDIA.value:
        ## Wikipedia搜索引擎配置
        wiki_lang = search_config.get("wikipedia_lang", "en")  ## Wikipedia语言设置，默认英语
        wiki_doc_content_chars_max = search_config.get(
            "wikipedia_doc_content_chars_max", 4000
        )  ## 文档最大字符数，默认4000
        
        return LoggedWikipediaSearch(
            name="web_search",
            api_wrapper=WikipediaAPIWrapper(
                lang=wiki_lang,  ## 设置Wikipedia语言
                top_k_results=max_search_results,  ## 设置返回结果数量
                load_all_available_meta=True,  ## 加载所有可用元数据
                doc_content_chars_max=wiki_doc_content_chars_max,  ## 设置文档最大字符数
            ),
        )
    else:
        ## 不支持的搜索引擎类型
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")
