# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 增强型Tavily搜索API包装器模块
## 提供对Tavily搜索API的增强封装，支持同步和异步搜索，以及带有图片的搜索结果处理

import json  ## JSON处理模块，用于解析API响应
from typing import Dict, List, Optional  ## 类型注解模块

import aiohttp  ## 异步HTTP客户端库，用于异步API调用
import requests  ## HTTP客户端库，用于同步API调用
from langchain_tavily._utilities import TAVILY_API_URL  ## Tavily API URL常量
from langchain_tavily.tavily_search import (
    TavilySearchAPIWrapper as OriginalTavilySearchAPIWrapper,  ## 原始Tavily搜索API包装器
)

from src.config import load_yaml_config  ## 配置加载函数
from src.tools.search_postprocessor import SearchResultPostProcessor  ## 搜索结果后处理器


## 获取搜索配置
## 从配置文件中加载搜索引擎相关配置
## 返回值：
##   包含搜索引擎配置的字典
def get_search_config():
    config = load_yaml_config("conf.yaml")  ## 加载YAML配置文件
    search_config = config.get("SEARCH_ENGINE", {})  ## 获取搜索引擎配置部分
    return search_config


## 增强型Tavily搜索API包装器类
## 继承自原始的TavilySearchAPIWrapper，添加了增强功能
## 支持同步和异步搜索，以及带有图片的搜索结果处理
class EnhancedTavilySearchAPIWrapper(OriginalTavilySearchAPIWrapper):
    ## 同步获取Tavily搜索API结果
    ## 参数：
    ##   query: 搜索查询字符串
    ##   max_results: 最大返回结果数量，默认值为5
    ##   search_depth: 搜索深度，默认值为advanced
    ##   include_domains: 包含的域名列表，默认值为空列表
    ##   exclude_domains: 排除的域名列表，默认值为空列表
    ##   include_answer: 是否包含搜索答案，默认值为False
    ##   include_raw_content: 是否包含原始内容，默认值为False
    ##   include_images: 是否包含图片，默认值为False
    ##   include_image_descriptions: 是否包含图片描述，默认值为False
    ## 返回值：
    ##   包含搜索结果的字典
    def raw_results(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
        include_image_descriptions: Optional[bool] = False,
    ) -> Dict:
        ## 构建API请求参数
        params = {
            "api_key": self.tavily_api_key.get_secret_value(),  ## 获取API密钥
            "query": query,  ## 搜索查询
            "max_results": max_results,  ## 最大返回结果数量
            "search_depth": search_depth,  ## 搜索深度
            "include_domains": include_domains,  ## 包含的域名列表
            "exclude_domains": exclude_domains,  ## 排除的域名列表
            "include_answer": include_answer,  ## 是否包含搜索答案
            "include_raw_content": include_raw_content,  ## 是否包含原始内容
            "include_images": include_images,  ## 是否包含图片
            "include_image_descriptions": include_image_descriptions,  ## 是否包含图片描述
        }
        ## 发送同步POST请求
        response = requests.post(
            f"{TAVILY_API_URL}/search",  ## Tavily搜索API URL
            json=params,  ## 请求参数
        )
        response.raise_for_status()  ## 检查响应状态，如有错误则抛出异常
        return response.json()  ## 返回响应JSON数据

    ## 异步获取Tavily搜索API结果
    ## 参数：
    ##   与raw_results方法相同
    ## 返回值：
    ##   包含搜索结果的字典
    async def raw_results_async(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
        include_image_descriptions: Optional[bool] = False,
    ) -> Dict:
        """Get results from the Tavily Search API asynchronously."""

        ## 异步API调用辅助函数
        async def fetch() -> str:
            ## 构建API请求参数
            params = {
                "api_key": self.tavily_api_key.get_secret_value(),  ## 获取API密钥
                "query": query,  ## 搜索查询
                "max_results": max_results,  ## 最大返回结果数量
                "search_depth": search_depth,  ## 搜索深度
                "include_domains": include_domains,  ## 包含的域名列表
                "exclude_domains": exclude_domains,  ## 排除的域名列表
                "include_answer": include_answer,  ## 是否包含搜索答案
                "include_raw_content": include_raw_content,  ## 是否包含原始内容
                "include_images": include_images,  ## 是否包含图片
                "include_image_descriptions": include_image_descriptions,  ## 是否包含图片描述
            }
            ## 创建异步HTTP会话
            async with aiohttp.ClientSession(trust_env=True) as session:
                ## 发送异步POST请求
                async with session.post(f"{TAVILY_API_URL}/search", json=params) as res:
                    if res.status == 200:  ## 检查响应状态码
                        data = await res.text()  ## 读取响应文本
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")  ## 抛出异常

        results_json_str = await fetch()  ## 调用异步辅助函数
        return json.loads(results_json_str)  ## 解析JSON响应

    ## 清理带有图片的搜索结果
    ## 将原始搜索结果转换为统一格式，并添加图片结果
    ## 参数：
    ##   raw_results: 原始搜索结果字典
    ## 返回值：
    ##   清理后的搜索结果列表
    def clean_results_with_images(
        self, raw_results: Dict[str, List[Dict]]
    ) -> List[Dict]:
        results = raw_results["results"]  ## 获取网页搜索结果
        clean_results = []  ## 初始化清理后的结果列表
        ## 处理网页搜索结果
        for result in results:
            ## 构建清理后的结果字典
            clean_result = {
                "type": "page",  ## 结果类型为网页
                "title": result["title"],  ## 网页标题
                "url": result["url"],  ## 网页URL
                "content": result["content"],  ## 网页内容
                "score": result["score"],  ## 相关性得分
            }
            ## 如果有原始内容，则添加到清理结果中
            if raw_content := result.get("raw_content"):
                clean_result["raw_content"] = raw_content
            clean_results.append(clean_result)  ## 添加到结果列表
        
        ## 处理图片搜索结果
        images = raw_results["images"]  ## 获取图片搜索结果
        for image in images:
            ## 构建清理后的图片结果字典
            clean_result = {
                "type": "image_url",  ## 结果类型为图片
                "image_url": {"url": image["url"]},  ## 图片URL
                "image_description": image["description"],  ## 图片描述
            }
            clean_results.append(clean_result)  ## 添加到结果列表

        ## 获取搜索配置
        search_config = get_search_config()
        ## 使用搜索结果后处理器处理结果
        clean_results = SearchResultPostProcessor(
            min_score_threshold=search_config.get("min_score_threshold"),  ## 最低得分阈值
            max_content_length_per_page=search_config.get(
                "max_content_length_per_page"
            ),  ## 每页最大内容长度
        ).process_results(clean_results)  ## 执行结果处理

        return clean_results  ## 返回清理后的结果列表
