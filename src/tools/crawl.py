# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 网页爬取工具模块
## 该模块提供了一个用于爬取网页并返回markdown格式内容的工具

import json                                 ## JSON处理模块，用于格式化输出结果
import logging                              ## 日志记录模块，用于记录爬取过程和错误
from typing import Annotated                ## 类型注解模块，用于参数类型提示

from langchain_core.tools import tool       ## LangChain工具装饰器，用于定义工具

from src.crawler import Crawler             ## 导入自定义的网页爬虫类

from .decorators import log_io              ## 导入日志记录装饰器

logger = logging.getLogger(__name__)        ## 创建日志记录器


@tool                                       ## 标记为LangChain工具
@log_io                                     ## 添加输入输出日志记录
## 网页爬取工具函数
## 用于爬取指定URL的网页内容，并返回markdown格式的可读内容
## 参数：
##   url: 要爬取的网页URL字符串
## 返回值：
##   JSON格式的字符串，包含原始URL和爬取的markdown内容（最多1000个字符）
def crawl_tool(
    url: Annotated[str, "The url to crawl."],
) -> str:
    """Use this to crawl a url and get a readable content in markdown format."""
    try:
        ## 创建爬虫实例
        crawler = Crawler()
        ## 执行爬取操作，获取文章对象
        article = crawler.crawl(url)
        ## 将爬取结果转换为JSON格式，包含URL和markdown内容（限制1000字符）
        return json.dumps({"url": url, "crawled_content": article.to_markdown()[:1000]})
    except BaseException as e:
        ## 捕获所有异常，记录错误日志
        error_msg = f"Failed to crawl. Error: {repr(e)}"
        logger.error(error_msg)
        ## 返回错误信息
        return error_msg
