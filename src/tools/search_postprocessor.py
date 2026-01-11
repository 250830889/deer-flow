# src/tools/search_postprocessor.py
## 搜索结果后处理工具模块
## 提供用于处理和优化搜索结果的功能，包括去重、过滤低质量结果、清理图片和截断长内容

import base64  ## Base64编码模块，用于处理图片数据
import logging  ## 日志记录模块
import re  ## 正则表达式模块，用于匹配和替换
from typing import Any, Dict, List  ## 类型注解模块
from urllib.parse import urlparse  ## URL解析模块

logger = logging.getLogger(__name__)  ## 创建日志记录器


## 搜索结果后处理器类
## 用于对搜索结果进行各种优化处理，提高搜索结果的质量和可用性
class SearchResultPostProcessor:
    """Search result post-processor"""

    ## Base64图片正则表达式模式，用于匹配和清理Base64编码的图片数据
    base64_pattern = r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+"

    ## 初始化后处理器
    ## 参数：
    ##   min_score_threshold: 最低相关性得分阈值，低于此值的结果将被过滤
    ##   max_content_length_per_page: 每页内容的最大长度，超过此长度的内容将被截断
    def __init__(self, min_score_threshold: float, max_content_length_per_page: int):
        """
        Initialize the post-processor

        Args:
            min_score_threshold: Minimum relevance score threshold
            max_content_length_per_page: Maximum content length
        """
        self.min_score_threshold = min_score_threshold  ## 最低得分阈值
        self.max_content_length_per_page = max_content_length_per_page  ## 每页最大内容长度

    ## 处理搜索结果
    ## 对搜索结果进行一系列优化处理，包括去重、过滤低质量结果、清理图片和截断长内容
    ## 参数：
    ##   results: 原始搜索结果列表
    ## 返回值：
    ##   处理后的结果列表
    def process_results(self, results: List[Dict]) -> List[Dict]:
        """
        Process search results

        Args:
            results: Original search result list

        Returns:
            Processed result list
        """
        if not results:
            return []

        ## 单循环组合处理，提高效率
        cleaned_results = []  ## 清理后的结果列表
        seen_urls = set()  ## 用于去重的URL集合

        for result in results:
            ## 1. 移除重复结果
            cleaned_result = self._remove_duplicates(result, seen_urls)
            if not cleaned_result:
                continue

            ## 2. 过滤低质量结果
            if (
                "page" == cleaned_result.get("type")
                and self.min_score_threshold
                and self.min_score_threshold > 0
                and cleaned_result.get("score", 0) < self.min_score_threshold
            ):
                continue

            ## 3. 清理内容中的base64图片
            cleaned_result = self._remove_base64_images(cleaned_result)
            if not cleaned_result:
                continue

            ## 4. 当设置了max_content_length_per_page时，截断长内容
            if (
                self.max_content_length_per_page
                and self.max_content_length_per_page > 0
            ):
                cleaned_result = self._truncate_long_content(cleaned_result)

            if cleaned_result:
                cleaned_results.append(cleaned_result)

        ## 5. 排序（按得分降序）
        sorted_results = sorted(
            cleaned_results, key=lambda x: x.get("score", 0), reverse=True
        )

        ## 记录处理前后的结果数量
        logger.info(
            f"Search result post-processing: {len(results)} -> {len(sorted_results)}"
        )
        return sorted_results

    ## 移除结果中的base64图片
    ## 根据结果类型调用不同的处理方法
    ## 参数：
    ##   result: 单个搜索结果
    ## 返回值：
    ##   清理后的结果
    def _remove_base64_images(self, result: Dict) -> Dict:
        """Remove base64 encoded images from content"""

        if "page" == result.get("type"):
            cleaned_result = self.processPage(result)  ## 处理页面类型结果
        elif "image" == result.get("type"):
            cleaned_result = self.processImage(result)  ## 处理图片类型结果
        else:
            ## 对于其他类型，保持不变
            cleaned_result = result.copy()

        return cleaned_result

    ## 处理页面类型结果
    ## 清理页面内容中的base64图片
    ## 参数：
    ##   result: 页面类型的搜索结果
    ## 返回值：
    ##   清理后的页面结果
    def processPage(self, result: Dict) -> Dict:
        """Process page type result"""
        ## 复制结果以避免修改原始数据
        cleaned_result = result.copy()

        ## 清理content字段中的base64图片
        if "content" in result:
            original_content = result["content"]  ## 原始内容
            cleaned_content = re.sub(self.base64_pattern, " ", original_content)  ## 替换base64图片为空格
            cleaned_result["content"] = cleaned_content  ## 更新清理后的内容

            ## 如果内容减少了20%以上，记录日志
            if len(cleaned_content) < len(original_content) * 0.8:
                logger.debug(
                    f"Removed base64 images from search content: {result.get('url', 'unknown')}"
                )

        ## 清理raw_content字段中的base64图片
        if "raw_content" in cleaned_result:
            original_raw_content = cleaned_result["raw_content"]  ## 原始原始内容
            cleaned_raw_content = re.sub(self.base64_pattern, " ", original_raw_content)  ## 替换base64图片为空格
            cleaned_result["raw_content"] = cleaned_raw_content  ## 更新清理后的原始内容

            ## 如果原始内容减少了20%以上，记录日志
            if len(cleaned_raw_content) < len(original_raw_content) * 0.8:
                logger.debug(
                    f"Removed base64 images from search raw content: {result.get('url', 'unknown')}"
                )

        return cleaned_result

    ## 处理图片类型结果
    ## 清理图片URL中的base64数据和截断过长的图片描述
    ## 参数：
    ##   result: 图片类型的搜索结果
    ## 返回值：
    ##   清理后的图片结果，或空字典（如果处理失败）
    def processImage(self, result: Dict) -> Dict:
        """Process image type result - clean up base64 data and long fields"""
        ## 复制结果以避免修改原始数据
        cleaned_result = result.copy()

        ## 从image_url中移除base64编码数据
        if "image_url" in cleaned_result and isinstance(
            cleaned_result["image_url"], str
        ):
            ## 检查image_url是否包含base64数据
            if "data:image" in cleaned_result["image_url"]:
                original_image_url = cleaned_result["image_url"]  ## 原始图片URL
                cleaned_image_url = re.sub(self.base64_pattern, " ", original_image_url)  ## 替换base64图片为空格
                ## 检查清理后的URL是否为空或不是http开头
                if len(cleaned_image_url) == 0 or not cleaned_image_url.startswith(
                    "http"
                ):
                    logger.debug(
                        f"Removed base64 data from image_url and the cleaned_image_url is empty or not start with http, origin image_url: {result.get('image_url', 'unknown')}"
                    )
                    return {}  ## 返回空字典表示处理失败
                cleaned_result["image_url"] = cleaned_image_url  ## 更新清理后的图片URL
                logger.debug(
                    f"Removed base64 data from image_url: {result.get('image_url', 'unknown')}"
                )

        ## 截断非常长的图片描述
        if "image_description" in cleaned_result and isinstance(
            cleaned_result["image_description"], str
        ):
            if (
                self.max_content_length_per_page
                and len(cleaned_result["image_description"])
                > self.max_content_length_per_page
            ):
                ## 截断图片描述并添加省略号
                cleaned_result["image_description"] = (
                    cleaned_result["image_description"][
                        : self.max_content_length_per_page
                    ]
                    + "..."
                )
                logger.info(
                    f"Truncated long image description from search result: {result.get('image_url', 'unknown')}"
                )

        return cleaned_result

    ## 截断长内容
    ## 对搜索结果中的长内容进行截断处理，避免内容过长
    ## 参数：
    ##   result: 单个搜索结果
    ## 返回值：
    ##   截断后的结果
    def _truncate_long_content(self, result: Dict) -> Dict:
        """Truncate long content"""

        truncated_result = result.copy()  ## 复制结果以避免修改原始数据

        ## 截断content字段长度
        if "content" in truncated_result:
            content = truncated_result["content"]  ## 原始内容
            if len(content) > self.max_content_length_per_page:
                ## 截断内容并添加省略号
                truncated_result["content"] = (
                    content[: self.max_content_length_per_page] + "..."
                )
                logger.info(
                    f"Truncated long content from search result: {result.get('url', 'unknown')}"
                )

        ## 截断raw_content字段长度（可以稍长一些，设置为2倍）
        if "raw_content" in truncated_result:
            raw_content = truncated_result["raw_content"]  ## 原始原始内容
            if len(raw_content) > self.max_content_length_per_page * 2:
                ## 截断原始内容并添加省略号
                truncated_result["raw_content"] = (
                    raw_content[: self.max_content_length_per_page * 2] + "..."
                )
                logger.info(
                    f"Truncated long raw content from search result: {result.get('url', 'unknown')}"
                )

        return truncated_result

    ## 移除重复结果
    ## 根据URL去重，确保每个URL只保留一个结果
    ## 参数：
    ##   result: 单个搜索结果
    ##   seen_urls: 已处理的URL集合
    ## 返回值：
    ##   去重后的结果，或None（如果是重复结果）
    def _remove_duplicates(self, result: Dict, seen_urls: set) -> Dict:
        """Remove duplicate results"""

        url = result.get("url")  ## 获取结果的URL
        if not url:
            ## 如果没有URL，尝试从image_url获取
            image_url_val = result.get("image_url", "")
            if isinstance(image_url_val, dict):
                url = image_url_val.get("url", "")
            else:
                url = image_url_val

        if url and url not in seen_urls:
            seen_urls.add(url)  ## 将URL添加到已处理集合
            return result.copy()  ## 返回结果副本
        elif not url:
            ## 保留没有URL的结果
            return result.copy()  ## 返回结果副本

        return {}  ## 返回空字典表示重复结果
