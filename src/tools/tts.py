# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 文本转语音模块
## 使用火山引擎TTS API将文本转换为语音

import json  ## JSON处理模块，用于构建API请求和解析响应
import logging  ## 日志记录模块
import uuid  ## UUID生成模块，用于生成唯一请求ID
from typing import Any, Dict, Optional  ## 类型注解模块

import requests  ## HTTP请求模块，用于调用TTS API

logger = logging.getLogger(__name__)  ## 创建日志记录器


## 火山引擎TTS客户端类
## 用于调用火山引擎的文本转语音API，将文本转换为语音
class VolcengineTTS:
    """
    Client for volcengine Text-to-Speech API.
    """

    ## 初始化火山引擎TTS客户端
    ## 参数：
    ##   appid: 平台应用ID
    ##   access_token: 认证访问令牌
    ##   cluster: TTS集群名称，默认值为volcano_tts
    ##   voice_type: 使用的语音类型，默认值为BV700_V2_streaming
    ##   host: API主机，默认值为openspeech.bytedance.com
    def __init__(
        self,
        appid: str,
        access_token: str,
        cluster: str = "volcano_tts",
        voice_type: str = "BV700_V2_streaming",
        host: str = "openspeech.bytedance.com",
    ):
        """
        Initialize the volcengine TTS client.

        Args:
            appid: Platform application ID
            access_token: Access token for authentication
            cluster: TTS cluster name
            voice_type: Voice type to use
            host: API host
        """
        self.appid = appid  ## 平台应用ID
        self.access_token = access_token  ## 认证访问令牌
        self.cluster = cluster  ## TTS集群名称
        self.voice_type = voice_type  ## 使用的语音类型
        self.host = host  ## API主机
        self.api_url = f"https://{host}/api/v1/tts"  ## 构建完整的API URL
        self.header = {"Authorization": f"Bearer;{access_token}"}  ## 构建认证头部

    ## 将文本转换为语音
    ## 使用火山引擎TTS API将文本转换为语音
    ## 参数：
    ##   text: 要转换为语音的文本
    ##   encoding: 音频编码格式，默认值为mp3
    ##   speed_ratio: 语速比例，默认值为1.0
    ##   volume_ratio: 音量比例，默认值为1.0
    ##   pitch_ratio: 音调比例，默认值为1.0
    ##   text_type: 文本类型，默认值为plain（纯文本）
    ##   with_frontend: 是否使用前端处理，默认值为1（是）
    ##   frontend_type: 前端类型，默认值为unitTson
    ##   uid: 用户ID，默认值为None（自动生成）
    ## 返回值：
    ##   包含API响应和base64编码音频数据的字典
    def text_to_speech(
        self,
        text: str,
        encoding: str = "mp3",
        speed_ratio: float = 1.0,
        volume_ratio: float = 1.0,
        pitch_ratio: float = 1.0,
        text_type: str = "plain",
        with_frontend: int = 1,
        frontend_type: str = "unitTson",
        uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert text to speech using volcengine TTS API.

        Args:
            text: Text to convert to speech
            encoding: Audio encoding format
            speed_ratio: Speech speed ratio
            volume_ratio: Speech volume ratio
            pitch_ratio: Speech pitch ratio
            text_type: Text type (plain or ssml)
            with_frontend: Whether to use frontend processing
            frontend_type: Frontend type
            uid: User ID (generated if not provided)

        Returns:
            Dictionary containing the API response and base64-encoded audio data
        """
        ## 如果没有提供uid，则生成一个UUID
        if not uid:
            uid = str(uuid.uuid4())

        ## 构建API请求JSON数据
        request_json = {
            "app": {
                "appid": self.appid,  ## 应用ID
                "token": self.access_token,  ## 访问令牌
                "cluster": self.cluster,  ## 集群名称
            },
            "user": {"uid": uid},  ## 用户ID
            "audio": {
                "voice_type": self.voice_type,  ## 语音类型
                "encoding": encoding,  ## 音频编码格式
                "speed_ratio": speed_ratio,  ## 语速比例
                "volume_ratio": volume_ratio,  ## 音量比例
                "pitch_ratio": pitch_ratio,  ## 音调比例
            },
            "request": {
                "reqid": str(uuid.uuid4()),  ## 生成唯一请求ID
                "text": text,  ## 要转换的文本
                "text_type": text_type,  ## 文本类型
                "operation": "query",  ## 操作类型
                "with_frontend": with_frontend,  ## 是否使用前端处理
                "frontend_type": frontend_type,  ## 前端类型
            },
        }

        try:
            ## 清理文本，移除换行符
            sanitized_text = text.replace("\r\n", "").replace("\n", "")
            ## 记录TTS请求日志，只显示前50个字符
            logger.debug(f"Sending TTS request for text: {sanitized_text[:50]}...")
            ## 发送POST请求到TTS API
            response = requests.post(
                self.api_url, json.dumps(request_json), headers=self.header
            )
            ## 解析响应JSON
            response_json = response.json()

            ## 检查响应状态码
            if response.status_code != 200:
                logger.error(f"TTS API error: {response_json}")  ## 记录错误日志
                return {"success": False, "error": response_json, "audio_data": None}

            ## 检查响应中是否包含数据
            if "data" not in response_json:
                logger.error(f"TTS API returned no data: {response_json}")  ## 记录错误日志
                return {
                    "success": False,
                    "error": "No audio data returned",
                    "audio_data": None,
                }

            ## 返回成功响应，包含音频数据
            return {
                "success": True,
                "response": response_json,
                "audio_data": response_json["data"],  ## Base64编码的音频数据
            }

        except Exception as e:
            ## 捕获所有异常
            logger.exception(f"Error in TTS API call: {str(e)}")  ## 记录异常日志
            return {"success": False, "error": "TTS API call error", "audio_data": None}
