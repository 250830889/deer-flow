# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

## 服务器模块初始化文件
## 该文件定义了服务器模块的公共接口，导出关键组件供外部使用

## 从app模块导入FastAPI应用实例
## app是整个DeerFlow API服务器的核心组件，包含所有路由和中间件配置
from .app import app

## 定义模块的公共接口，仅导出app对象
## 这确保了外部只能访问模块明确导出的组件，保持了良好的封装性
__all__ = ["app"]
