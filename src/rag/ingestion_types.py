# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngestedChunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestedDocument:
    document_id: str
    title: str | None
    url: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[IngestedChunk] = field(default_factory=list)
