# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from src.rag.ingestion import _split_long_text, _split_text


def test_split_long_text_respects_chunk_size_and_overlap():
    text = "a" * 45
    chunks = _split_long_text(text, chunk_size=20, chunk_overlap=5)
    assert chunks[0] == "a" * 20
    assert chunks[1] == "a" * 20
    assert chunks[2] == "a" * 15
    for chunk in chunks:
        assert len(chunk) <= 20


def test_split_text_handles_paragraphs_and_overlap():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = _split_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk
        assert len(chunk) <= 30
