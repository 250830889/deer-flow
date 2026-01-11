# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from src.rag.evaluation import EvaluationCase, evaluate_retriever
from src.rag.retriever import Document, Resource, Retriever


class DummyRetriever(Retriever):
    def __init__(self, docs_by_query: dict[str, list[Document]]):
        self.docs_by_query = docs_by_query

    def list_resources(self, query: str | None = None) -> list[Resource]:
        return []

    def query_relevant_documents(
        self, query: str, resources: list[Resource] = []
    ) -> list[Document]:
        return self.docs_by_query.get(query, [])


def test_evaluate_retriever_metrics():
    docs = [
        Document(id="doc-a", title="A", chunks=[]),
        Document(id="doc-b", title="B", chunks=[]),
    ]
    retriever = DummyRetriever({"q1": docs})
    case = EvaluationCase(query="q1", relevant_doc_ids=["doc-b"], relevant_sources=[])

    report = evaluate_retriever(retriever, [case], k_values=[1, 2])
    metrics = report["summary"]["metrics"]

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 1.0
    assert metrics["precision@1"] == 0.0
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr"] == 0.5
