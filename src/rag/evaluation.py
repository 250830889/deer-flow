# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.rag.builder import build_retriever
from src.rag.retriever import Document, Resource, Retriever


@dataclass
class EvaluationCase:
    query: str
    relevant_doc_ids: list[str]
    relevant_sources: list[str]
    resources: list[Resource] | None = None


def load_cases(path: str) -> list[EvaluationCase]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        records = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    else:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            records = raw.get("cases", [])
        else:
            records = raw

    cases: list[EvaluationCase] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        query = record.get("query") or record.get("question")
        if not query:
            continue
        relevant_docs = (
            record.get("relevant_documents")
            or record.get("relevant_docs")
            or record.get("relevant_doc_ids")
            or record.get("relevant")
            or []
        )
        relevant_docs = [str(doc_id) for doc_id in relevant_docs if doc_id]
        relevant_sources = (
            record.get("relevant_sources")
            or record.get("relevant_uris")
            or record.get("relevant_paths")
            or []
        )
        relevant_sources = [str(source) for source in relevant_sources if source]
        resources = _parse_resources(record)
        cases.append(
            EvaluationCase(
                query=str(query).strip(),
                relevant_doc_ids=relevant_docs,
                relevant_sources=relevant_sources,
                resources=resources,
            )
        )
    return cases


def _parse_resources(record: dict) -> list[Resource] | None:
    if "resources" in record and isinstance(record["resources"], list):
        resources = []
        for item in record["resources"]:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            title = item.get("title") or uri or "resource"
            if uri:
                resources.append(
                    Resource(uri=str(uri), title=str(title), description=item.get("description", ""))
                )
        return resources or None
    if "resource_uris" in record and isinstance(record["resource_uris"], list):
        return [
            Resource(uri=str(uri), title=str(uri), description="")
            for uri in record["resource_uris"]
            if uri
        ]
    return None


def evaluate_retriever(
    retriever: Retriever,
    cases: Iterable[EvaluationCase],
    k_values: list[int],
) -> dict:
    summary_totals = {f"recall@{k}": 0.0 for k in k_values}
    summary_totals.update({f"precision@{k}": 0.0 for k in k_values})
    summary_totals.update({f"ndcg@{k}": 0.0 for k in k_values})
    summary_totals["mrr"] = 0.0
    total_cases = 0
    case_reports = []

    for case in cases:
        total_cases += 1
        start = time.perf_counter()
        documents = retriever.query_relevant_documents(
            case.query, case.resources or []
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        ranked_docs = _extract_ranked_docs(documents)
        relevant_set = set(case.relevant_doc_ids + case.relevant_sources)
        metrics = _compute_metrics(ranked_docs, relevant_set, k_values)
        for metric, value in metrics.items():
            summary_totals[metric] += value
        case_reports.append(
            {
                "query": case.query,
                "relevant_doc_ids": case.relevant_doc_ids,
                "relevant_sources": case.relevant_sources,
                "retrieved_docs": ranked_docs,
                "metrics": metrics,
                "latency_ms": latency_ms,
            }
        )

    summary = {
        "cases": total_cases,
        "metrics": {
            metric: (summary_totals[metric] / total_cases if total_cases else 0.0)
            for metric in summary_totals
        },
    }
    return {"summary": summary, "cases": case_reports}


def _extract_ranked_docs(documents: list[Document]) -> list[dict]:
    ranked = []
    for doc in documents:
        ranked.append(
            {
                "id": str(doc.id),
                "url": doc.url,
                "metadata": doc.metadata,
            }
        )
    return ranked


def _compute_metrics(
    ranked_docs: list[dict],
    relevant_set: set[str],
    k_values: list[int],
) -> dict:
    ranked_hits = [_doc_matches(doc, relevant_set) for doc in ranked_docs]
    relevant_count = len(relevant_set)
    metrics = {}
    for k in k_values:
        hits = sum(1 for hit in ranked_hits[:k] if hit)
        recall = hits / relevant_count if relevant_count else 0.0
        precision = hits / k if k else 0.0
        metrics[f"recall@{k}"] = recall
        metrics[f"precision@{k}"] = precision
        metrics[f"ndcg@{k}"] = _ndcg_at_k(ranked_hits, k, relevant_count)
    metrics["mrr"] = _mrr(ranked_hits)
    return metrics


def _mrr(ranked_hits: list[bool]) -> float:
    for index, hit in enumerate(ranked_hits, start=1):
        if hit:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(
    ranked_hits: list[bool],
    k: int,
    relevant_count: int,
) -> float:
    if relevant_count <= 0:
        return 0.0
    gains = [1.0 if hit else 0.0 for hit in ranked_hits[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted([1.0] * min(relevant_count, k), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg else 0.0


def _doc_matches(doc: dict, relevant_set: set[str]) -> bool:
    if not relevant_set:
        return False
    identifiers = set()
    doc_id = doc.get("id")
    if doc_id:
        identifiers.add(str(doc_id))
    url = doc.get("url")
    if url:
        identifiers.add(str(url))
    metadata = doc.get("metadata") or {}
    for key in ("document_id", "source_uri", "source_path", "file_name"):
        value = metadata.get(key)
        if value:
            identifiers.add(str(value))
    return bool(identifiers & relevant_set)


def render_markdown_report(report: dict) -> str:
    summary = report.get("summary", {})
    metrics = summary.get("metrics", {})
    lines = [
        "# RAG Retrieval Evaluation Report",
        "",
        f"- Cases: {summary.get('cases', 0)}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for metric, value in sorted(metrics.items()):
        lines.append(f"| {metric} | {value:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--cases", required=True, help="Path to JSON/JSONL cases file")
    parser.add_argument(
        "--k",
        default="1,3,5,10",
        help="Comma-separated k values for metrics",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional Markdown report output path",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of cases",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]

    retriever = build_retriever()
    if not retriever:
        raise RuntimeError("No RAG provider configured for evaluation.")

    k_values = sorted(
        {int(k.strip()) for k in args.k.split(",") if k.strip().isdigit()}
    )
    report = evaluate_retriever(retriever, cases, k_values)
    markdown = render_markdown_report(report)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(markdown)


if __name__ == "__main__":
    main()
