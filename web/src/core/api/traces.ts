// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { resolveServiceURL } from "./resolve-service-url";

export type TraceRun = {
  run_id: string;
  thread_id: string;
  status: string;
  created_at: string;
  finished_at?: string | null;
  title?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type TraceEvent = {
  id: number;
  ts: string;
  event_type: string;
  agent?: string | null;
  node?: string | null;
  step?: string | null;
  duration_ms?: number | null;
  token_usage?: Record<string, unknown> | null;
  payload?: Record<string, unknown> | null;
};

export async function fetchTraceRuns(
  limit = 50,
  offset = 0,
): Promise<TraceRun[]> {
  const url = resolveServiceURL(`traces/runs?limit=${limit}&offset=${offset}`);
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch trace runs: ${response.status}`);
  }
  const data = (await response.json()) as { runs: TraceRun[] };
  return data.runs ?? [];
}

export async function fetchTraceEvents(
  runId: string,
  sinceId = 0,
  limit = 500,
): Promise<TraceEvent[]> {
  const url = resolveServiceURL(
    `traces/runs/${runId}/events?since_id=${sinceId}&limit=${limit}`,
  );
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch trace events: ${response.status}`);
  }
  const data = (await response.json()) as { events: TraceEvent[] };
  return data.events ?? [];
}

export async function fetchAllTraceEvents(
  runId: string,
  pageSize = 500,
): Promise<TraceEvent[]> {
  const events: TraceEvent[] = [];
  let sinceId = 0;
  while (true) {
    const page = await fetchTraceEvents(runId, sinceId, pageSize);
    if (page.length === 0) {
      break;
    }
    events.push(...page);
    if (page.length < pageSize) {
      break;
    }
    sinceId = page[page.length - 1]!.id;
  }
  return events;
}
