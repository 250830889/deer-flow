// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import {
  Background,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Brain,
  FilePen,
  MessageSquareQuote,
  Microscope,
  SquareTerminal,
  UserCheck,
  Users,
} from "lucide-react";
import { useTheme } from "next-themes";
import type { ComponentType } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Tooltip } from "~/components/deer-flow/tooltip";
import { ShineBorder } from "~/components/magicui/shine-border";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { ScrollArea } from "~/components/ui/scroll-area";
import {
  fetchAllTraceEvents,
  fetchTraceEvents,
  fetchTraceRuns,
} from "~/core/api";
import type { TraceEvent, TraceRun } from "~/core/api/traces";
import { cn } from "~/lib/utils";

type GraphNode = Node<{
  label: string;
  icon?: ComponentType<{ className?: string }>;
  active?: boolean;
  visited?: boolean;
}>;

const ROW_HEIGHT = 80;

const baseNodes: GraphNode[] = [
  {
    id: "start",
    type: "circle",
    data: { label: "Start" },
    position: { x: -140, y: 0 },
  },
  {
    id: "coordinator",
    data: { icon: MessageSquareQuote, label: "Coordinator" },
    position: { x: 80, y: 0 },
  },
  {
    id: "background_investigator",
    data: { icon: Brain, label: "Background" },
    position: { x: 320, y: 0 },
  },
  {
    id: "planner",
    data: { icon: Brain, label: "Planner" },
    position: { x: 80, y: ROW_HEIGHT },
  },
  {
    id: "human_feedback",
    data: { icon: UserCheck, label: "Human Feedback" },
    position: { x: -30, y: ROW_HEIGHT * 2 },
  },
  {
    id: "research_team",
    data: { icon: Users, label: "Research Team" },
    position: { x: -30, y: ROW_HEIGHT * 3 },
  },
  {
    id: "researcher",
    data: { icon: Microscope, label: "Researcher" },
    position: { x: -140, y: ROW_HEIGHT * 4 },
  },
  {
    id: "coder",
    data: { icon: SquareTerminal, label: "Coder" },
    position: { x: 40, y: ROW_HEIGHT * 4 },
  },
  {
    id: "reporter",
    data: { icon: FilePen, label: "Reporter" },
    position: { x: 320, y: ROW_HEIGHT * 2 },
  },
  {
    id: "end",
    type: "circle",
    data: { label: "End" },
    position: { x: 460, y: ROW_HEIGHT * 4 },
  },
];

const edges: Edge[] = [
  {
    id: "start->coordinator",
    source: "start",
    target: "coordinator",
    sourceHandle: "right",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "coordinator->background",
    source: "coordinator",
    target: "background_investigator",
    sourceHandle: "right",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "background->planner",
    source: "background_investigator",
    target: "planner",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "planner->human",
    source: "planner",
    target: "human_feedback",
    sourceHandle: "left",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "human->planner",
    source: "human_feedback",
    target: "planner",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: true,
  },
  {
    id: "human->team",
    source: "human_feedback",
    target: "research_team",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "planner->reporter",
    source: "planner",
    target: "reporter",
    sourceHandle: "right",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "reporter->end",
    source: "reporter",
    target: "end",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "team->researcher",
    source: "research_team",
    target: "researcher",
    sourceHandle: "left",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "team->coder",
    source: "research_team",
    target: "coder",
    sourceHandle: "right",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "team->planner",
    source: "research_team",
    target: "planner",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: true,
  },
  {
    id: "researcher->team",
    source: "researcher",
    target: "research_team",
    sourceHandle: "right",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "coder->team",
    source: "coder",
    target: "research_team",
    sourceHandle: "top",
    targetHandle: "right",
    animated: true,
  },
];

const nodeTypes = {
  circle: CircleNode,
  agent: AgentNode,
  default: AgentNode,
};

export default function ObservabilityMain() {
  const { resolvedTheme } = useTheme();
  const [runs, setRuns] = useState<TraceRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [mode, setMode] = useState<"live" | "replay">("live");
  const [replayEvents, setReplayEvents] = useState<TraceEvent[]>([]);
  const [replayCursor, setReplayCursor] = useState(0);
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [isReplaying, setIsReplaying] = useState(false);
  const [isReplayLoading, setIsReplayLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastEventIdRef = useRef(0);
  const replayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refreshRuns = useCallback(async () => {
    try {
      const data = await fetchTraceRuns();
      setRuns(data);
      if (!selectedRunId && data.length > 0) {
        setSelectedRunId(data[0].run_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load runs");
    }
  }, [selectedRunId]);

  const stopReplayTimer = useCallback(() => {
    if (replayTimerRef.current) {
      window.clearTimeout(replayTimerRef.current);
      replayTimerRef.current = null;
    }
  }, []);

  const resetReplay = useCallback(() => {
    stopReplayTimer();
    setIsReplaying(false);
    setReplayCursor(0);
  }, [stopReplayTimer]);

  const loadReplayEvents = useCallback(async (runId: string) => {
    setIsReplayLoading(true);
    setError(null);
    setReplayEvents([]);
    try {
      const data = await fetchAllTraceEvents(runId, 1000);
      setReplayEvents(data);
      setReplayCursor(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load replay events");
    } finally {
      setIsReplayLoading(false);
    }
  }, []);

  const refreshEvents = useCallback(
    async (runId: string) => {
      try {
        const data = await fetchTraceEvents(runId, lastEventIdRef.current);
        if (data.length > 0) {
          lastEventIdRef.current = data[data.length - 1]!.id;
          setEvents((prev) => [...prev, ...data]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load events");
      }
    },
    [],
  );

  useEffect(() => {
    void refreshRuns();
    const interval = setInterval(() => {
      void refreshRuns();
    }, 5000);
    return () => clearInterval(interval);
  }, [refreshRuns]);

  useEffect(() => {
    if (mode !== "live" || !selectedRunId) {
      return;
    }
    lastEventIdRef.current = 0;
    setEvents([]);
    void refreshEvents(selectedRunId);
    const interval = setInterval(() => {
      void refreshEvents(selectedRunId);
    }, 1200);
    return () => clearInterval(interval);
  }, [refreshEvents, selectedRunId, mode]);

  useEffect(() => {
    if (mode === "live") {
      resetReplay();
      return;
    }
    if (!selectedRunId) {
      return;
    }
    resetReplay();
    void loadReplayEvents(selectedRunId);
  }, [loadReplayEvents, mode, resetReplay, selectedRunId]);

  useEffect(() => {
    if (mode !== "replay" || !isReplaying) {
      stopReplayTimer();
      return;
    }
    if (replayCursor >= replayEvents.length) {
      setIsReplaying(false);
      return;
    }
    const delay = getReplayDelay(replayEvents, replayCursor, replaySpeed);
    replayTimerRef.current = window.setTimeout(() => {
      setReplayCursor((prev) => Math.min(prev + 1, replayEvents.length));
    }, delay);
    return () => stopReplayTimer();
  }, [
    isReplaying,
    mode,
    replayCursor,
    replayEvents,
    replaySpeed,
    stopReplayTimer,
  ]);

  const displayEvents = useMemo(() => {
    if (mode === "replay") {
      return replayEvents.slice(0, replayCursor);
    }
    return events;
  }, [events, mode, replayCursor, replayEvents]);

  const activeNodeId = useMemo(() => {
    const last = displayEvents[displayEvents.length - 1];
    return last?.node ?? null;
  }, [displayEvents]);

  const visitedNodes = useMemo(() => {
    const visited = new Set<string>();
    displayEvents.forEach((event) => {
      if (event.node) {
        visited.add(event.node);
      }
    });
    return visited;
  }, [displayEvents]);

  const graphNodes = useMemo(() => {
    return baseNodes.map((node) => ({
      ...node,
      data: {
        ...node.data,
        active: node.id === activeNodeId,
        visited: visitedNodes.has(node.id),
      },
    }));
  }, [activeNodeId, visitedNodes]);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId);
  const replayTotal = replayEvents.length;
  const replayProgress = displayEvents.length;
  const replayReady = replayTotal > 0 && !isReplayLoading;
  const replaySpeedOptions = [0.5, 1, 2, 4];

  const handleReplayToggle = useCallback(() => {
    if (!replayTotal) {
      return;
    }
    if (isReplaying) {
      setIsReplaying(false);
      return;
    }
    if (replayCursor >= replayTotal) {
      setReplayCursor(0);
    }
    setIsReplaying(true);
  }, [isReplaying, replayCursor, replayTotal]);

  return (
    <div className="flex h-full w-full justify-center-safe px-4 pt-12 pb-4">
      <div className="flex h-full w-full max-w-[1400px] flex-col gap-4 xl:flex-row">
        <section className="flex w-full flex-col gap-3 xl:w-72">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Runs</h2>
            <Button variant="ghost" size="sm" onClick={refreshRuns}>
              Refresh
            </Button>
          </div>
          <ScrollArea className="h-[220px] rounded-md border bg-card/40 xl:h-full">
            <div className="flex flex-col gap-2 p-2">
              {runs.map((run) => (
                <button
                  key={run.run_id}
                  onClick={() => setSelectedRunId(run.run_id)}
                  className={cn(
                    "flex flex-col gap-1 rounded-md border px-3 py-2 text-left text-xs transition",
                    run.run_id === selectedRunId
                      ? "border-primary/60 bg-primary/10"
                      : "border-border/60 hover:border-primary/40",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">
                      {run.title || run.run_id}
                    </span>
                    <Badge variant={run.status === "error" ? "destructive" : "secondary"}>
                      {run.status}
                    </Badge>
                  </div>
                  <div className="text-muted-foreground">
                    {formatTimestamp(run.created_at)}
                  </div>
                </button>
              ))}
              {runs.length === 0 && (
                <div className="text-muted-foreground px-3 py-2 text-xs">
                  No runs yet. Start a chat to create one.
                </div>
              )}
            </div>
          </ScrollArea>
        </section>

        <section className="flex min-h-[420px] flex-1 flex-col gap-3 rounded-md border bg-card/40 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">Agent DAG</h2>
              <p className="text-xs text-muted-foreground">
                {selectedRun?.title || selectedRun?.run_id || "Select a run"}
              </p>
            </div>
            {selectedRun && (
              <div className="text-xs text-muted-foreground">
                {selectedRun.status} · {formatTimestamp(selectedRun.created_at)}
              </div>
            )}
          </div>
          <div className="flex min-h-[360px] flex-1 rounded-md border bg-background/50">
            <ReactFlow
              nodes={graphNodes}
              edges={edges}
              nodeTypes={nodeTypes}
              fitView
              proOptions={{ hideAttribution: true }}
              colorMode={resolvedTheme === "dark" ? "dark" : "light"}
              panOnScroll={false}
              zoomOnScroll={false}
              preventScrolling={false}
              panOnDrag={false}
            >
              <Background
                className="[mask-image:radial-gradient(600px_circle_at_center,white,transparent)]"
                bgColor="var(--background)"
              />
            </ReactFlow>
          </div>
        </section>

        <section className="flex w-full flex-col gap-3 xl:w-[360px]">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">Timeline</h2>
              <span className="text-xs text-muted-foreground">
                {mode === "replay"
                  ? `${replayProgress}/${replayTotal} events`
                  : `${displayEvents.length} events`}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={mode === "live" ? "secondary" : "outline"}
                size="sm"
                onClick={() => setMode("live")}
              >
                Live
              </Button>
              <Button
                variant={mode === "replay" ? "secondary" : "outline"}
                size="sm"
                onClick={() => setMode("replay")}
              >
                Replay
              </Button>
            </div>
          </div>
          {mode === "replay" && (
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border bg-background/40 p-2 text-xs">
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={!replayReady}
                  onClick={handleReplayToggle}
                >
                  {isReplaying ? "Pause" : "Play"}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!replayReady}
                  onClick={resetReplay}
                >
                  Restart
                </Button>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-muted-foreground">Speed</span>
                <div className="flex items-center gap-1">
                  {replaySpeedOptions.map((option) => (
                    <Button
                      key={option}
                      variant={replaySpeed === option ? "secondary" : "outline"}
                      size="sm"
                      onClick={() => setReplaySpeed(option)}
                    >
                      {option}x
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          )}
          <ScrollArea className="h-[260px] rounded-md border bg-card/40 p-3 xl:h-full">
            <div className="flex flex-col gap-3">
              {isReplayLoading && (
                <div className="text-muted-foreground text-xs">
                  Loading replay events...
                </div>
              )}
              {displayEvents.map((event) => (
                <EventItem key={event.id} event={event} />
              ))}
              {displayEvents.length === 0 && !isReplayLoading && (
                <div className="text-muted-foreground text-xs">
                  {mode === "replay"
                    ? "No events to replay yet."
                    : "Waiting for events..."}
                </div>
              )}
            </div>
          </ScrollArea>
          {error && (
            <div className="text-xs text-red-500">Error: {error}</div>
          )}
        </section>
      </div>
    </div>
  );
}

function CircleNode({ data }: { data: { label: string; active?: boolean } }) {
  return (
    <>
      {data.active && (
        <ShineBorder
          className="rounded-full"
          shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
        />
      )}
      <div className="flex h-10 w-10 items-center justify-center rounded-full border bg-[var(--xy-node-background-color-default)] text-[11px]">
        {data.label}
      </div>
      <Handle className="invisible" type="source" position={Position.Right} />
      <Handle className="invisible" type="target" position={Position.Left} />
      <Handle className="invisible" type="source" position={Position.Top} />
      <Handle className="invisible" type="target" position={Position.Bottom} />
    </>
  );
}

function AgentNode({
  data,
  id,
}: {
  data: {
    icon?: ComponentType<{ className?: string }>;
    label: string;
    active?: boolean;
    visited?: boolean;
  };
  id: string;
}) {
  return (
    <>
      {data.active && (
        <ShineBorder
          shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
          className="rounded-[2px]"
        />
      )}
      <Tooltip
        className="max-w-60 text-[13px] font-light opacity-70"
        style={{
          ["--primary" as string]: "#333",
          ["--primary-foreground" as string]: "white",
        }}
        title={data.label}
        side="top"
        sideOffset={14}
      >
        <div
          id={id}
          className={cn(
            "relative flex w-full items-center justify-center gap-2 rounded-sm border px-3 py-2 text-[11px] transition",
            data.visited && "border-primary/40 bg-primary/5",
            data.active && "border-primary/80 bg-primary/10",
          )}
        >
          {data.icon && <data.icon className="h-3.5 w-3.5" />}
          <span>{data.label}</span>
        </div>
      </Tooltip>
      <Handle className="invisible" type="source" position={Position.Left} />
      <Handle className="invisible" type="source" position={Position.Right} />
      <Handle className="invisible" type="source" position={Position.Top} />
      <Handle className="invisible" type="source" position={Position.Bottom} />
      <Handle className="invisible" type="target" position={Position.Left} />
      <Handle className="invisible" type="target" position={Position.Right} />
      <Handle className="invisible" type="target" position={Position.Top} />
      <Handle className="invisible" type="target" position={Position.Bottom} />
    </>
  );
}

function EventItem({ event }: { event: TraceEvent }) {
  const summary = getEventSummary(event);
  const tokenSummary = formatTokens(event.token_usage ?? undefined);
  return (
    <div className="rounded-md border border-border/60 bg-background/40 p-3 text-xs">
      <div className="flex items-start justify-between gap-2">
        <div className="font-medium">
          {event.node || event.agent || "system"}
        </div>
        <Badge variant="outline">{event.event_type}</Badge>
      </div>
      <div className="text-muted-foreground mt-1 flex items-center justify-between">
        <span>{formatTimestamp(event.ts)}</span>
        {event.duration_ms != null && (
          <span>{event.duration_ms.toFixed(0)} ms</span>
        )}
      </div>
      {summary && (
        <div className="text-muted-foreground mt-2 max-h-16 overflow-hidden break-words">
          {summary}
        </div>
      )}
      {tokenSummary && (
        <div className="text-muted-foreground mt-2">
          Tokens: {tokenSummary}
        </div>
      )}
    </div>
  );
}

function formatTimestamp(value?: string | null) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleTimeString();
}

function formatTokens(usage?: Record<string, unknown>) {
  if (!usage) {
    return null;
  }
  const input =
    (usage.input_tokens as number | undefined) ??
    (usage.prompt_tokens as number | undefined);
  const output =
    (usage.output_tokens as number | undefined) ??
    (usage.completion_tokens as number | undefined);
  const total = usage.total_tokens as number | undefined;
  if (input == null && output == null && total == null) {
    return null;
  }
  const parts = [];
  if (input != null) parts.push(`in ${input}`);
  if (output != null) parts.push(`out ${output}`);
  if (total != null) parts.push(`total ${total}`);
  return parts.join(" · ");
}

function getEventSummary(event: TraceEvent) {
  const payload = event.payload ?? {};
  if (event.event_type.startsWith("tool_")) {
    if (payload.tool) {
      return `Tool: ${payload.tool}`;
    }
  }
  if (payload.output) {
    return String(payload.output).slice(0, 180);
  }
  if (payload.content) {
    return String(payload.content).slice(0, 180);
  }
  if (payload.messages) {
    return `Messages: ${payload.messages.length ?? 0}`;
  }
  return "";
}

const REPLAY_MIN_DELAY_MS = 80;
const REPLAY_MAX_DELAY_MS = 3000;
const REPLAY_INITIAL_DELAY_MS = 200;

function getReplayDelay(
  events: TraceEvent[],
  cursor: number,
  speed: number,
) {
  const safeSpeed = speed > 0 ? speed : 1;
  if (cursor <= 0 || cursor >= events.length) {
    return REPLAY_INITIAL_DELAY_MS / safeSpeed;
  }
  const prevTs = Date.parse(events[cursor - 1]?.ts ?? "");
  const nextTs = Date.parse(events[cursor]?.ts ?? "");
  if (Number.isNaN(prevTs) || Number.isNaN(nextTs)) {
    return REPLAY_INITIAL_DELAY_MS / safeSpeed;
  }
  const delta = Math.max(0, nextTs - prevTs);
  const clamped = Math.min(delta, REPLAY_MAX_DELAY_MS);
  return Math.max(REPLAY_MIN_DELAY_MS, clamped / safeSpeed);
}
