import { useEffect, useReducer } from "react";

import { wsUrl } from "./client";
import type { Job, JobEvent, StageEventData } from "./types";

const MAX_LINES = 5000;

export interface StageView extends StageEventData {
  startedAt?: string;
}

export interface JobStreamState {
  job: Job | null;
  stages: StageView[];        // in pipeline order
  lines: string[];            // last MAX_LINES log lines
  droppedLines: number;
  cursor: number;
  connected: boolean;
  error: string | null;
}

type Action =
  | { type: "job"; job: Job }
  | { type: "events"; events: JobEvent[] }
  | { type: "connected"; value: boolean }
  | { type: "error"; error: string }
  | { type: "reset" };

const initial: JobStreamState = {
  job: null,
  stages: [],
  lines: [],
  droppedLines: 0,
  cursor: 0,
  connected: false,
  error: null,
};

function reduce(state: JobStreamState, action: Action): JobStreamState {
  switch (action.type) {
    case "reset":
      return initial;
    case "job":
      return { ...state, job: action.job };
    case "connected":
      return { ...state, connected: action.value, error: action.value ? null : state.error };
    case "error":
      return { ...state, error: action.error };
    case "events": {
      let { lines, droppedLines } = state;
      const stages = new Map(state.stages.map((s) => [s.stage, s]));
      const newLines: string[] = [];
      for (const e of action.events) {
        if (e.type === "log") newLines.push(...e.data.lines);
        else if (e.type === "stage") {
          const prev = stages.get(e.data.stage);
          stages.set(e.data.stage, {
            ...prev,
            ...e.data,
            // a repeated "started" (progress update) keeps the original start time
            startedAt: e.data.state === "started" ? (prev?.state === "started" ? prev.startedAt : e.ts) : prev?.startedAt,
          });
        }
      }
      if (newLines.length) {
        lines = lines.concat(newLines);
        if (lines.length > MAX_LINES) {
          droppedLines += lines.length - MAX_LINES;
          lines = lines.slice(-MAX_LINES);
        }
      }
      const last = action.events[action.events.length - 1];
      return {
        ...state,
        lines,
        droppedLines,
        stages: [...stages.values()].sort((a, b) => a.index - b.index),
        cursor: last ? last.seq : state.cursor,
      };
    }
  }
}

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

/** Follow a job over WS /jobs/{id}/stream, reconnecting from the last cursor until it ends. */
export function useJobStream(jobId: string | undefined): JobStreamState {
  const [state, dispatch] = useReducer(reduce, initial);

  useEffect(() => {
    if (!jobId) return;
    dispatch({ type: "reset" });
    let cursor = 0;
    let finished = false;
    let closedByUs = false;
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    const connect = () => {
      ws = new WebSocket(wsUrl(`/jobs/${jobId}/stream?after=${cursor}`));
      ws.onopen = () => {
        attempt = 0;
        dispatch({ type: "connected", value: true });
      };
      ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data);
        if (data.type === "job") {
          dispatch({ type: "job", job: data.job });
          if (TERMINAL.has(data.job.status)) finished = true;
        } else if (data.type === "events") {
          const events = data.events as JobEvent[];
          if (events.length) cursor = events[events.length - 1]!.seq;
          dispatch({ type: "events", events });
        } else if (data.type === "error") {
          finished = true;
          dispatch({ type: "error", error: String(data.detail) });
        }
      };
      ws.onclose = () => {
        dispatch({ type: "connected", value: false });
        if (closedByUs || finished) return;
        attempt += 1;
        retry = setTimeout(connect, Math.min(5000, 400 * attempt));
      };
    };
    connect();
    return () => {
      closedByUs = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, [jobId]);

  return state;
}
