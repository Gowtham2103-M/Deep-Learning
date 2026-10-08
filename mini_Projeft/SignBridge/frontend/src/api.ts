// All calls to the FastAPI backend live here, so the URLs match backend/app/routes/*.py in one place.
// In development Vite forwards /api to http://127.0.0.1:8000 (see vite.config.ts).
const BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export interface AppConfig {
  num_frames: number;
  capture_fps: number;
  window_seconds: number;
  confidence_threshold: number;
  stable_count: number;
  speak_cooldown_seconds: number;
  use_mediapipe: boolean;
  match_threshold: number;
}

export interface DatasetStatus {
  available: boolean;
  path: string;
  message: string;
  videos: number;
  classes: number;
  csv: string | null;
  signer_column: string | null;
  warnings: string[];
}

export interface ModelStatus {
  trained: boolean;
  message: string;
  classes?: number;
  num_frames?: number;
  backbone?: string;
  use_attention?: boolean;
  use_mediapipe?: boolean;
  device?: string;
  split_mode?: string;
}

export interface StatusResponse {
  dataset: DatasetStatus;
  model: ModelStatus;
  config: AppConfig;
}

export interface TopK {
  class_id: number;
  label: string;
  confidence: number;
}

export interface PredictResponse {
  status: "ok";
  class_id: number;
  label: string;
  confidence: number;
  accepted: boolean;
  threshold: number;
  text: string;
  top_k: TopK[];
  frames_received: number;
  frames_used: number;
  device: string;
  latency_ms: { preprocess_ms: number; cnn_ms: number; head_ms: number; total_ms: number };
}

export interface MatchResponse {
  status: "matched" | "no_match" | "dataset_missing";
  input: string;
  normalized: string;
  message: string;
  closest_score?: number;
  match: null | { class_id: number; label: string; score: number; method: string; video_url: string };
}

export interface MetricsResponse {
  trained: boolean;
  message: string;
  metrics: null | {
    evaluated_at: string;
    num_samples: number;
    split_mode: string;
    split_warnings: string[];
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
  };
  latency: null | {
    measured_at: string;
    videos: number;
    device: string;
    total_ms: { mean: number; median: number; p95: number };
  };
}

export class ApiError extends Error {
  constructor(message: string, public status: number, public code?: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the backend. Is it running on port 8000?", 0, "offline");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = body?.message ?? body?.detail ?? `Request failed (${res.status})`;
    throw new ApiError(typeof msg === "string" ? msg : JSON.stringify(msg), res.status, body?.status);
  }
  return body as T;
}

export const getStatus = () => request<StatusResponse>("/api/status");
export const getMetrics = () => request<MetricsResponse>("/api/metrics");

export function predictFrames(frames: Blob[]): Promise<PredictResponse> {
  const form = new FormData();
  frames.forEach((b, i) => form.append("frames", b, `frame_${i}.jpg`));
  return request<PredictResponse>("/api/predict/frames", { method: "POST", body: form });
}

export const matchSpeech = (text: string) =>
  request<MatchResponse>("/api/speech/match", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

export const referenceVideoUrl = (path: string) => `${BASE}${path}`;
