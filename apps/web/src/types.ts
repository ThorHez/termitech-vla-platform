export type JsonObject = Record<string, unknown>;
export interface Gpu {
  uuid: string;
  name: string;
  memory_total_mb?: number | null;
  memory_used_mb?: number | null;
  utilization_pct?: number | null;
  processes?:
    | {
        pid: number;
        memory_mb?: number | null;
        attempt_id?: string | null;
        ownership?: string;
      }[]
    | null;
  reserved_by?: string | null;
}
export interface Node {
  id: string;
  name: string;
  hostname: string;
  status: string;
  last_seen?: string | null;
  roots?: string[];
  output_root?: string;
  gpus?: Gpu[];
  metrics?: JsonObject;
}
export interface Asset {
  id: string;
  name: string;
  kind: string;
  node_id: string;
  path: string;
  version: string;
  status?: string;
  state?: string;
  metadata?: JsonObject;
  created_at?: string;
}
export interface Profile {
  id: string;
  node_id: string;
  name: string;
  code_dir: string;
  python: string;
  adapter: string;
  recipe?: string;
  execution_mode?: string;
  docker_image?: string;
  metadata?: JsonObject;
}
export interface ParameterSpec {
  name?: string;
  type?: string;
  default?: number;
  minimum?: number;
  maximum?: number;
  min?: number;
  max?: number;
  label?: string;
}
export interface Template {
  id: string;
  name: string;
  algorithm: string;
  status: string;
  description: string;
  parameters?: Record<string, ParameterSpec | number> | ParameterSpec[];
  required_inputs?: string[] | Record<string, unknown>;
  min_gpus?: number;
}
export interface CodeSnapshot {
  repo_url?: string;
  code_dir?: string;
  branch?: string | null;
  commit_sha?: string;
  commit_message?: string;
  committed_at?: string;
  dirty?: boolean;
  changes_digest?: string;
  untracked_digest?: string;
  module_origin?: string;
  captured_at?: string;
  complete?: boolean;
}
export interface Attempt {
  id: string;
  state?: string;
  status?: string;
  code_snapshot?: CodeSnapshot;
  started_at?: string;
  finished_at?: string;
  output_dir?: string;
  error?: { code?: string; message?: string };
  pid?: number;
  gpu_uuids?: string[];
  latest_metrics?: JsonObject;
  resources?: JsonObject;
  code_changed_during_run?: boolean;
}
export interface Run {
  id: string;
  name: string;
  node_id: string;
  profile_id: string;
  template_id: string;
  state: string;
  created_at: string;
  updated_at?: string;
  inputs?: Record<string, string | { id?: string; path?: string; name?: string }>;
  parameters?: Record<string, number>;
  gpu_count?: number;
  code_expectation?: CodeSnapshot;
  attempts?: Attempt[];
  outputs?: Asset[];
  queue_reason?: string | null;
  error?: { code?: string; message?: string };
}
export interface Overview {
  nodes_online: number;
  nodes_total: number;
  gpus_total: number;
  gpus_available: number;
  runs_active: number;
  runs_failed: number;
  assets_total: number;
}
export interface Alert {
  id?: string;
  code?: string;
  severity?: string;
  message?: string;
  title?: string;
  node_id?: string;
  run_id?: string;
  created_at?: string;
}
export interface RunRequest {
  name: string;
  node_id: string;
  profile_id: string;
  template_id: string;
  inputs: Record<string, string>;
  parameters: Record<string, number>;
  gpu_count: number;
}
export interface Preview {
  id: string;
  status: string;
  checks?: { name: string; ok: boolean; message?: string }[];
  code_snapshot?: CodeSnapshot;
  error?: { code?: string; message?: string };
  request?: RunRequest;
}
export interface RunEvent {
  seq: number;
  type: string;
  timestamp?: string;
  attempt_id?: string;
  data: JsonObject;
}
export interface LogEntry {
  seq: number;
  text: string;
  timestamp?: string;
}
export interface Logs {
  items: LogEntry[];
  next_seq: number;
}
export interface NodeDetail extends Node {
  assets?: Asset[];
  profiles?: Profile[];
  runs?: Run[];
}
