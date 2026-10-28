import type { DatasetProfile } from './DatasetProfileForm';

export type JsonObject = Record<string, unknown>;
export interface Gpu {
  uuid: string;
  name: string;
  memory_total_mb?: number | null;
  memory_used_mb?: number | null;
  memory_free_mb?: number | null;
  utilization_pct?: number | null;
  processes?:
    | {
        pid: number;
        memory_mb?: number | null;
        kind?: 'graphics' | 'compute' | 'compute_graphics' | 'unknown';
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
  connection_address?: {
    host: string;
    port: number | null;
    source: 'ssh' | 'hostname';
  } | null;
  status: string;
  last_seen?: string | null;
  roots?: string[];
  output_root?: string;
  cache_root?: string | null;
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
export interface PathSuggestion {
  name: string;
  path: string;
  kind: 'directory' | 'file';
}
export interface PathSuggestions {
  id: string;
  node_id: string;
  status: 'PENDING' | 'READY' | 'FAILED';
  prefix: string;
  profile_id?: string | null;
  base_path?: string | null;
  parent?: string | null;
  entries: PathSuggestion[];
  truncated: boolean;
  error?: { code: string; message: string } | string | null;
}
export interface RecipeSuggestion {
  name: string;
  validated: boolean;
  source: 'code' | 'registered' | 'both';
}
export interface RecipeSuggestions {
  id: string;
  node_id: string;
  profile_id: string;
  prefix: string;
  status: 'PENDING' | 'READY' | 'FAILED';
  entries: RecipeSuggestion[];
  truncated: boolean;
  source: string | null;
  warnings: string[];
  error: { code: string; message: string } | null;
}
export interface RecipeSource {
  path: string;
  sha256: string;
  line: number;
  end_line: number;
  commit_sha?: string | null;
  code_dir: string;
  dirty?: boolean;
  changes_digest?: string;
}
export interface RecipeField {
  path: string;
  kind: 'literal' | 'expression';
  value?: unknown;
  expression: string;
  origin: 'recipe' | 'default';
}
export interface RecipeDetail {
  profile_id: string;
  recipe: string;
  source: RecipeSource;
  fields: RecipeField[];
  source_code: string;
  warnings: string[];
}
export interface RecipeDatasetSlot {
  multiple?: boolean;
  recipe_index: number;
  weight: number;
  processing: string;
}
export interface DataProcessorSchema {
  type: 'boolean' | 'integer' | 'number' | 'string' | 'array' | 'object' | 'null';
  choices?: RecipeLiteral[];
  nullable?: boolean;
  items?: DataProcessorSchema;
  values?: DataProcessorSchema;
  container?: string;
}
export interface DataProcessorParameter extends DataProcessorSchema {
  path: string;
  default?: RecipeLiteral;
  required: boolean;
}
export interface DataProcessorEntry {
  name: string;
  description: string;
  parameters: DataProcessorParameter[];
  unsupported_parameters: { path: string; reason: string; required: boolean }[];
  available: boolean;
  reason?: string;
  source?: { path: string; sha256: string };
  dataset_contract?: DatasetProcessorContract;
  composition_schema_version?: 1;
}
export interface DatasetProcessorContract {
  arm_mode: 'single' | 'bimanual';
  action_representation: 'eef' | 'joint';
  camera_keys: string[];
  required_features: string[];
  action_mode?: 'absolute' | 'delta' | 'unknown';
  rotation_representation?: 'axis_angle' | 'quaternion' | 'rotation_6d' | 'none' | 'unknown';
  training_stage: 'sft' | 'value' | 'offline_rl';
  model_family?: 'standard' | 'memory' | 'hybrid';
  evidence: 'static_repack';
}
export interface DataProcessorCatalogValue {
  profile_id: string;
  entries: DataProcessorEntry[];
  source?: string | Record<string, unknown> | null;
}
export interface DataProcessorCatalog {
  id: string;
  node_id: string;
  profile_id: string;
  status: 'PENDING' | 'READY' | 'FAILED';
  catalog: DataProcessorCatalogValue | null;
  error?: { code: string; message: string } | null;
}
export interface DataProcessorChoice {
  name: string;
  parameters: Record<string, RecipeLiteral>;
}
interface DatasetBindingBase {
  dataset_asset_id: string;
  norm_asset_id: string;
  weight: number;
}
export type DatasetBinding = DatasetBindingBase &
  (
    | { recipe_index: number; processor?: never }
    | { recipe_index?: never; processor: DataProcessorChoice }
  );
export interface RecipeDetailQuery {
  id: string;
  node_id: string;
  profile_id: string;
  recipe: string;
  status: 'PENDING' | 'READY' | 'FAILED';
  detail: RecipeDetail | null;
  error: { code: string; message: string } | null;
}
export type RecipeLiteral =
  | null
  | boolean
  | number
  | string
  | RecipeLiteral[]
  | { [key: string]: RecipeLiteral };
export type RecipeOverrides = Record<string, RecipeLiteral>;
export interface RecipeWriteback {
  id: string;
  node_id: string;
  profile_id: string;
  recipe_revision_id: string;
  config_name: string;
  status: 'PENDING' | 'SUCCEEDED' | 'FAILED';
  commit_sha: string | null;
  recipe: SavedRecipe | null;
  error: { code: string; message: string } | null;
}
export interface RecipeDefinition {
  schema_version: 1;
  id: string;
  recipe_id: string;
  name: string;
  revision: number;
  base_recipe: string;
  node_id: string;
  profile_id: string;
  source: RecipeSource;
  overrides: RecipeOverrides;
  digest: string;
}
export interface RecipeRevision {
  id: string;
  recipe_id: string;
  revision: number;
  definition: RecipeDefinition;
  detail?: RecipeDetail;
  description?: string;
  created_at: string;
  validated: boolean;
  status?: 'VALIDATED' | 'UNVALIDATED';
}
export interface SavedRecipe {
  id: string;
  name: string;
  description: string;
  node_id: string;
  profile_id: string;
  latest_revision: RecipeRevision;
  created_at: string;
  validated: boolean;
  status?: 'VALIDATED' | 'UNVALIDATED';
}
export interface Profile {
  id: string;
  node_id: string;
  name: string;
  code_dir: string;
  python: string;
  adapter: string;
  recipe?: string | null;
  execution_mode?: string;
  docker_image?: string;
  metadata?: JsonObject;
}
export interface RuntimeCheck {
  profile_id: string;
  status: 'CHECKING' | 'READY' | 'FAILED';
  checked_at?: string;
  checks?: { name: string; ok: boolean; message: string }[];
  python_version?: string;
  executable?: string;
  module_origin?: string;
  error?: string;
}
export interface AgentProfileInput {
  id: string;
  name: string;
  code_dir: string;
  python: string;
  adapter: 'smoke' | 'openpi-sft';
  recipe?: string | null;
  execution_mode: 'native';
}
export interface AgentConfigRequest {
  name: string;
  control_plane_url?: string;
  roots: string[];
  output_root: string;
  state_dir: string;
  profiles: AgentProfileInput[];
}
export interface AgentConfigDefaults {
  control_plane_url: string | null;
  source: 'configured' | 'request' | 'unavailable';
  message: string;
}
export interface AgentConfigResult {
  id: string;
  filename: string;
  config: JsonObject;
}
export interface SshConnectionResult {
  connection_id: string;
  host: string;
  port: number;
  username: string;
  fingerprint: string;
  hostname: string;
  expires_at: string;
  can_install: boolean;
  checks: { name: string; ok: boolean; message: string }[];
  existing_agent?: (AgentConfigRequest & { node_id?: string }) | null;
}
export interface RemoteDeployment {
  id: string;
  connection_id: string;
  host: string;
  port: number;
  username: string;
  install_dir: string;
  onboarding_id: string;
  node_id: string | null;
  state:
    | 'QUEUED'
    | 'RUNNING'
    | 'WAITING_AGENT'
    | 'SUCCEEDED'
    | 'FAILED'
    | 'INTERRUPTED'
    | 'DISCONNECTED'
    | 'UNINSTALLED';
  operation?: 'install' | 'disconnect' | 'uninstall';
  operation_result?: {
    backup_dir?: string;
    removed_paths?: string[];
    service_removed?: boolean;
  } | null;
  stage: string;
  logs: { timestamp: string; stage: string; message: string }[];
  history: { state: string; stage: string; timestamp: string }[];
  connection_mode: 'ssh-tunnel' | 'direct';
  tunnel_state:
    | 'STARTING'
    | 'HEALTHY'
    | 'RECONNECTING'
    | 'FAILED'
    | 'NOT_REQUIRED'
    | 'DISCONNECTED'
    | 'UNINSTALLED';
  error: { code: string; message: string } | null;
  agent: Partial<AgentConfigRequest>;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
}
export interface NodeManagementInfo {
  node_id: string;
  node_status: string;
  managed: boolean;
  host?: string | null;
  port?: number | null;
  username?: string | null;
  install_dir?: string | null;
  deployment_id?: string | null;
  can_disconnect: boolean;
  can_uninstall: boolean;
  reason: string;
  disconnect_reason?: string | null;
  uninstall_reason?: string | null;
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
  runtime_adapter?: string;
  stages?: TrainingStageSpec[];
}
export type TrainingStage = 'value' | 'offline_rl';
export interface TrainingStageSpec {
  id: TrainingStage;
  name: string;
  description: string;
  parameters?: Record<string, ParameterSpec | number> | ParameterSpec[];
  required_inputs?: string[] | Record<string, unknown>;
  default_entrypoint?: string;
  default_multi_entrypoint?: string;
  recipe_prefixes?: string[];
}
export interface AutomaticPreflight {
  schema_version: 1;
  scope: 'configuration_and_local_assets';
  plan_digest: string;
  plan: {
    generated_config?: GeneratedTrainingConfig;
    input_authority?: 'registered_assets';
    model?: { path: string; asset_id?: string };
    entrypoint: string;
    datasets: {
      path: string;
      norm_path?: string;
      frames?: number;
      episodes?: number;
      [key: string]: unknown;
    }[];
    [key: string]: unknown;
  };
}
export interface AutoTrainingConfig {
  model_type: 'pi0' | 'pi05';
  finetune_mode: 'full' | 'lora';
  action_horizon: number;
  action_dim: number;
}
export interface GeneratedTrainingConfig {
  schema_version: 1;
  language: 'python';
  filename: string;
  source: string;
  sha256: string;
}
export interface CodeSnapshot {
  auto_config?: AutoTrainingConfig | null;
  confirmed_config_sha256?: string | null;
  config_confirmed_at?: string | null;
  dataset_profiles?: Record<string, DatasetProfile>;
  training_stage?: TrainingStage | null;
  stage_config?: Record<string, number> | null;
  dataset_bindings?: DatasetBinding[] | null;
  requested_entrypoint?: string | null;
  recipe?: string | null;
  recipe_definition?: RecipeDefinition | null;
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
  automatic_preflight?: AutomaticPreflight;
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
  auto_config?: AutoTrainingConfig | null;
  confirmed_config_sha256?: string | null;
  dataset_profiles?: Record<string, DatasetProfile> | null;
  training_stage?: TrainingStage | null;
  stage_config?: Record<string, number> | null;
  dataset_bindings?: DatasetBinding[] | null;
  entrypoint?: string | null;
  recipe?: string | null;
  recipe_definition?: RecipeDefinition | null;
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
  auto_config?: AutoTrainingConfig;
  training_stage?: TrainingStage;
  stage_config?: Record<string, number>;
  dataset_bindings?: DatasetBinding[];
  entrypoint?: string;
  recipe?: string;
  recipe_revision_id?: string;
  recipe_definition?: RecipeDefinition;
  name: string;
  node_id: string;
  profile_id: string;
  template_id: string;
  inputs: Record<string, string>;
  parameters: Record<string, number>;
  gpu_count: number;
}
export interface PreflightCheck {
  name: string;
  ok: boolean;
  message?: string;
  severity?: 'warning' | 'error';
  code?: string;
}
export interface Preview {
  id: string;
  status: string;
  checks?: PreflightCheck[];
  code_snapshot?: CodeSnapshot;
  error?: { code?: string; message?: string };
  request?: RunRequest & { dataset_profiles?: Record<string, DatasetProfile> };
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
