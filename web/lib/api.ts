// Thin typed client for the FastAPI backend. All calls go through /api/* on this
// origin (proxied by next.config.ts). With local auth the session cookie does the
// work; with Clerk we attach a Bearer token obtained from the Clerk hook.

export type Provider = "anthropic" | "openai" | "vercel" | "vllm";

export interface ServerInfo {
  auth_provider: "local" | "clerk";
  allow_signup: boolean;
  clerk_publishable_key: string;
  allow_server_keys: boolean;
  providers: Provider[];
  version: string;
}

export interface User {
  id: string;
  email: string;
  display_name: string;
  is_admin: boolean;
  auth_provider: string;
}

export interface KeyOut {
  provider: Provider;
  masked: string;
  base_url: string | null;
  updated_at: string;
  verified_at: string | null; // set when the provider accepted the key on save
  n_models: number | null;
  verify_note: string | null;
}

export interface Workspace {
  id: string;
  slug: string;
  name: string;
  role: "owner" | "member";
  default_model: string;
  default_pack: string;
  default_episodes: number;
  builder_model: string;
  train_base_model: string;
}

export interface Pack {
  name: string;
  kind: string;
  tiers: number[];
  max_steps: number;
  description: string;
  custom?: boolean; // published from the builder in this workspace
  project_id?: string | null;
}

// ---- environment builder ----
export interface Readiness {
  item: string;
  ok: boolean;
  detail?: unknown;
}
export interface Validation {
  spec_ok: boolean;
  spec_problems: string[];
  tasks_ok: number;
  task_problems: Record<string, string[]>;
  tiers: Record<string, number>;
}
export interface BuilderUpload {
  id: string;
  filename: string;
  size_bytes: number;
  status: "uploaded" | "processing" | "ready" | "failed";
  format: { format?: string; mapping?: unknown; warnings?: string[]; skipped_rows?: number } | null;
  n_conversations: number;
  redactions: Record<string, number> | null;
  profile: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
}
export interface BuilderProject {
  id: string;
  name: string;
  title: string;
  purpose: string;
  created_at: string;
  head: { id: string; number: number; spec: Record<string, unknown> | null; n_tasks: number; validation: Validation | null } | null;
  readiness: Readiness[];
  versions: { version: number; pack_name: string; revision_id: string; created_at: string }[];
  job: { id: string; kind: string; status: string; progress: string } | null;
  uploads: BuilderUpload[];
  builder_model: string;
}
export interface BuilderProjectSummary {
  id: string;
  name: string;
  title: string;
  purpose: string;
  created_at: string;
  head_number: number | null;
  n_tasks: number;
  versions: number;
}
export interface BuilderMessage {
  id: string;
  seq: number;
  role: "user" | "assistant" | "tool";
  content: string;
  tool_calls?: { name: string; args_preview: string }[];
  name?: string;
  meta?: { revision?: number } | null;
  created_at: string;
}
export interface RevisionSummary {
  id: string;
  number: number;
  parent_number: number | null;
  message: string;
  author: "builder" | "user";
  n_tasks: number;
  spec_ok: boolean;
  tasks_ok: number;
  task_problem_count: number;
  is_head: boolean;
  published_version: number | null;
  created_at: string;
}
export interface RevisionDiff {
  from: number | null;
  to: number;
  spec_diff: string;
  tasks_added: string[];
  tasks_removed: string[];
  tasks_changed_file: boolean;
}
export interface TaskRow {
  id: string;
  tier: number;
  goal: string;
  n_success: number;
  n_reference: number;
  source: Record<string, unknown> | null;
  problems: string[];
}

// ---- environments (see envforge/core/env.py: Diagram, EnvDescription) ----
export interface DiagramNode {
  id: string;
  label: string;
  kind: string;
  group: string | null;
  state: string; // ok | broken | target | inactive (unknown states render as ok)
  badges: string[];
  detail: string;
  x: number | null;
  y: number | null;
  parent: string | null;
}
export interface DiagramEdge {
  source: string;
  target: string;
  label: string;
  state: string; // ok | broken | path | inactive
  directed: boolean;
}
export interface Diagram {
  title: string;
  layout: "graph" | "tree";
  nodes: DiagramNode[];
  edges: DiagramEdge[];
  notes: string[];
}
export interface ToolSpec {
  name: string;
  description: string;
  parameters: { type?: string; properties?: Record<string, { type?: string; enum?: unknown[]; description?: string }>; required?: string[] };
  category: "observe" | "act" | null;
}
export interface PackDetail {
  name: string;
  kind: string;
  description: string;
  overview: string;
  world: string[];
  observation: string;
  action: string;
  max_steps: number;
  tiers: number[];
  tier_notes: Record<string, string>;
  reward_rules: { outcome: string; reward: number }[];
  tables: { title: string; columns: string[]; rows: string[][]; note: string }[];
  tools: ToolSpec[];
  finish_tool: ToolSpec;
  world_diagram: Diagram | null;
  dimensions: [string, string][];
  replayable: boolean;
}
export interface TaskPreview {
  task: { id: string; tier: number; seed: number; goal: string; hidden: Record<string, unknown> };
  system_prompt: string;
  first_observation: string | null;
  diagram_before: Diagram | null;
  solution: { name: string; args: Record<string, unknown>; observation: string | null }[];
  diagram_after: Diagram | null;
  verdict: { success: boolean; reward: number; details: Record<string, unknown> } | null;
}
export interface ReplayFrame {
  step: number | null;
  call: { name: string; args: Record<string, unknown> } | null;
  observation: string | null;
  error: boolean;
  diagram: Diagram;
  changed: boolean;
}

export interface Metrics {
  n: number;
  solve_rate: number;
  mean_reward: number;
  mean_steps: number;
  mean_tool_calls: number;
  tool_error_rate: number;
  collateral_rate: number;
  max_steps_rate: number;
  error_rate: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  wall_time_s: number;
}

export interface Run {
  id: string;
  workspace_id: string;
  pack: string;
  model: string;
  episodes: number;
  tiers: number[];
  seed_start: number;
  max_steps: number | null;
  temperature: number;
  label: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  error: string | null;
  metrics: { overall: Metrics; by_tier: Record<string, Metrics> } | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cost_usd: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface EpisodeSummary {
  episode: number;
  task_id: string;
  tier: number;
  seed: number;
  success: boolean;
  reward: number;
  steps: number;
  tool_calls: number;
  tool_errors: number;
  finished_reason: string;
  collateral: string[];
  tokens: number;
}

export interface Trajectory {
  task: { id: string; tier: number; seed: number; goal: string; hidden: Record<string, unknown> };
  system_prompt: string;
  steps: {
    index: number;
    assistant_text: string;
    tool_calls: { id: string; name: string; args: Record<string, unknown> }[];
    observations: string[];
    errors: boolean[];
  }[];
  verdict: { success: boolean; reward: number; details: Record<string, unknown> } | null;
  finished_reason: string;
  error: string | null;
  usage: { input_tokens: number; output_tokens: number };
  wall_time_s: number;
}

export interface ModelInfo {
  id: string;
  provider: string;
  name: string;
  owned_by: string;
  context_window: number;
  input_per_mtok: number | null;
  output_per_mtok: number | null;
  tags: string[];
  released: number | null;
}

export interface ModelsResponse {
  providers: Provider[];
  catalog_available: boolean;
  models: ModelInfo[];
}

export interface UsageBucket {
  runs: number;
  episodes: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  unpriced_runs: number;
  cache_read_tokens?: number;
}

export interface Usage {
  days: number;
  totals: UsageBucket;
  by_model: (UsageBucket & { model: string })[];
  // workspace usage groups by member; personal usage groups by workspace
  groups: (UsageBucket & { user_id?: string; email?: string; display_name?: string; workspace_id?: string; name?: string })[];
  by_day: (UsageBucket & { date: string })[];
  builder?: { jobs: number; input_tokens: number; output_tokens: number; cache_read_tokens?: number; cost_usd: number };
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// Set by the auth provider layer when using Clerk; null for local auth.
let tokenGetter: (() => Promise<string | null>) | null = null;
export function setTokenGetter(fn: typeof tokenGetter) {
  tokenGetter = fn;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json", ...(init.headers as Record<string, string>) };
  if (tokenGetter) {
    const t = await tokenGetter();
    if (t) headers.Authorization = `Bearer ${t}`;
  }
  const res = await fetch(path, { credentials: "include", ...init, headers });
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? res.statusText);
  return body as T;
}

export const Api = {
  info: () => api<ServerInfo>("/api/auth/info"),
  me: () => api<User>("/api/auth/me"),
  signup: (email: string, password: string, display_name: string) =>
    api<User>("/api/auth/signup", { method: "POST", body: JSON.stringify({ email, password, display_name }) }),
  login: (email: string, password: string) =>
    api<User>("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  logout: () => api<void>("/api/auth/logout", { method: "POST" }),

  packs: () => api<Pack[]>("/api/packs"),
  workspaces: () => api<Workspace[]>("/api/workspaces"),
  createWorkspace: (name: string) => api<Workspace>("/api/workspaces", { method: "POST", body: JSON.stringify({ name }) }),
  updateWorkspace: (id: string, patch: Partial<Pick<Workspace, "name" | "default_model" | "default_pack" | "default_episodes" | "builder_model" | "train_base_model">>) =>
    api<Workspace>(`/api/workspaces/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),

  // the signed-in user's own account: profile, password, provider keys (shared by all their workspaces)
  updateProfile: (display_name: string) => api<User>("/api/me", { method: "PATCH", body: JSON.stringify({ display_name }) }),
  changePassword: (current_password: string, new_password: string) =>
    api<void>("/api/me/password", { method: "POST", body: JSON.stringify({ current_password, new_password }) }),
  keys: () => api<KeyOut[]>("/api/me/keys"),
  putKey: (provider: Provider, secret: string, base_url?: string) =>
    api<KeyOut[]>("/api/me/keys", { method: "PUT", body: JSON.stringify({ provider, secret, base_url: base_url || null }) }),
  deleteKey: (provider: Provider) => api<KeyOut[]>(`/api/me/keys/${provider}`, { method: "DELETE" }),

  // environments runnable in a workspace: built-in packs + environments published from the builder
  wsPacks: (ws: string) => api<Pack[]>(`/api/workspaces/${ws}/packs`),
  pack: (ws: string, name: string) => api<PackDetail>(`/api/workspaces/${ws}/packs/${encodeURIComponent(name)}`),
  preview: (ws: string, name: string, seed: number, tier: number) =>
    api<TaskPreview>(`/api/workspaces/${ws}/packs/${encodeURIComponent(name)}/preview?seed=${seed}&tier=${tier}`),

  // environment builder
  projects: (ws: string) => api<BuilderProjectSummary[]>(`/api/workspaces/${ws}/builder/projects`),
  createProject: (ws: string, title: string, purpose: string) =>
    api<BuilderProject>(`/api/workspaces/${ws}/builder/projects`, { method: "POST", body: JSON.stringify({ title, purpose }) }),
  project: (ws: string, id: string) => api<BuilderProject>(`/api/workspaces/${ws}/builder/projects/${id}`),
  deleteProject: (ws: string, id: string) => api<void>(`/api/workspaces/${ws}/builder/projects/${id}`, { method: "DELETE" }),
  builderMessages: (ws: string, id: string, after = 0) =>
    api<{ messages: BuilderMessage[]; job: { status: string; progress: string } | null }>(`/api/workspaces/${ws}/builder/projects/${id}/messages?after=${after}`),
  sendBuilderMessage: (ws: string, id: string, content: string) =>
    api<{ job_id: string }>(`/api/workspaces/${ws}/builder/projects/${id}/messages`, { method: "POST", body: JSON.stringify({ content }) }),
  revisions: (ws: string, id: string) => api<RevisionSummary[]>(`/api/workspaces/${ws}/builder/projects/${id}/revisions`),
  revision: (ws: string, id: string, rid: string) =>
    api<{ id: string; number: number; message: string; spec: Record<string, unknown> | null; validation: Validation | null; n_tasks: number }>(
      `/api/workspaces/${ws}/builder/projects/${id}/revisions/${rid}`,
    ),
  revisionDiff: (ws: string, id: string, rid: string) => api<RevisionDiff>(`/api/workspaces/${ws}/builder/projects/${id}/revisions/${rid}/diff`),
  restoreRevision: (ws: string, id: string, rid: string) =>
    api<{ revision: number }>(`/api/workspaces/${ws}/builder/projects/${id}/revisions/${rid}/restore`, { method: "POST" }),
  checkoutRevision: (ws: string, id: string, rid: string) =>
    api<{ head: number }>(`/api/workspaces/${ws}/builder/projects/${id}/revisions/${rid}/checkout`, { method: "POST" }),
  editSpec: (ws: string, id: string, spec: unknown, message: string) =>
    api<{ revision: number; validation: Validation }>(`/api/workspaces/${ws}/builder/projects/${id}/spec`, { method: "PUT", body: JSON.stringify({ spec, message }) }),
  revisionTasks: (ws: string, id: string, rid: string) => api<TaskRow[]>(`/api/workspaces/${ws}/builder/projects/${id}/revisions/${rid}/tasks`),
  deleteUpload: (ws: string, id: string, uid: string) => api<void>(`/api/workspaces/${ws}/builder/projects/${id}/uploads/${uid}`, { method: "DELETE" }),
  builderPreview: (ws: string, id: string, seed: number, tier?: number) =>
    api<{ overview: PackDetail; preview: TaskPreview }>(`/api/workspaces/${ws}/builder/projects/${id}/preview?seed=${seed}${tier ? `&tier=${tier}` : ""}`),
  publish: (ws: string, id: string, revision_id?: string) =>
    api<{ version: number; pack_name: string }>(`/api/workspaces/${ws}/builder/projects/${id}/publish`, { method: "POST", body: JSON.stringify({ revision_id: revision_id ?? null }) }),

  runs: (ws: string) => api<Run[]>(`/api/workspaces/${ws}/runs`),
  run: (ws: string, id: string) => api<Run>(`/api/workspaces/${ws}/runs/${id}`),
  createRun: (ws: string, body: Partial<Run> & { tiers?: number[] }) =>
    api<Run>(`/api/workspaces/${ws}/runs`, { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (ws: string, id: string) => api<Run>(`/api/workspaces/${ws}/runs/${id}/cancel`, { method: "POST" }),
  deleteRun: (ws: string, id: string) => api<void>(`/api/workspaces/${ws}/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  replay: (ws: string, id: string, ep: number) =>
    api<{ frames: ReplayFrame[]; verdict: Trajectory["verdict"] }>(`/api/workspaces/${ws}/runs/${encodeURIComponent(id)}/episodes/${ep}/replay`),
  episodes: (ws: string, id: string) => api<EpisodeSummary[]>(`/api/workspaces/${ws}/runs/${id}/episodes`),
  episode: (ws: string, id: string, ep: number) => api<Trajectory>(`/api/workspaces/${ws}/runs/${id}/episodes/${ep}`),

  models: (ws: string, refresh = false) => api<ModelsResponse>(`/api/workspaces/${ws}/models${refresh ? "?refresh=true" : ""}`),
  usage: (ws: string, days = 30) => api<Usage>(`/api/workspaces/${ws}/usage?days=${days}`),
  myUsage: (days = 30) => api<Usage>(`/api/auth/me/usage?days=${days}`),
  logoutAll: () => api<void>("/api/auth/logout-all", { method: "POST" }),
};

/** Upload a chat-log file to a builder project (streamed as the raw request body). */
export async function uploadBuilderFile(ws: string, project: string, file: File): Promise<BuilderUpload> {
  const headers: Record<string, string> = { "Content-Type": "application/octet-stream" };
  if (tokenGetter) {
    const t = await tokenGetter();
    if (t) headers.Authorization = `Bearer ${t}`;
  }
  const res = await fetch(`/api/workspaces/${ws}/builder/projects/${project}/uploads?filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    credentials: "include",
    headers,
    body: file,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? res.statusText);
  return body as BuilderUpload;
}

/** Download filtered episodes as JSONL (rejection sampling → training data). */
export async function downloadExport(ws: string, runIds: string[], minReward: number, format: "chat" | "trajectory"): Promise<number> {
  const qs = new URLSearchParams({ runs: runIds.join(","), min_reward: String(minReward), format });
  const headers: Record<string, string> = {};
  if (tokenGetter) {
    const t = await tokenGetter();
    if (t) headers.Authorization = `Bearer ${t}`;
  }
  const res = await fetch(`/api/workspaces/${ws}/runs/export?${qs}`, { credentials: "include", headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body?.detail ?? res.statusText);
  }
  const blob = await res.blob();
  const name = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") ?? "")?.[1] ?? "envforge.jsonl";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
  return Number(res.headers.get("x-episodes") ?? 0);
}

export const fmtCost = (usd: number | null | undefined) =>
  usd === null || usd === undefined ? "?" : usd === 0 ? "$0" : usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;

export const fmtTokens = (n: number) =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)}M` : n >= 10_000 ? `${Math.round(n / 1000)}k` : n.toLocaleString();
