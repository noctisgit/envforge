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
  max_active_runs: number;
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
  monthly_spend_limit_usd: number | null; // runs + builder, per calendar month (UTC); null = no limit
  month_spend_usd: number;
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
  bytes_received: number;
  source: "upload" | "url";
  progress: { stage?: string; bytes?: number; total?: number | null; conversations?: number } | null;
  chunk_bytes: number;
  status: "receiving" | "uploaded" | "processing" | "ready" | "failed";
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
  privacy: BuilderPrivacy;
  // a cost estimate waiting for the user's click (only that approves spending) + conversations approved so far
  extraction: { quote: ExtractionQuote | null; approved: number };
  spend_left_usd: number | null; // under the workspace's monthly limit; null = no limit
}
export interface ExtractionQuote {
  id: string;
  conversations: number;
  model: string;
  cost_usd: number | null;
  cost_usd_with_retries: number | null;
  created_at: string;
}
export type PrivacyProfile = "off" | "standard" | "strict";
export interface ReviewResult {
  reviewed: number;
  flagged: number;
  population: number;
  leak_rate: number;
  ci_low: number;
  ci_high: number;
  threshold: number;
  passed: boolean;
  exact: boolean;
}
export interface PrivacyReview {
  id: string;
  status: "open" | "passed" | "failed" | "outdated";
  n: number;
  population: number;
  threshold: number;
  profile: PrivacyProfile;
  created_at: string;
  flagged?: string[];
  result?: ReviewResult;
  reviewed_at?: string;
}
export interface BuilderPrivacy {
  profile: PrivacyProfile;
  ner: boolean;
  surrogates: boolean;
  tier2: "off" | "auto" | "all";
  threshold: number;
  review_size: number;
  signed_off_at: string | null;
  review: PrivacyReview | null;
  needs_signoff: boolean;
  blocked: boolean;
}
export interface CorpusConversation {
  id: string;
  source_id: string | null;
  upload_id: string | null;
  cluster: number | null;
  turns: { role: string; text: string }[];
}
export interface CorpusOverview {
  conversations: number;
  unique: number;
  exact_duplicates: number;
  near_duplicates: number;
  clusters: { id: number; label: string; size: number; share: number; terms: string[]; example: string | null }[];
  stale?: string[];
  [k: string]: unknown;
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
  repeats?: number; // attempts per task in a row (sampled when temperature > 0)
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
  updateWorkspace: (id: string, patch: Partial<Pick<Workspace, "name" | "default_model" | "default_pack" | "default_episodes" | "builder_model" | "train_base_model" | "monthly_spend_limit_usd">>) =>
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
  deletePack: (ws: string, name: string) => api<void>(`/api/workspaces/${ws}/packs/${encodeURIComponent(name)}`, { method: "DELETE" }),
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
  // the user's click on the builder's cost estimate: the only way extraction beyond the free few is approved
  approveExtraction: (ws: string, id: string, quote_id: string) =>
    api<{ job_id: string }>(`/api/workspaces/${ws}/builder/projects/${id}/extraction/approve`, { method: "POST", body: JSON.stringify({ quote_id }) }),
  declineExtraction: (ws: string, id: string) =>
    api<void>(`/api/workspaces/${ws}/builder/projects/${id}/extraction/quote`, { method: "DELETE" }),
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

async function authHeaders(extra: Record<string, string> = {}): Promise<Record<string, string>> {
  const headers: Record<string, string> = { ...extra };
  if (tokenGetter) {
    const t = await tokenGetter();
    if (t) headers.Authorization = `Bearer ${t}`;
  }
  return headers;
}

/**
 * Upload a chat-log file of any size to a builder project in CHUNK-sized pieces.
 * Resumable: the upload id is remembered per file (name + size + mtime), and the server
 * says how many bytes it already has, so re-picking the same file after an interruption
 * continues where it stopped. `onProgress(sent, total)` fires after every piece.
 */
export async function uploadBuilderFile(
  ws: string,
  project: string,
  file: File,
  onProgress?: (sent: number, total: number) => void,
): Promise<BuilderUpload> {
  const base = `/api/workspaces/${ws}/builder/projects/${project}/uploads`;
  const resumeKey = `envforge.upload.${project}.${file.name}.${file.size}.${file.lastModified}`;
  let up: BuilderUpload | null = null;
  try {
    const saved = localStorage.getItem(resumeKey);
    if (saved) {
      const proj = await Api.project(ws, project);
      up = proj.uploads.find((u) => u.id === saved && u.status === "receiving") ?? null;
    }
  } catch {}
  if (!up) {
    up = await api<BuilderUpload>(`${base}/init`, { method: "POST", body: JSON.stringify({ filename: file.name, size_bytes: file.size }) });
    try {
      localStorage.setItem(resumeKey, up.id);
    } catch {}
  }
  const chunk = up.chunk_bytes || 32 * 1024 * 1024;
  let offset = up.bytes_received;
  onProgress?.(offset, file.size);
  while (offset < file.size) {
    const res = await fetch(`${base}/${up.id}/chunk?offset=${offset}`, {
      method: "PUT",
      credentials: "include",
      headers: await authHeaders({ "Content-Type": "application/octet-stream" }),
      body: file.slice(offset, Math.min(offset + chunk, file.size)),
    });
    const body = await res.json().catch(() => ({}));
    if (res.status === 409 && typeof body?.detail?.bytes_received === "number") {
      offset = body.detail.bytes_received; // the server is elsewhere: resume from its position
      continue;
    }
    if (!res.ok) throw new ApiError(res.status, typeof body?.detail === "string" ? body.detail : res.statusText);
    offset = body.bytes_received;
    onProgress?.(offset, file.size);
  }
  const done = await api<BuilderUpload>(`${base}/${up.id}/complete`, { method: "POST" });
  try {
    localStorage.removeItem(resumeKey);
  } catch {}
  return done;
}

export const importBuilderUrl = (ws: string, project: string, url: string) =>
  api<BuilderUpload>(`/api/workspaces/${ws}/builder/projects/${project}/uploads/import`, { method: "POST", body: JSON.stringify({ url }) });

export interface PrivacyPatch {
  profile?: PrivacyProfile;
  confirm_off?: boolean;
  ner?: boolean;
  surrogates?: boolean;
  tier2?: "off" | "auto" | "all";
  threshold?: number;
}
export const setBuilderPrivacy = (ws: string, project: string, patch: PrivacyPatch) =>
  api<BuilderProject>(`/api/workspaces/${ws}/builder/projects/${project}/privacy`, { method: "PATCH", body: JSON.stringify(patch) });

export const builderCorpus = (ws: string, project: string) =>
  api<{ corpus: CorpusOverview | null }>(`/api/workspaces/${ws}/builder/projects/${project}/corpus`);

export const clusterExamples = (ws: string, project: string, cluster: number, n = 3) =>
  api<{ conversations: CorpusConversation[] }>(`/api/workspaces/${ws}/builder/projects/${project}/corpus/clusters/${cluster}?n=${n}`);

export const startPrivacyReview = (ws: string, project: string, n?: number) =>
  api<{ review: PrivacyReview; conversations: CorpusConversation[] }>(`/api/workspaces/${ws}/builder/projects/${project}/privacy/review`, {
    method: "POST",
    body: JSON.stringify(n ? { n } : {}),
  });

export const getPrivacyReview = (ws: string, project: string) =>
  api<{ review: PrivacyReview | null; conversations: CorpusConversation[] }>(`/api/workspaces/${ws}/builder/projects/${project}/privacy/review`);

export const submitPrivacyReview = (ws: string, project: string, reviewId: string, flagged: string[], notes?: Record<string, string>) =>
  api<BuilderProject>(`/api/workspaces/${ws}/builder/projects/${project}/privacy/review/submit`, {
    method: "POST",
    body: JSON.stringify({ review_id: reviewId, flagged, notes }),
  });

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

/** Delete a workspace with all its runs, builder projects and files (owner only). */
export const deleteWorkspace = (id: string) => api<void>(`/api/workspaces/${id}`, { method: "DELETE" });

// ---- GPU boxes, datasets, training, adapters (Phase 2/3 in the browser) ----

export interface BoxCheck {
  runtime: "docker" | "pod";
  support: { docker: boolean; pod: boolean } | null;
  checks: { name: string; ok: boolean; detail: string }[];
  gpus: { name: string; memory_total_mib: number | null; memory_used_mib: number | null; utilization_pct: number | null }[];
  driver: string | null;
  cuda: string | null;
  disk_free_gib: number | null;
  vllm_image: boolean | null;
  vllm_python: string | null;
  problems: string[];
}
export interface GpuBox {
  id: string;
  name: string;
  host: string;
  port: number;
  user: string;
  runtime: "docker" | "pod" | "auto";
  detected_runtime: "docker" | "pod" | null;
  ssh_key_source: "workspace" | "custom"; // the workspace's SSH key, or one pasted for this box
  ssh_key_fingerprint: string; // masked
  vllm_key: string | null; // masked
  vllm_key_generated: boolean;
  hf_token: string | null; // masked
  vllm_port: number;
  endpoint_host: string | null;
  bind: string | null;
  vllm_access: "direct" | "tunnel"; // tunnel = vLLM reached through an SSH port forward
  hourly_usd: number | null;
  status: "unknown" | "ok" | "degraded" | "unreachable";
  check_result: BoxCheck | null;
  checked_at: string | null;
  serve_state: "stopped" | "starting" | "serving" | "stopping" | "failed";
  served_model: string | null;
  served_adapters: string[];
  endpoint_url: string | null;
  serve_error: string | null;
  delete_failed: boolean; // the last removal couldn't reach the box; it can be forgotten instead
  created_at: string;
  job: { id: string; kind: string; status: string; progress: string } | null;
}
export interface WorkspaceSshKey {
  public_key: string; // one line, for RunPod / Vast "SSH public keys"
  fingerprint: string;
  created_at: string | null;
}
export interface BoxesResponse {
  boxes: GpuBox[];
  can_manage: boolean;
  admin_only: boolean;
}
export interface BoxIn {
  name: string;
  host: string;
  port: number;
  user: string;
  ssh_private_key?: string | null; // empty: the workspace SSH key
  runtime: "docker" | "pod" | "auto";
  vllm_port: number;
  endpoint_host?: string | null;
  bind?: string | null;
  vllm_access?: "direct" | "tunnel" | null; // null: picked from the host
  vllm_api_key?: string | null;
  hf_token?: string | null;
  hourly_usd?: number | null;
}

export interface DatasetSource {
  id: string;
  label: string;
  pack: string;
  model: string;
  episodes: number;
  seed_start: number;
  created_at: string;
  solve_rate: number | null;
  eligible: boolean;
  reason: string | null;
}
export type DatasetKind = "sft" | "preference";
export interface DatasetFilters {
  run_ids: string[];
  min_reward: number;
  val_frac: number;
  best_per_task: boolean;
  kind?: DatasetKind; // preference: chosen/rejected pairs of the same task, for DPO
  max_pairs_per_task?: number;
}
export interface DatasetPreview {
  kind?: DatasetKind;
  kept: number; // examples, or pairs for a preference dataset
  n_train: number;
  n_val: number;
  tasks: number;
  episodes_seen?: number;
  duplicates_dropped?: number;
  examples_per_task?: { min: number; max: number; mean: number };
  by_pack_tier?: Record<string, number>;
  models?: Record<string, number>;
  // preference datasets
  chosen_episodes?: number;
  rejected_episodes?: number;
  tasks_without_rejected?: number;
  tasks_without_chosen?: number;
  packs: string[];
  overlap: { run_id: string; label: string; model: string; tasks: number; sample: string[] }[];
  warnings: string[];
}
export interface DatasetRow {
  id: string;
  name: string;
  kind: DatasetKind;
  source_runs: string[];
  filters: { min_reward: number; val_frac: number; best_per_task?: boolean; max_pairs_per_task?: number };
  packs: string[];
  counts: DatasetPreview;
  n_train: number;
  n_val: number;
  created_at: string;
  tasks: Record<string, number[][]> | null;
}

export type TrainMethod = "sft" | "dpo" | "grpo";
export interface TrainParams {
  epochs: number;
  lr: number;
  lora_r: number;
  lora_alpha: number;
  max_len: number;
  batch_size: number;
  grad_accum: number;
  max_steps: number | null;
  // dpo
  beta?: number;
  // grpo: the loop, and each round's update
  rounds?: number;
  tasks_per_round?: number;
  group_size?: number;
  temperature?: number;
  kl_beta?: number;
  clip_eps?: number;
}
export interface GrpoRound {
  round: number;
  run_id: string;
  attempts: number;
  mean_reward: number;
  solve_rate: number;
  groups: number;
  informative_groups: number;
  trained_on: number;
}
export interface TrainingJob {
  id: string;
  status: "queued" | "starting" | "running" | "fetching" | "completed" | "failed" | "cancelled";
  adapter_name: string;
  base_model: string;
  method: TrainMethod;
  init_adapter: string | null; // dpo / grpo continue from this adapter
  pack: string | null; // grpo: the environment it learns in
  dataset_id: string | null;
  dataset_name: string | null;
  box_id: string | null;
  box_name: string | null;
  hyperparams: TrainParams;
  grpo: { round: number; rounds_total: number; phase: string; run_id: string | null; rounds: GrpoRound[] } | null;
  step: number;
  total_steps: number | null;
  epoch: number | null;
  last_loss: number | null;
  cancel_requested: boolean;
  error: string | null;
  elapsed_s: number;
  hourly_usd: number | null;
  est_cost_usd: number | null;
  adapter_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  polled_at: string | null;
  losses?: { step: number; loss?: number; eval_loss?: number; accuracy?: number; eval_accuracy?: number; kl?: number; round?: number }[];
  log_tail?: string;
}
export interface TrainingList {
  jobs: TrainingJob[];
  defaults: TrainParams & { base_model: string };
  method_defaults: Record<TrainMethod, Partial<TrainParams>>;
  can_manage: boolean;
}
export interface AdapterRow {
  id: string;
  name: string;
  model_spec: string;
  base_model: string;
  pack: string | null;
  dataset_id: string | null;
  dataset_name: string | null;
  dataset_deleted: boolean;
  training_job_id: string | null;
  lora_rank: number | null;
  metrics: {
    n_train?: number;
    global_step?: number;
    final_train_loss?: number | null;
    final_eval_loss?: number | null;
    final_eval_accuracy?: number | null; // dpo: share of held-out pairs it prefers correctly
    rounds?: GrpoRound[]; // grpo: each round's rollouts
    [k: string]: unknown;
  };
  method: TrainMethod;
  init_adapter: string | null; // dpo / grpo: the adapter it continued from
  source_runs: string[];
  test_run_id: string | null;
  served_on: { box_id: string; box_name: string; endpoint_url: string | null }[];
  created_at: string;
}

const W = (ws: string) => `/api/workspaces/${ws}`;
const post = (body?: unknown): RequestInit => ({ method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const Compute = {
  boxes: (ws: string) => api<BoxesResponse>(`${W(ws)}/gpu-boxes`),
  addBox: (ws: string, body: BoxIn) => api<GpuBox>(`${W(ws)}/gpu-boxes`, post(body)),
  // stops vLLM on the box, then forgets it (a job); force forgets it now without touching the machine
  deleteBox: (ws: string, id: string, force = false) =>
    api<void>(`${W(ws)}/gpu-boxes/${id}${force ? "?force=true" : ""}`, { method: "DELETE" }),
  checkBox: (ws: string, id: string) => api<GpuBox>(`${W(ws)}/gpu-boxes/${id}/check`, post()),
  serve: (ws: string, id: string, body: { model: string; adapters?: string[]; max_model_len?: number | null; gpu_memory_utilization?: number }) =>
    api<GpuBox>(`${W(ws)}/gpu-boxes/${id}/serve`, post(body)),
  stop: (ws: string, id: string) => api<GpuBox>(`${W(ws)}/gpu-boxes/${id}/stop`, post()),
  // replace the key vLLM requires (only while it is stopped); the key is never sent back
  setVllmKey: (ws: string, id: string, key: string) => api<GpuBox>(`${W(ws)}/gpu-boxes/${id}/vllm-key`, post({ key })),
  // the workspace's SSH key: created on first ask by someone who may manage boxes (null until then)
  sshKey: (ws: string) => api<{ key: WorkspaceSshKey | null }>(`${W(ws)}/gpu-boxes/ssh-key`),
  rotateSshKey: (ws: string) => api<{ key: WorkspaceSshKey | null }>(`${W(ws)}/gpu-boxes/ssh-key/rotate`, post()),

  datasetSources: (ws: string) => api<DatasetSource[]>(`${W(ws)}/datasets/sources`),
  previewDataset: (ws: string, f: DatasetFilters) => api<DatasetPreview>(`${W(ws)}/datasets/preview`, post(f)),
  datasets: (ws: string) => api<DatasetRow[]>(`${W(ws)}/datasets`),
  saveDataset: (ws: string, f: DatasetFilters & { name: string }) => api<DatasetRow>(`${W(ws)}/datasets`, post(f)),
  deleteDataset: (ws: string, id: string) => api<void>(`${W(ws)}/datasets/${id}`, { method: "DELETE" }),

  trainings: (ws: string) => api<TrainingList>(`${W(ws)}/training`),
  training: (ws: string, id: string) => api<TrainingJob>(`${W(ws)}/training/${id}`),
  startTraining: (
    ws: string,
    body: Partial<TrainParams> & { method?: TrainMethod; dataset_id?: string | null; box_id: string; base_model: string; adapter_name?: string; init_adapter?: string | null; pack?: string | null },
  ) =>
    api<TrainingJob>(`${W(ws)}/training`, post(body)),
  cancelTraining: (ws: string, id: string) => api<TrainingJob>(`${W(ws)}/training/${id}/cancel`, post()),

  adapters: (ws: string) => api<AdapterRow[]>(`${W(ws)}/adapters`),
  serveAdapter: (ws: string, id: string, box_id: string) => api<{ job_id: string }>(`${W(ws)}/adapters/${id}/serve`, post({ box_id })),
  testAdapter: (ws: string, id: string, box_id: string, episodes: number) =>
    api<{ job_id: string }>(`${W(ws)}/adapters/${id}/test`, post({ box_id, episodes })),
  deleteAdapter: (ws: string, id: string) => api<void>(`${W(ws)}/adapters/${id}`, { method: "DELETE" }),
};

/** Download a file from an /api/* GET (dataset JSONL, adapter tar.gz) with the session or Clerk token. */
export async function downloadFile(path: string, fallbackName: string): Promise<void> {
  const res = await fetch(path, { credentials: "include", headers: await authHeaders() });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body?.detail ?? res.statusText);
  }
  const blob = await res.blob();
  const name = /filename="?([^";]+)"?/.exec(res.headers.get("content-disposition") ?? "")?.[1] ?? fallbackName;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/** 3725 → "1h 2m". */
export const fmtDuration = (s: number) => {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${Math.floor(s % 60)}s` : `${Math.floor(s)}s`;
};