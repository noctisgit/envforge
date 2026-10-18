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
}

export interface Workspace {
  id: string;
  slug: string;
  name: string;
  role: "owner" | "member";
  default_model: string;
  default_pack: string;
  default_episodes: number;
  keys: KeyOut[];
}

export interface Pack {
  name: string;
  kind: string;
  tiers: number[];
  max_steps: number;
  description: string;
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
  updateWorkspace: (id: string, patch: Partial<Pick<Workspace, "name" | "default_model" | "default_pack" | "default_episodes">>) =>
    api<Workspace>(`/api/workspaces/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  putKey: (id: string, provider: Provider, secret: string, base_url?: string) =>
    api<KeyOut[]>(`/api/workspaces/${id}/keys`, { method: "PUT", body: JSON.stringify({ provider, secret, base_url: base_url || null }) }),
  deleteKey: (id: string, provider: Provider) => api<KeyOut[]>(`/api/workspaces/${id}/keys/${provider}`, { method: "DELETE" }),

  runs: (ws: string) => api<Run[]>(`/api/workspaces/${ws}/runs`),
  run: (ws: string, id: string) => api<Run>(`/api/workspaces/${ws}/runs/${id}`),
  createRun: (ws: string, body: Partial<Run> & { tiers?: number[] }) =>
    api<Run>(`/api/workspaces/${ws}/runs`, { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (ws: string, id: string) => api<Run>(`/api/workspaces/${ws}/runs/${id}/cancel`, { method: "POST" }),
  episodes: (ws: string, id: string) => api<EpisodeSummary[]>(`/api/workspaces/${ws}/runs/${id}/episodes`),
  episode: (ws: string, id: string, ep: number) => api<Trajectory>(`/api/workspaces/${ws}/runs/${id}/episodes/${ep}`),
};
