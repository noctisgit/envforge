"use client";

import { useEffect, useState } from "react";
import { Api, ApiError, type Pack, type Provider } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const PROVIDERS: { id: Provider; label: string; hint: string; needsUrl?: boolean }[] = [
  { id: "anthropic", label: "Anthropic", hint: "sk-ant-… · model ids like claude-sonnet-4-5" },
  { id: "openai", label: "OpenAI", hint: "sk-… · model ids like gpt-4o-mini" },
  { id: "vercel", label: "Vercel AI Gateway", hint: "AI Gateway key · model ids like anthropic/claude-opus-5 or openai/gpt-4o-mini" },
  { id: "vllm", label: "vLLM (self-hosted)", hint: "your GPU box, e.g. http://100.x.y.z:8000/v1 · key optional", needsUrl: true },
];

export default function SettingsPage() {
  const { current: ws, reload } = useWorkspace();
  const [packs, setPacks] = useState<Pack[]>([]);
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [pack, setPack] = useState("");
  const [episodes, setEpisodes] = useState(10);
  const [msg, setMsg] = useState<string | null>(null);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [newWs, setNewWs] = useState("");

  useEffect(() => {
    Api.packs().then(setPacks).catch(() => {});
  }, []);
  useEffect(() => {
    if (!ws) return;
    setName(ws.name);
    setModel(ws.default_model);
    setPack(ws.default_pack);
    setEpisodes(ws.default_episodes);
    const u: Record<string, string> = {};
    ws.keys.forEach((k) => {
      if (k.base_url) u[k.provider] = k.base_url;
    });
    setUrls(u);
  }, [ws]);

  if (!ws) return null;
  const isOwner = ws.role === "owner";

  async function saveDefaults(e: React.FormEvent) {
    e.preventDefault();
    try {
      await Api.updateWorkspace(ws!.id, { name, default_model: model, default_pack: pack, default_episodes: episodes });
      await reload();
      setMsg("Saved.");
    } catch (ex) {
      setMsg(ex instanceof ApiError ? ex.message : String(ex));
    }
  }

  async function saveKey(p: Provider) {
    try {
      await Api.putKey(ws!.id, p, secrets[p] ?? "", urls[p]);
      setSecrets({ ...secrets, [p]: "" });
      await reload();
      setMsg(`${p} key saved.`);
    } catch (ex) {
      setMsg(ex instanceof ApiError ? ex.message : String(ex));
    }
  }

  async function removeKey(p: Provider) {
    await Api.deleteKey(ws!.id, p);
    await reload();
    setMsg(`${p} key removed.`);
  }

  return (
    <div className="space-y-6">
      {msg && (
        <div className="text-sm" style={{ color: "var(--accent)" }}>
          {msg}
        </div>
      )}

      <form onSubmit={saveDefaults} className="panel grid grid-cols-1 gap-4 p-5 md:grid-cols-4">
        <div className="md:col-span-4 text-sm font-semibold">Workspace defaults</div>
        <div>
          <label className="label">Name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} disabled={!isOwner} />
        </div>
        <div>
          <label className="label">Default model</label>
          <input className="input mono" value={model} onChange={(e) => setModel(e.target.value)} disabled={!isOwner} />
        </div>
        <div>
          <label className="label">Default pack</label>
          <select className="input" value={pack} onChange={(e) => setPack(e.target.value)} disabled={!isOwner}>
            {packs.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Default episodes</label>
          <input className="input" type="number" min={1} max={1000} value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} disabled={!isOwner} />
        </div>
        {isOwner && (
          <div className="md:col-span-4">
            <button className="btn">Save defaults</button>
          </div>
        )}
      </form>

      <div className="panel space-y-5 p-5">
        <div className="text-sm font-semibold">Provider keys</div>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Keys are encrypted at rest and never shown again after saving. Runs in this workspace use these keys; a
          model whose provider has no key here cannot be queued.
        </p>
        {PROVIDERS.map((p) => {
          const existing = ws.keys.find((k) => k.provider === p.id);
          return (
            <div key={p.id} className="grid grid-cols-1 gap-3 border-t pt-4 md:grid-cols-6" style={{ borderColor: "var(--line)" }}>
              <div className="md:col-span-2">
                <div className="font-medium">{p.label}</div>
                <div className="text-xs" style={{ color: "var(--muted)" }}>
                  {p.hint}
                </div>
                {existing && (
                  <div className="mt-1 text-xs">
                    saved <span className="mono">{existing.masked}</span> · {new Date(existing.updated_at).toLocaleDateString()}
                  </div>
                )}
              </div>
              {p.needsUrl && (
                <div className="md:col-span-2">
                  <label className="label">Base URL</label>
                  <input className="input mono" value={urls[p.id] ?? ""} onChange={(e) => setUrls({ ...urls, [p.id]: e.target.value })} disabled={!isOwner} />
                </div>
              )}
              <div className={p.needsUrl ? "md:col-span-1" : "md:col-span-3"}>
                <label className="label">{existing ? "Replace key" : "API key"}</label>
                <input className="input mono" type="password" autoComplete="off" value={secrets[p.id] ?? ""} onChange={(e) => setSecrets({ ...secrets, [p.id]: e.target.value })} disabled={!isOwner} />
              </div>
              {isOwner && (
                <div className="flex items-end gap-2">
                  <button className="btn" type="button" onClick={() => saveKey(p.id)} disabled={!(secrets[p.id] || (p.needsUrl && urls[p.id]))}>
                    Save
                  </button>
                  {existing && (
                    <button className="btn-ghost" type="button" onClick={() => removeKey(p.id)}>
                      Remove
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="panel flex items-end gap-3 p-5">
        <div className="flex-1">
          <div className="mb-1 text-sm font-semibold">New workspace</div>
          <input className="input" placeholder="e.g. team-alpha" value={newWs} onChange={(e) => setNewWs(e.target.value)} />
        </div>
        <button
          className="btn"
          disabled={!newWs.trim()}
          onClick={async () => {
            await Api.createWorkspace(newWs.trim());
            setNewWs("");
            await reload();
          }}
        >
          Create
        </button>
      </div>
    </div>
  );
}
