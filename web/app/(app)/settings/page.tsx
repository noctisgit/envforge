"use client";

// Workspace settings: the current workspace's name and run defaults, and creating
// new workspaces. Provider keys are per user and live on the Profile page.

import Link from "next/link";
import { useEffect, useState } from "react";
import { ModelPicker } from "@/components/ModelPicker";
import { Api, ApiError, type Pack } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

export default function WorkspaceSettingsPage() {
  const { current: ws, workspaces, reload, select } = useWorkspace();
  const [packs, setPacks] = useState<Pack[]>([]);
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [pack, setPack] = useState("");
  const [episodes, setEpisodes] = useState(10);
  const [builderModel, setBuilderModel] = useState("");
  const [trainBase, setTrainBase] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [newWs, setNewWs] = useState("");

  useEffect(() => {
    if (ws) Api.wsPacks(ws.id).then(setPacks).catch(() => {});
  }, [ws]);
  useEffect(() => {
    if (!ws) return;
    setName(ws.name);
    setModel(ws.default_model);
    setPack(ws.default_pack);
    setEpisodes(ws.default_episodes);
    setBuilderModel(ws.builder_model);
    setTrainBase(ws.train_base_model);
    setMsg(null);
  }, [ws]);

  if (!ws) return null;
  const isOwner = ws.role === "owner";

  async function saveDefaults(e: React.FormEvent) {
    e.preventDefault();
    try {
      await Api.updateWorkspace(ws!.id, {
        name,
        default_model: model,
        default_pack: pack,
        default_episodes: episodes,
        builder_model: builderModel,
        train_base_model: trainBase,
      });
      await reload();
      setMsg("Saved.");
    } catch (ex) {
      setMsg(ex instanceof ApiError ? ex.message : String(ex));
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Workspace settings</h1>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          A workspace groups runs and holds their defaults. It&apos;s private to you. Provider keys belong to your account — see{" "}
          <Link href="/account" className="underline">
            Profile
          </Link>
          .
        </p>
      </div>

      <form onSubmit={saveDefaults} className="panel grid grid-cols-1 gap-4 p-5 md:grid-cols-4">
        <div className="md:col-span-4 flex items-center gap-3 text-sm font-semibold">
          {ws.name}
          <span className="mono text-xs font-normal" style={{ color: "var(--muted)" }}>
            {ws.slug}
          </span>
        </div>
        <div>
          <label className="label">Name</label>
          <input className="input" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} disabled={!isOwner} />
        </div>
        <div className="md:col-span-2">
          <label className="label">Default model</label>
          <ModelPicker wsId={ws.id} value={model} onChange={setModel} disabled={!isOwner} />
        </div>
        <div>
          <label className="label">Default episodes</label>
          <input className="input" type="number" min={1} max={1000} value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} disabled={!isOwner} />
        </div>
        <div>
          <label className="label">Default environment</label>
          <select className="input" value={pack} onChange={(e) => setPack(e.target.value)} disabled={!isOwner}>
            {packs.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
        <div className="md:col-span-4 mt-2 border-t pt-4 text-sm font-semibold" style={{ borderColor: "var(--line)" }}>
          Models for building and training
        </div>
        <div className="md:col-span-2">
          <label className="label">Builder model</label>
          <ModelPicker wsId={ws.id} value={builderModel} onChange={setBuilderModel} disabled={!isOwner} />
          <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
            Drives the environment builder chat and task extraction. A strong model works best (e.g. anthropic:claude-opus-5-5).
          </p>
        </div>
        <div className="md:col-span-2">
          <label className="label">Training base model</label>
          <input className="input mono" value={trainBase} maxLength={200} onChange={(e) => setTrainBase(e.target.value)} disabled={!isOwner} placeholder="Qwen/Qwen2.5-7B-Instruct" />
          <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
            The open-weight model (Hugging Face id) you fine-tune with <span className="mono">envforge train sft</span> and serve with vLLM. See docs/TRAINING.md.
          </p>
        </div>
        {isOwner && (
          <div className="md:col-span-4 flex items-end gap-3">
            <button className="btn">Save defaults</button>
            {msg && (
              <span className="text-sm" style={{ color: "var(--accent)" }}>
                {msg}
              </span>
            )}
          </div>
        )}
      </form>

      <div className="panel space-y-3 p-5">
        <div className="text-sm font-semibold">Your workspaces</div>
        <ul className="text-sm">
          {workspaces.map((w) => (
            <li key={w.id} className="flex items-center gap-3 border-t py-2" style={{ borderColor: "var(--line)" }}>
              <span className="flex-1">{w.name}</span>
              <span className="mono text-xs" style={{ color: "var(--muted)" }}>
                {w.default_model}
              </span>
              {w.id === ws.id ? (
                <span className="pill pill-completed">current</span>
              ) : (
                <button className="btn-ghost py-0.5" onClick={() => select(w.id)}>
                  Switch
                </button>
              )}
            </li>
          ))}
        </ul>
        <div className="flex items-end gap-3 border-t pt-3" style={{ borderColor: "var(--line)" }}>
          <div className="flex-1">
            <label className="label">New workspace</label>
            <input className="input" placeholder="e.g. qwen-baselines" maxLength={120} value={newWs} onChange={(e) => setNewWs(e.target.value)} />
          </div>
          <button
            className="btn"
            disabled={!newWs.trim()}
            onClick={async () => {
              const created = await Api.createWorkspace(newWs.trim());
              setNewWs("");
              await reload();
              select(created.id);
            }}
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}
