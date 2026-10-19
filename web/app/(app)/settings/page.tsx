"use client";

// Workspace settings: the current workspace's name and run defaults, creating new
// workspaces, and deleting this one. Provider keys are per user and live on the
// Profile page.

import Link from "next/link";
import { useEffect, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { HubModelPicker } from "@/components/HubModelPicker";
import { ModelPicker } from "@/components/ModelPicker";
import { Page, PageHeader, Panel } from "@/components/Page";
import { Api, ApiError, deleteWorkspace, type Pack } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const msgOf = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

export default function WorkspaceSettingsPage() {
  const { current: ws, workspaces, reload, select } = useWorkspace();
  const confirm = useConfirm();
  const [packs, setPacks] = useState<Pack[]>([]);
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [pack, setPack] = useState("");
  const [episodes, setEpisodes] = useState(10);
  const [builderModel, setBuilderModel] = useState("");
  const [trainBase, setTrainBase] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [newWs, setNewWs] = useState("");
  const [createMsg, setCreateMsg] = useState<string | null>(null);
  const [deleteMsg, setDeleteMsg] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (ws)
      Api.wsPacks(ws.id)
        .then(setPacks)
        .catch(() => {});
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
    setDeleteMsg(null);
  }, [ws]);

  if (!ws) return null;
  const isOwner = ws.role === "owner";
  const others = workspaces.filter((w) => w.id !== ws.id);

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
      setMsg(msgOf(ex));
    }
  }

  async function createWs(e: React.FormEvent) {
    e.preventDefault();
    setCreateMsg(null);
    try {
      const created = await Api.createWorkspace(newWs.trim());
      setNewWs("");
      await reload();
      select(created.id);
    } catch (ex) {
      setCreateMsg(msgOf(ex));
    }
  }

  async function removeWs() {
    const target = ws!;
    const ok = await confirm({
      title: `Delete workspace “${target.name}”?`,
      body: (
        <div className="space-y-2">
          <p>
            This permanently deletes every run and its recorded episodes, every builder project with its uploads, revisions and published environments, and the
            workspace&apos;s usage history. It can&apos;t be undone.
          </p>
          <p>Your provider keys belong to your account and are not affected.</p>
        </div>
      ),
      confirmLabel: "Delete workspace",
      tone: "danger",
      requireText: target.name,
    });
    if (!ok) return;
    setDeleting(true);
    setDeleteMsg(null);
    try {
      await deleteWorkspace(target.id);
      // move to another workspace before the list refreshes, so nothing points at the gone one
      if (others[0]) select(others[0].id);
      await reload();
    } catch (ex) {
      setDeleteMsg(msgOf(ex));
    } finally {
      setDeleting(false);
    }
  }

  return (
    <Page>
      <PageHeader
        title="Workspace settings"
        description={
          <>
            A workspace groups runs and holds their defaults. It&apos;s private to you. Provider keys belong to your account — see{" "}
            <Link href="/account" className="underline">
              Profile
            </Link>
            .
          </>
        }
      />

      <form onSubmit={saveDefaults} className="grid grid-cols-1 items-start gap-6 xl:grid-cols-2">
        <Panel
          title="General"
          hint={
            <>
              <span className="mono">{ws.slug}</span> · {isOwner ? "you own this workspace" : "only the owner can change these"}
            </>
          }
        >
          <div className="grid grid-cols-1 items-end gap-4 md:grid-cols-2">
            <div className="md:col-span-2">
              <label className="label" htmlFor="ws-name">
                Name
              </label>
              <input id="ws-name" className="input" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} disabled={!isOwner} />
            </div>
            <div className="md:col-span-2">
              <label className="label">Default model</label>
              <ModelPicker wsId={ws.id} value={model} onChange={setModel} disabled={!isOwner} />
            </div>
            <div>
              <label className="label" htmlFor="ws-pack">
                Default environment
              </label>
              <select id="ws-pack" className="input" value={pack} onChange={(e) => setPack(e.target.value)} disabled={!isOwner}>
                {packs.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label" htmlFor="ws-episodes">
                Default episodes
              </label>
              <input
                id="ws-episodes"
                className="input"
                type="number"
                min={1}
                max={1000}
                value={episodes}
                onChange={(e) => setEpisodes(Number(e.target.value))}
                disabled={!isOwner}
              />
            </div>
          </div>
        </Panel>

        <Panel title="Models for building and training">
          <div className="space-y-4">
            <div>
              <label className="label">Builder model</label>
              <ModelPicker wsId={ws.id} value={builderModel} onChange={setBuilderModel} disabled={!isOwner} />
              <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                Drives the environment builder chat and task extraction. A strong model works best (e.g. anthropic:claude-opus-5-5).
              </p>
            </div>
            <div>
              <label className="label">Training base model</label>
              <HubModelPicker value={trainBase} onChange={setTrainBase} disabled={!isOwner} />
              <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                The open-weight model (Hugging Face id) you fine-tune with <span className="mono">envforge train sft</span> and serve with vLLM. See
                docs/TRAINING.md.
              </p>
            </div>
          </div>
        </Panel>

        {isOwner && (
          <div className="flex items-center gap-3 xl:col-span-2">
            <button className="btn">Save settings</button>
            {msg && (
              <span className="text-sm" style={{ color: msg === "Saved." ? "var(--accent)" : "var(--bad)" }}>
                {msg}
              </span>
            )}
          </div>
        )}
      </form>

      <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-2">
        <Panel title="Your workspaces" hint="Switch here or from the menu in the header.">
          <ul className="text-sm">
            {workspaces.map((w) => (
              <li key={w.id} className="flex min-h-11 items-center gap-3 border-t py-2 first:border-t-0 first:pt-0" style={{ borderColor: "var(--line)" }}>
                <div className="min-w-0 flex-1">
                  <div className="truncate">{w.name}</div>
                  <div className="mono truncate text-xs" style={{ color: "var(--muted)" }}>
                    {w.default_model}
                  </div>
                </div>
                {w.role !== "owner" && <span className="pill pill-queued">member</span>}
                {w.id === ws.id ? (
                  <span className="pill pill-completed">current</span>
                ) : (
                  <button type="button" className="btn-ghost" onClick={() => select(w.id)}>
                    Switch
                  </button>
                )}
              </li>
            ))}
          </ul>
          <form onSubmit={createWs} className="flex items-end gap-3 border-t pt-4" style={{ borderColor: "var(--line)" }}>
            <div className="min-w-0 flex-1">
              <label className="label" htmlFor="new-ws">
                New workspace
              </label>
              <input id="new-ws" className="input" placeholder="e.g. qwen-baselines" maxLength={120} value={newWs} onChange={(e) => setNewWs(e.target.value)} />
            </div>
            <button className="btn btn-field" disabled={!newWs.trim()}>
              Create
            </button>
          </form>
          {createMsg && (
            <div className="text-sm" style={{ color: "var(--bad)" }}>
              {createMsg}
            </div>
          )}
        </Panel>

        {isOwner && (
          <Panel title="Danger zone" tone="danger">
            <div className="flex flex-wrap items-center gap-4">
              <div className="min-w-0 flex-1 text-sm">
                <div className="font-medium">Delete this workspace</div>
                <div style={{ color: "var(--muted)" }}>
                  {others.length === 0
                    ? "This is your only workspace. Create another one first; you always keep at least one."
                    : `Permanently deletes “${ws.name}” with all its runs, episodes, builder projects and published environments. Queued or running work must be cancelled first.`}
                </div>
              </div>
              <button type="button" className="btn-danger" disabled={deleting || others.length === 0} onClick={removeWs}>
                {deleting ? "Deleting…" : "Delete workspace"}
              </button>
            </div>
            {deleteMsg && (
              <div className="text-sm" style={{ color: "var(--bad)" }}>
                {deleteMsg}
              </div>
            )}
          </Panel>
        )}
      </div>
    </Page>
  );
}
