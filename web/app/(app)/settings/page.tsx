"use client";

// Workspace settings: the current workspace's name and run defaults, creating new
// workspaces, and deleting this one. Provider keys are per user and live on the
// Profile page.

import Link from "next/link";
import { useEffect, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { HubModelPicker } from "@/components/HubModelPicker";
import { ModelPicker } from "@/components/ModelPicker";
import { WorkspaceMark } from "@/components/Avatar";
import { Page, PageHeader, Panel } from "@/components/Page";
import { SettingsTabs } from "@/components/SettingsTabs";
import { errorText, useToast } from "@/components/Toast";
import { Api, deleteWorkspace, type Pack } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

export default function WorkspaceSettingsPage() {
  const { current: ws, workspaces, reload, select } = useWorkspace();
  const confirm = useConfirm();
  const toast = useToast();
  const [packs, setPacks] = useState<Pack[]>([]);
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [pack, setPack] = useState("");
  const [episodes, setEpisodes] = useState(10);
  const [builderModel, setBuilderModel] = useState("");
  const [trainBase, setTrainBase] = useState("");
  const [limit, setLimit] = useState(""); // USD per month; empty = no limit
  const [saving, setSaving] = useState(false);
  const [newWs, setNewWs] = useState("");
  const [createErr, setCreateErr] = useState<string | null>(null);
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
    setLimit(ws.monthly_spend_limit_usd === null ? "" : String(ws.monthly_spend_limit_usd));
  }, [ws]);

  if (!ws) return null;
  const isOwner = ws.role === "owner";
  const others = workspaces.filter((w) => w.id !== ws.id);

  async function saveDefaults(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    try {
      await Api.updateWorkspace(ws!.id, {
        name,
        default_model: model,
        default_pack: pack,
        default_episodes: episodes,
        builder_model: builderModel,
        train_base_model: trainBase,
        monthly_spend_limit_usd: limit.trim() === "" ? null : Number(limit),
      });
      await reload();
      toast.success("Settings saved", `Defaults for ${name || ws!.name} are updated.`);
    } catch (ex) {
      toast.error("Couldn't save settings", ex);
    } finally {
      setSaving(false);
    }
  }

  async function createWs(e: React.FormEvent) {
    e.preventDefault();
    setCreateErr(null);
    try {
      const created = await Api.createWorkspace(newWs.trim());
      setNewWs("");
      await reload();
      select(created.id);
      toast.success("Workspace created", `Switched to ${created.name}.`);
    } catch (ex) {
      setCreateErr(errorText(ex)); // a name problem belongs next to the field
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
    try {
      await deleteWorkspace(target.id);
      // move to another workspace before the list refreshes, so nothing points at the gone one
      if (others[0]) select(others[0].id);
      await reload();
      toast.success("Workspace deleted", `“${target.name}” and everything in it are gone.`);
    } catch (ex) {
      toast.error("Couldn't delete the workspace", ex);
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
            Defaults for new runs and the builder in <span className="mono">{ws.name}</span>. A workspace is private to you. Provider keys belong to your account,
            not the workspace — manage them under{" "}
            <Link href="/account#keys" className="underline">
              Account
            </Link>
            .
          </>
        }
      />
      <SettingsTabs active="general" />

      <div className="flex flex-wrap items-start gap-6">
        <form onSubmit={saveDefaults} aria-label="Defaults" className="panel min-w-0 flex-[2_1_520px] space-y-5 p-5">
          <div className="flex items-center gap-3">
            <WorkspaceMark name={ws.name} size={32} />
            <div className="min-w-0">
              <div className="truncate text-lg font-semibold">{ws.name}</div>
              <div className="mono hint truncate">
                {ws.slug} · {isOwner ? "owner" : "member — only the owner can change these"}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2">
            <div>
              <label className="label" htmlFor="ws-name">
                Name
              </label>
              <input id="ws-name" className="input" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} disabled={!isOwner} />
            </div>
            <div>
              <label className="label">Default model</label>
              <ModelPicker wsId={ws.id} value={model} onChange={setModel} disabled={!isOwner} />
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
          </div>

          <div className="space-y-4 border-t pt-4" style={{ borderColor: "var(--line)" }}>
            <div>
              <div className="font-semibold">Models for building and training</div>
              <div className="hint">these are billed to your own provider keys</div>
            </div>
            <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2">
              <div>
                <label className="label">Builder model</label>
                <ModelPicker wsId={ws.id} value={builderModel} onChange={setBuilderModel} disabled={!isOwner} />
                <p className="hint mt-1">Drives the environment-builder chat and task extraction. A strong model works best (e.g. anthropic:claude-opus-5-5).</p>
              </div>
              <div>
                <label className="label">Training base model</label>
                <HubModelPicker value={trainBase} onChange={setTrainBase} disabled={!isOwner} />
                <p className="hint mt-1">
                  The open-weight model (Hugging Face id) you fine-tune with <span className="mono">envforge train sft</span> and serve with vLLM ·
                  docs/TRAINING.md
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-2 border-t pt-4" style={{ borderColor: "var(--line)" }}>
            <label className="label" htmlFor="ws-limit">
              Monthly spending limit (USD)
            </label>
            <input
              id="ws-limit"
              className="input md:max-w-xs"
              type="number"
              min={0}
              step="0.01"
              placeholder="no limit"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              disabled={!isOwner}
              aria-describedby="ws-limit-hint"
            />
            <p id="ws-limit-hint" className="hint">
              ${ws.month_spend_usd.toFixed(2)} spent this month (UTC) on model APIs: runs and the builder. At the limit, new runs and builder messages are refused, a running
              run stops after its current episode and the builder pauses. Self-hosted (vLLM) and scripted models don&apos;t count. Leave empty for no limit.
            </p>
          </div>

          {isOwner && (
            <div className="flex justify-end border-t pt-4" style={{ borderColor: "var(--line)" }}>
              <button className="btn" disabled={saving}>
                {saving ? "Saving…" : "Save defaults"}
              </button>
            </div>
          )}
        </form>

        <div className="flex min-w-0 flex-[1_1_320px] flex-col gap-6">
          <Panel title="Your workspaces" flush>
            <ul>
              {workspaces.map((w) => (
                <li key={w.id} className="flex items-center gap-3 border-b px-4 py-3" style={{ borderColor: "var(--line)" }}>
                  <WorkspaceMark name={w.name} size={24} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-medium">{w.name}</div>
                    <div className="mono hint truncate">{w.default_model}</div>
                  </div>
                  {w.role !== "owner" && <span className="pill pill-queued">member</span>}
                  {w.id === ws.id ? (
                    <span className="pill pill-running">current</span>
                  ) : (
                    <button type="button" className="btn-ghost btn-sm" onClick={() => select(w.id)}>
                      Switch
                    </button>
                  )}
                </li>
              ))}
            </ul>
            <form onSubmit={createWs} className="space-y-2 p-4">
              <label className="label mb-0" htmlFor="new-ws">
                New workspace
              </label>
              <div className="flex gap-2">
                <input
                  id="new-ws"
                  className="input"
                  placeholder="e.g. sonnet-eval"
                  maxLength={120}
                  value={newWs}
                  onChange={(e) => {
                    setNewWs(e.target.value);
                    setCreateErr(null);
                  }}
                  aria-invalid={!!createErr}
                  aria-describedby={createErr ? "new-ws-err" : undefined}
                />
                <button className="btn-ghost btn-field" disabled={!newWs.trim()}>
                  Create
                </button>
              </div>
              {createErr && (
                <div id="new-ws-err" className="text-sm" style={{ color: "var(--bad)" }}>
                  {createErr}
                </div>
              )}
              <div className="hint">Each workspace has its own runs, defaults and builder projects.</div>
            </form>
          </Panel>

          {isOwner && (
            <Panel title="Danger zone" tone="danger">
              <div className="space-y-3">
                <div>
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
            </Panel>
          )}
        </div>
      </div>
    </Page>
  );
}
