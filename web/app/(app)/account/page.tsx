"use client";

// Your account: profile, password, provider keys and sessions. Provider keys live on
// the account, so every workspace you run in uses the same keys.

import { useEffect, useState } from "react";
import { Avatar } from "@/components/AvatarMenu";
import { useConfirm } from "@/components/ConfirmDialog";
import { invalidateModelCache } from "@/components/ModelPicker";
import { Page, PageHeader, Panel } from "@/components/Page";
import { Api, ApiError, type KeyOut, type Provider } from "@/lib/api";
import { useAuthState } from "@/lib/auth";

const PROVIDERS: { id: Provider; label: string; hint: string; needsUrl?: boolean }[] = [
  { id: "anthropic", label: "Anthropic", hint: "sk-ant-… · checked with Anthropic when saved; models like claude-sonnet-5-5" },
  { id: "openai", label: "OpenAI", hint: "sk-… · checked with OpenAI when saved; models like gpt-5-mini" },
  { id: "vercel", label: "Vercel AI Gateway", hint: "AI Gateway key · 400+ models, ids like anthropic/claude-sonnet-4.5" },
  {
    id: "vllm",
    label: "vLLM (self-hosted)",
    hint: "your GPU box (public host or Tailscale 100.x), e.g. http://100.x.y.z:8000/v1 · key optional · docs/COMPUTE.md",
    needsUrl: true,
  },
];

const msgOf = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

export default function AccountPage() {
  const { user, refresh } = useAuthState();
  const confirm = useConfirm();
  const [name, setName] = useState(user?.display_name ?? "");
  const [profileMsg, setProfileMsg] = useState<string | null>(null);
  const [pw, setPw] = useState({ current: "", next: "", again: "" });
  const [pwMsg, setPwMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [keys, setKeys] = useState<KeyOut[]>([]);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [keyMsg, setKeyMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    Api.keys()
      .then((ks) => {
        setKeys(ks);
        setUrls(Object.fromEntries(ks.filter((k) => k.base_url).map((k) => [k.provider, k.base_url!])));
      })
      .catch(() => {});
  }, []);

  if (!user) return null;
  const local = user.auth_provider === "local";

  async function saveProfile(e: React.FormEvent) {
    e.preventDefault();
    try {
      await Api.updateProfile(name);
      await refresh();
      setProfileMsg("Saved.");
    } catch (ex) {
      setProfileMsg(msgOf(ex));
    }
  }

  async function savePassword(e: React.FormEvent) {
    e.preventDefault();
    if (pw.next !== pw.again) return setPwMsg({ ok: false, text: "The new passwords don't match." });
    try {
      await Api.changePassword(pw.current, pw.next);
      setPw({ current: "", next: "", again: "" });
      setPwMsg({ ok: true, text: "Password changed. Other devices have been signed out." });
    } catch (ex) {
      setPwMsg({ ok: false, text: msgOf(ex) });
    }
  }

  async function saveKey(p: Provider) {
    setBusy(p);
    setKeyMsg(null);
    try {
      setKeys(await Api.putKey(p, secrets[p] ?? "", urls[p]));
      setSecrets({ ...secrets, [p]: "" });
      invalidateModelCache();
      setKeyMsg({ ok: true, text: `${p} key saved.` });
    } catch (ex) {
      setKeyMsg({ ok: false, text: msgOf(ex) });
    } finally {
      setBusy(null);
    }
  }

  async function removeKey(p: Provider, label: string) {
    const ok = await confirm({
      title: `Remove your ${label} key?`,
      body: "Runs in all of your workspaces will stop being able to use this provider until you add a key again. Queued runs that need it will fail.",
      confirmLabel: "Remove key",
      tone: "danger",
    });
    if (!ok) return;
    setKeys(await Api.deleteKey(p));
    invalidateModelCache();
    setKeyMsg({ ok: true, text: `${p} key removed.` });
  }

  const sessions = (
    <Panel title="Sessions">
      <div className="flex flex-wrap items-center gap-4">
        <div className="min-w-0 flex-1 text-sm" style={{ color: "var(--muted)" }}>
          Sign out of envforge on every device and browser, including this one.
        </div>
        <button
          className="btn-danger"
          onClick={async () => {
            const ok = await confirm({
              title: "Sign out everywhere?",
              body: "Every browser and device signed in to your account will need to sign in again, including this one.",
              confirmLabel: "Sign out everywhere",
              tone: "danger",
            });
            if (!ok) return;
            await Api.logoutAll();
            window.location.href = "/login";
          }}
        >
          Sign out everywhere
        </button>
      </div>
    </Panel>
  );

  return (
    <Page>
      <PageHeader
        title="Profile"
        description="Your account: profile, password, provider keys and sessions. Provider keys live on your account, so every workspace you run in uses the same keys."
      />

      <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-2">
        <Panel title="Profile">
          <div className="flex items-center gap-4">
            <Avatar name={user.display_name} email={user.email} size={48} />
            <div className="min-w-0">
              <div className="truncate font-medium">{user.display_name || user.email}</div>
              <div className="truncate text-sm" style={{ color: "var(--muted)" }}>
                {user.email} · {user.is_admin ? "admin · " : ""}signed in with {local ? "email & password" : "Clerk"}
              </div>
            </div>
          </div>
          <form onSubmit={saveProfile} className="grid grid-cols-1 items-end gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
            <div>
              <label className="label" htmlFor="name">
                Display name
              </label>
              <input id="name" className="input" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
              <label className="label" htmlFor="email">
                Email
              </label>
              <input id="email" className="input" value={user.email} disabled />
            </div>
            <button className="btn btn-field">Save</button>
          </form>
          {profileMsg && (
            <div className="text-sm" style={{ color: profileMsg === "Saved." ? "var(--accent)" : "var(--bad)" }}>
              {profileMsg}
            </div>
          )}
        </Panel>

        {local ? (
          <Panel title="Change password" hint="Changing it signs out your other devices.">
            <form onSubmit={savePassword} className="space-y-4">
              <div className="grid grid-cols-1 items-end gap-4 md:grid-cols-3">
                <div>
                  <label className="label" htmlFor="pw-current">
                    Current password
                  </label>
                  <input
                    id="pw-current"
                    className="input"
                    type="password"
                    autoComplete="current-password"
                    value={pw.current}
                    onChange={(e) => setPw({ ...pw, current: e.target.value })}
                  />
                </div>
                <div>
                  <label className="label" htmlFor="pw-next">
                    New password
                  </label>
                  <input
                    id="pw-next"
                    className="input"
                    type="password"
                    placeholder="10+ characters"
                    autoComplete="new-password"
                    minLength={10}
                    value={pw.next}
                    onChange={(e) => setPw({ ...pw, next: e.target.value })}
                  />
                </div>
                <div>
                  <label className="label" htmlFor="pw-again">
                    Repeat new password
                  </label>
                  <input
                    id="pw-again"
                    className="input"
                    type="password"
                    autoComplete="new-password"
                    value={pw.again}
                    onChange={(e) => setPw({ ...pw, again: e.target.value })}
                  />
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <button className="btn" disabled={!pw.current || pw.next.length < 10}>
                  Change password
                </button>
                {pwMsg && (
                  <span className="text-sm" style={{ color: pwMsg.ok ? "var(--accent)" : "var(--bad)" }}>
                    {pwMsg.text}
                  </span>
                )}
              </div>
            </form>
          </Panel>
        ) : (
          sessions
        )}
      </div>

      <Panel
        title="Provider keys"
        hint="Your keys, used by every workspace you run in. They're encrypted at rest, never shown again after saving, and only decrypted by the worker that executes your runs. Anthropic and OpenAI keys are checked with the provider when you save them, and the model picker then lists exactly the models your key can use."
      >
        {keyMsg && (
          <div className="text-sm" style={{ color: keyMsg.ok ? "var(--accent)" : "var(--bad)" }}>
            {keyMsg.text}
          </div>
        )}
        <div>
          {PROVIDERS.map((p) => {
            const existing = keys.find((k) => k.provider === p.id);
            return (
              // one grid for every provider: about | inputs | actions. Each column starts
              // with a label line, so inputs and buttons line up across rows.
              <div
                key={p.id}
                className="grid grid-cols-1 items-start gap-x-6 gap-y-3 border-t py-4 first:border-t-0 first:pt-0 last:pb-0 lg:grid-cols-[minmax(14rem,2fr)_minmax(0,3fr)_12rem]"
                style={{ borderColor: "var(--line)" }}
              >
                <div className="min-w-0">
                  <div className="font-medium">{p.label}</div>
                  <div className="text-xs" style={{ color: "var(--muted)" }}>
                    {p.hint}
                  </div>
                  {existing && (
                    <div className="mt-1 text-xs">
                      saved <span className="mono">{existing.masked}</span> · {new Date(existing.updated_at).toLocaleDateString()}
                      {existing.verified_at && (
                        <span style={{ color: "var(--good)" }}> · ✓ verified{existing.n_models !== null && ` · ${existing.n_models} models`}</span>
                      )}
                      {existing.verify_note && <div style={{ color: "var(--warn)" }}>{existing.verify_note}</div>}
                      {!existing.verified_at && !existing.verify_note && (p.id === "vercel" || p.id === "vllm") && (
                        <span style={{ color: "var(--muted)" }}> · checked on first run</span>
                      )}
                    </div>
                  )}
                </div>
                <div className={`grid grid-cols-1 gap-3 ${p.needsUrl ? "sm:grid-cols-2" : ""}`}>
                  {p.needsUrl && (
                    <div>
                      <label className="label" htmlFor={`url-${p.id}`}>
                        Base URL
                      </label>
                      <input
                        id={`url-${p.id}`}
                        className="input mono"
                        value={urls[p.id] ?? ""}
                        onChange={(e) => setUrls({ ...urls, [p.id]: e.target.value })}
                      />
                    </div>
                  )}
                  <div>
                    <label className="label" htmlFor={`key-${p.id}`}>
                      {existing ? "Replace key" : "API key"}
                      {p.needsUrl && " (optional)"}
                    </label>
                    <input
                      id={`key-${p.id}`}
                      className="input mono"
                      type="password"
                      autoComplete="off"
                      value={secrets[p.id] ?? ""}
                      onChange={(e) => setSecrets({ ...secrets, [p.id]: e.target.value })}
                    />
                  </div>
                </div>
                <div>
                  <span className="label invisible hidden lg:block" aria-hidden>
                    actions
                  </span>
                  <div className="flex gap-2">
                    <button
                      className="btn btn-field min-w-[6rem]"
                      type="button"
                      onClick={() => saveKey(p.id)}
                      disabled={busy === p.id || !(secrets[p.id] || (p.needsUrl && urls[p.id]))}
                    >
                      {busy === p.id ? "Checking…" : "Save"}
                    </button>
                    {existing && (
                      <button className="btn-ghost btn-field" type="button" onClick={() => removeKey(p.id, p.label)}>
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </Panel>

      {local && sessions}
    </Page>
  );
}
