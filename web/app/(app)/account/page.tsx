"use client";

// Your account: profile, security (password, sessions) and provider keys, one tab each.
// Provider keys live on the account, so every workspace you run in uses the same keys.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Avatar } from "@/components/Avatar";
import { useConfirm } from "@/components/ConfirmDialog";
import { invalidateModelCache } from "@/components/ModelPicker";
import { Page, PageHeader, Panel } from "@/components/Page";
import { Tabs, tabPanelProps, useHashTab } from "@/components/Tabs";
import { useToast } from "@/components/Toast";
import { Api, type KeyOut, type Provider } from "@/lib/api";
import { useAuthState } from "@/lib/auth";
import { useWorkspace } from "@/lib/workspace";

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

const TABS = ["profile", "security", "keys"] as const;

export default function AccountPage() {
  const { user, refresh, logout } = useAuthState();
  const { workspaces } = useWorkspace();
  const router = useRouter();
  const confirm = useConfirm();
  const toast = useToast();
  const [tab, setTab] = useHashTab(TABS, "profile");
  const [name, setName] = useState(user?.display_name ?? "");
  const [pw, setPw] = useState({ current: "", next: "", again: "" });
  const [pwErr, setPwErr] = useState<string | null>(null);
  const [keys, setKeys] = useState<KeyOut[]>([]);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    Api.keys()
      .then((ks) => {
        setKeys(ks);
        setUrls(Object.fromEntries(ks.filter((k) => k.base_url).map((k) => [k.provider, k.base_url!])));
      })
      .catch((e) => toast.error("Couldn't load your provider keys", e));
  }, [toast]);

  if (!user) return null;
  const local = user.auth_provider === "local";

  async function saveProfile(e: React.FormEvent) {
    e.preventDefault();
    try {
      await Api.updateProfile(name);
      await refresh();
      toast.success("Profile saved", "Your display name is updated everywhere it appears.");
    } catch (ex) {
      toast.error("Couldn't save your profile", ex);
    }
  }

  async function savePassword(e: React.FormEvent) {
    e.preventDefault();
    if (pw.next !== pw.again) return setPwErr("The new passwords don't match.");
    setPwErr(null);
    try {
      await Api.changePassword(pw.current, pw.next);
      setPw({ current: "", next: "", again: "" });
      toast.success("Password changed", "Your other devices have been signed out.");
    } catch (ex) {
      toast.error("Couldn't change your password", ex);
    }
  }

  async function signOutEverywhere() {
    const ok = await confirm({
      title: "Sign out everywhere?",
      body: "Every browser and device signed in to your account will need to sign in again, including this one. Provider keys and workspaces are unaffected.",
      confirmLabel: "Sign out everywhere",
      tone: "danger",
    });
    if (!ok) return;
    try {
      await Api.logoutAll();
      window.location.href = "/login";
    } catch (ex) {
      toast.error("Couldn't sign out everywhere", ex);
    }
  }

  async function saveKey(p: Provider, label: string) {
    setBusy(p);
    try {
      const ks = await Api.putKey(p, secrets[p] ?? "", urls[p]);
      setKeys(ks);
      setSecrets({ ...secrets, [p]: "" });
      invalidateModelCache();
      const k = ks.find((x) => x.provider === p);
      toast.success(
        `${label} key saved`,
        k?.verified_at ? `Verified with the provider${k.n_models !== null ? ` · ${k.n_models} models` : ""}.` : (k?.verify_note ?? "It will be checked on the first run."),
      );
    } catch (ex) {
      toast.error(`${label} key not saved`, ex);
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
    try {
      setKeys(await Api.deleteKey(p));
      invalidateModelCache();
      toast.warning(`${label} key removed`, `Runs that use ${label} models will fail until you add a key again.`);
    } catch (ex) {
      toast.error(`Couldn't remove the ${label} key`, ex);
    }
  }

  const facts: [string, React.ReactNode][] = [
    ["role", user.is_admin ? "admin" : "member"],
    ["auth", local ? "local · email & password" : "Clerk"],
    [
      "workspaces",
      <>
        {workspaces.length} ·{" "}
        <Link href="/settings" className="underline">
          manage
        </Link>
      </>,
    ],
  ];
  if (user.is_admin) facts.push(["admin tools", <span className="mono">envforge admin users | create-user | reset-password | sign-out | set-admin</span>]);

  return (
    <Page>
      <PageHeader
        leading={<Avatar name={user.display_name} email={user.email} size={56} />}
        title={user.display_name || user.email}
        description={
          <>
            {user.email} · {user.is_admin ? "admin · " : ""}signed in with {local ? "email & password" : "Clerk"}
          </>
        }
        actions={
          <button
            className="btn-ghost"
            onClick={async () => {
              await logout();
              router.replace("/login");
            }}
          >
            Sign out
          </button>
        }
      />

      <Tabs
        idPrefix="acct"
        label="Account sections"
        tabs={[
          { id: "profile", label: "Profile" },
          { id: "security", label: "Security" },
          { id: "keys", label: "Provider keys", count: keys.length },
        ]}
        active={tab}
        onChange={setTab}
      />

      <div {...tabPanelProps("acct", "profile")} hidden={tab !== "profile"} className="grid grid-cols-1 items-start gap-6 lg:grid-cols-2">
        <Panel title="Profile" hint="how you appear in the sidebar and in Usage → By member">
          <form onSubmit={saveProfile} className="space-y-4">
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
              <div className="hint mt-1">Email is your sign-in and can&apos;t be changed here.</div>
            </div>
            <div className="flex justify-end border-t pt-4" style={{ borderColor: "var(--line)" }}>
              <button className="btn">Save</button>
            </div>
          </form>
        </Panel>
        <Panel title="Account">
          <table className="data inset">
            <tbody>
              {facts.map(([k, v]) => (
                <tr key={k}>
                  <td style={{ color: "var(--muted)" }}>{k}</td>
                  <td className="[overflow-wrap:anywhere]">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>

      <div {...tabPanelProps("acct", "security")} hidden={tab !== "security"} className="grid grid-cols-1 items-start gap-6 lg:grid-cols-2">
        {local && (
          <Panel title="Change password" hint="Changing it signs out every other device.">
            <form onSubmit={savePassword} className="space-y-3">
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
                  aria-invalid={!!pwErr}
                  aria-describedby={pwErr ? "pw-err" : undefined}
                  onChange={(e) => {
                    setPw({ ...pw, again: e.target.value });
                    setPwErr(null);
                  }}
                />
                {pwErr && (
                  <div id="pw-err" className="mt-1 text-sm" style={{ color: "var(--bad)" }}>
                    {pwErr}
                  </div>
                )}
              </div>
              <div className="flex justify-end border-t pt-3" style={{ borderColor: "var(--line)" }}>
                <button className="btn" disabled={!pw.current || pw.next.length < 10}>
                  Change password
                </button>
              </div>
            </form>
          </Panel>
        )}
        <Panel title="Sessions">
          <div className="space-y-3">
            <div style={{ color: "var(--muted)" }}>Signing out always ends every session: on every device and browser, including this one.</div>
            <button className="btn-danger" onClick={signOutEverywhere}>
              Sign out everywhere
            </button>
          </div>
        </Panel>
      </div>

      <div {...tabPanelProps("acct", "keys")} hidden={tab !== "keys"}>
        <Panel
          flush
          title="Provider keys"
          footer={
            <span>
              Try the <span className="mono">scripted</span> model to run without any key. The vLLM base URL is checked against private-address rules before it is
              saved.
            </span>
          }
        >
          <div className="border-b px-5 py-3" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>
            Encrypted at rest, decrypted only inside the worker that executes your runs, never shown again after saving. Anthropic and OpenAI keys are checked with
            the provider when you save them, and the model picker then lists exactly the models your key can use. Keys are yours across every workspace.
          </div>
          {PROVIDERS.map((p) => {
            const existing = keys.find((k) => k.provider === p.id);
            return (
              <div
                key={p.id}
                className="grid grid-cols-1 items-start gap-4 border-b px-5 py-4 last:border-b-0 md:grid-cols-[220px_minmax(0,1fr)]"
                style={{ borderColor: "var(--line)" }}
              >
                <div className="min-w-0 space-y-1">
                  <div className="font-medium">{p.label}</div>
                  <div className="hint">{p.hint}</div>
                </div>
                <div className="min-w-0 space-y-2">
                  {existing && (
                    <div className="flex flex-wrap items-center gap-2">
                      {existing.verified_at ? (
                        <span className="pill pill-completed">✓ verified</span>
                      ) : (
                        <span className="pill pill-queued">saved</span>
                      )}
                      <span className="mono">{existing.masked}</span>
                      <span className="hint">
                        saved {new Date(existing.updated_at).toLocaleDateString()}
                        {existing.n_models !== null && ` · ${existing.n_models} models`}
                        {!existing.verified_at && !existing.verify_note && (p.id === "vercel" || p.id === "vllm") && " · checked on first run"}
                      </span>
                    </div>
                  )}
                  {existing?.verify_note && <div className="text-xs" style={{ color: "var(--warn)" }}>{existing.verify_note}</div>}
                  {p.needsUrl && (
                    <input
                      id={`url-${p.id}`}
                      className="input mono"
                      aria-label={`${p.label} base URL`}
                      placeholder="Base URL — http://your-box:8000/v1"
                      value={urls[p.id] ?? ""}
                      onChange={(e) => setUrls({ ...urls, [p.id]: e.target.value })}
                    />
                  )}
                  <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap">
                    <input
                      id={`key-${p.id}`}
                      className="input mono"
                      type="password"
                      autoComplete="off"
                      aria-label={`${p.label} API key`}
                      placeholder={existing ? "Replace key" : p.needsUrl ? "API key (optional)" : "API key"}
                      value={secrets[p.id] ?? ""}
                      onChange={(e) => setSecrets({ ...secrets, [p.id]: e.target.value })}
                    />
                    <button
                      className="btn-ghost btn-field min-w-[5.5rem]"
                      type="button"
                      onClick={() => saveKey(p.id, p.label)}
                      disabled={busy === p.id || !(secrets[p.id] || (p.needsUrl && urls[p.id]))}
                    >
                      {busy === p.id ? "Checking…" : "Save"}
                    </button>
                    {existing && (
                      <button className="btn-danger btn-field" type="button" onClick={() => removeKey(p.id, p.label)}>
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </Panel>
      </div>
    </Page>
  );
}
