"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Api, ApiError, type ServerInfo } from "@/lib/api";
import { AUTH_PROVIDER, useAuthState } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const { user, loading, refresh } = useAuthState();
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    Api.info().then(setInfo).catch(() => setInfo(null));
  }, []);
  useEffect(() => {
    if (!loading && user) router.replace("/home");
  }, [user, loading, router]);

  if (AUTH_PROVIDER === "clerk") {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { SignIn } = require("@clerk/nextjs") as typeof import("@clerk/nextjs");
    return (
      <main className="flex min-h-screen items-center justify-center p-6">
        <SignIn routing="hash" forceRedirectUrl="/home" />
      </main>
    );
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      if (mode === "signup") await Api.signup(email, password, name);
      else await Api.login(email, password);
      await refresh();
      router.replace("/home");
    } catch (ex) {
      setErr(ex instanceof ApiError ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  const dot = { width: 4, height: 4, borderRadius: 2, background: "var(--fg)" };
  return (
    <main className="flex min-h-screen flex-wrap">
      {/* left: what envforge is */}
      <section
        className="flex min-w-0 flex-[1_1_480px] flex-col justify-between gap-12 border-b p-8 md:border-r md:border-b-0 md:p-12"
        style={{ borderColor: "var(--line)" }}
      >
        <div className="flex flex-col gap-4">
          <div className="text-lg font-semibold tracking-tight">envforge</div>
          <h1 className="max-w-[440px] text-[40px] leading-[44px] font-semibold tracking-tight">Environments in, trained agents out.</h1>
          <p className="max-w-[440px]" style={{ color: "var(--muted)" }}>
            Plug in a pack, roll out a model, score every episode, and turn the clean solves into training data. The same loop for every domain.
          </p>
        </div>
        <div className="hidden max-w-[560px] grid-cols-3 gap-4 sm:grid" aria-hidden>
          <div className="h-[120px] rounded-md" style={{ background: "var(--accent)" }} />
          <div className="panel grid h-[120px] grid-cols-4 gap-[18px] rounded-md p-[18px]">
            {Array.from({ length: 12 }, (_, i) => (
              <div key={i} style={dot} />
            ))}
          </div>
          <div className="h-[120px] rounded-md" style={{ background: "var(--fg)" }} />
          <div className="flex gap-2">
            <div className="h-2 w-12 rounded" style={{ background: "var(--good)" }} />
            <div className="h-2 w-12 rounded" style={{ background: "var(--warn)" }} />
            <div className="h-2 w-8 rounded" style={{ background: "var(--bad)" }} />
          </div>
        </div>
        <div className="hint flex flex-wrap gap-x-6 gap-y-1">
          <span>3 packs · netops, sipops, repofix</span>
          <span>Anthropic · OpenAI · Vercel AI Gateway · vLLM</span>
          <span>LoRA SFT on rented GPUs</span>
        </div>
      </section>

      {/* right: the sign-in card */}
      <section className="flex min-w-0 flex-[1_1_420px] items-center justify-center p-8 md:p-12">
        <form onSubmit={submit} className="panel w-full max-w-sm space-y-4 p-5">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold">{mode === "login" ? "Sign in to your workspace" : "Create your account"}</h2>
            <div style={{ color: "var(--muted)" }}>Workspaces are private to you. Provider keys are stored encrypted and never shown again.</div>
          </div>
          {mode === "signup" && (
            <div>
              <label className="label" htmlFor="login-name">
                Name
              </label>
              <input id="login-name" className="input" autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
          )}
          <div>
            <label className="label" htmlFor="login-email">
              Email
            </label>
            <input
              id="login-email"
              className="input"
              type="email"
              required
              autoComplete="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div>
            <label className="label" htmlFor="login-password">
              Password
            </label>
            <input
              id="login-password"
              className="input"
              type="password"
              required
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              placeholder={mode === "signup" ? "10+ characters" : undefined}
              minLength={mode === "signup" ? 10 : 1}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          {err && (
            <div role="alert" className="text-sm" style={{ color: "var(--bad)" }}>
              {err}
            </div>
          )}
          <button className="btn w-full justify-center" disabled={busy}>
            {busy ? "…" : mode === "login" ? "Sign in" : "Sign up"}
          </button>
          {(info?.allow_signup ?? true) && (
            <button type="button" className="btn-ghost w-full justify-center" onClick={() => setMode(mode === "login" ? "signup" : "login")}>
              {mode === "login" ? "Need an account? Sign up" : "Have an account? Sign in"}
            </button>
          )}
          <div className="hint text-center">
            Or use <span className="mono">scripted</span> after signing in — it runs offline and needs no key.
          </div>
        </form>
      </section>
    </main>
  );
}
