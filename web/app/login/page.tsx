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
    if (!loading && user) router.replace("/runs");
  }, [user, loading, router]);

  if (AUTH_PROVIDER === "clerk") {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { SignIn } = require("@clerk/nextjs") as typeof import("@clerk/nextjs");
    return (
      <main className="flex min-h-screen items-center justify-center p-6">
        <SignIn routing="hash" forceRedirectUrl="/runs" />
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
      router.replace("/runs");
    } catch (ex) {
      setErr(ex instanceof ApiError ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <form onSubmit={submit} className="panel w-full max-w-sm space-y-4 p-6">
        <div>
          <div className="text-lg font-semibold">envforge</div>
          <div className="text-sm" style={{ color: "var(--muted)" }}>
            {mode === "login" ? "Sign in to your workspace" : "Create your account"}
          </div>
        </div>
        {mode === "signup" && (
          <div>
            <label className="label">Name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
        )}
        <div>
          <label className="label">Email</label>
          <input className="input" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div>
          <label className="label">Password</label>
          <input className="input" type="password" required minLength={mode === "signup" ? 10 : 1} value={password} onChange={(e) => setPassword(e.target.value)} />
        </div>
        {err && <div className="text-sm" style={{ color: "var(--bad)" }}>{err}</div>}
        <button className="btn w-full justify-center" disabled={busy}>
          {busy ? "…" : mode === "login" ? "Sign in" : "Sign up"}
        </button>
        {(info?.allow_signup ?? true) && (
          <button type="button" className="w-full text-center text-sm underline" style={{ color: "var(--muted)" }} onClick={() => setMode(mode === "login" ? "signup" : "login")}>
            {mode === "login" ? "Need an account? Sign up" : "Have an account? Sign in"}
          </button>
        )}
      </form>
    </main>
  );
}
