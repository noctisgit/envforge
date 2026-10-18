"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { useAuthState } from "@/lib/auth";
import { WorkspaceProvider, useWorkspace } from "@/lib/workspace";

function Shell({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuthState();
  const { workspaces, current, select } = useWorkspace();
  const path = usePathname();
  const router = useRouter();

  const nav = [
    { href: "/runs", label: "Runs" },
    { href: "/settings", label: "Settings" },
  ];

  return (
    <div className="min-h-screen">
      <header className="flex items-center gap-6 border-b px-6 py-3" style={{ borderColor: "var(--line)" }}>
        <Link href="/runs" className="font-semibold tracking-tight">
          envforge
        </Link>
        <nav className="flex gap-4 text-sm">
          {nav.map((n) => (
            <Link key={n.href} href={n.href} className={path.startsWith(n.href) ? "font-medium" : ""} style={{ color: path.startsWith(n.href) ? "var(--fg)" : "var(--muted)" }}>
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-sm">
          {workspaces.length > 0 && (
            <select className="input !w-auto py-1" value={current?.id ?? ""} onChange={(e) => select(e.target.value)}>
              {workspaces.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name}
                </option>
              ))}
            </select>
          )}
          <span style={{ color: "var(--muted)" }}>{user?.email}</span>
          <button
            className="btn-ghost"
            onClick={async () => {
              await logout();
              router.replace("/login");
            }}
          >
            Sign out
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-6xl p-6">{children}</main>
    </div>
  );
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuthState();
  const router = useRouter();
  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [user, loading, router]);
  if (loading || !user) return null;
  return (
    <WorkspaceProvider>
      <Shell>{children}</Shell>
    </WorkspaceProvider>
  );
}
