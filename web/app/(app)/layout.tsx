"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Avatar } from "@/components/Avatar";
import { ConfirmProvider } from "@/components/ConfirmDialog";
import { ToastProvider } from "@/components/Toast";
import { WorkspaceSwitcher } from "@/components/WorkspaceSwitcher";
import { useAuthState } from "@/lib/auth";
import { WorkspaceProvider } from "@/lib/workspace";

// grouped by where each page sits in the pipeline: build an environment, evaluate models on it, train
const NAV: { group?: string; items: { href: string; label: string; also?: string[] }[] }[] = [
  { items: [{ href: "/home", label: "Home" }] },
  { group: "Build", items: [{ href: "/environments", label: "Environments" }, { href: "/builder", label: "Builder" }] },
  { group: "Evaluate", items: [{ href: "/runs", label: "Runs", also: ["/compare"] }] },
  {
    group: "Train",
    items: [
      { href: "/train", label: "Guide" },
      { href: "/datasets", label: "Datasets" },
      { href: "/training", label: "Fine-tune" },
      { href: "/models", label: "Models" },
    ],
  },
];

function NavLink({ href, active, children, onNavigate }: { href: string; active: boolean; children: React.ReactNode; onNavigate?: () => void }) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      onClick={onNavigate}
      className={`flex items-center gap-2 rounded-md px-2 py-1.5 ${active ? "font-medium" : "hover:text-[var(--fg)]"}`}
      style={{ color: active ? "var(--fg)" : "var(--muted)", background: active ? "var(--panel)" : undefined }}
    >
      {children}
    </Link>
  );
}

/** The left navigation: brand, workspace switcher, pages, then settings and the account at the bottom. */
function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const path = usePathname();
  const router = useRouter();
  const { user, logout } = useAuthState();
  const on = (href: string, also: string[] = []) => [href, ...also].some((h) => path === h || path.startsWith(h + "/"));
  const name = user ? user.display_name || user.email.split("@")[0] : "";

  return (
    <div className="flex h-full flex-col gap-5 p-4">
      <Link href="/home" onClick={onNavigate} className="px-2 text-lg font-semibold tracking-tight" style={{ color: "var(--fg)" }}>
        envforge
      </Link>
      <WorkspaceSwitcher />
      <nav aria-label="Main" className="flex flex-col gap-3">
        {NAV.map((g, i) => (
          // a thin rule above each section after Home: Build | Evaluate | Train
          <div key={g.group ?? i} className={`flex flex-col gap-0.5 ${i > 0 ? "border-t pt-3" : ""}`} style={i > 0 ? { borderColor: "var(--line)" } : undefined}>
            {g.group && (
              <div className="px-2 pb-1 text-[11px] font-medium tracking-wide uppercase" style={{ color: "var(--muted)" }}>
                {g.group}
              </div>
            )}
            {g.items.map((n) => (
              <NavLink key={n.href} href={n.href} active={on(n.href, n.also)} onNavigate={onNavigate}>
                {n.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div className="mt-auto flex flex-col gap-0.5 border-t pt-4" style={{ borderColor: "var(--line)" }}>
        <NavLink href="/usage" active={on("/usage")} onNavigate={onNavigate}>
          Usage
        </NavLink>
        <NavLink href="/settings" active={on("/settings")} onNavigate={onNavigate}>
          Workspace settings
        </NavLink>
        {user && (
          <div className="flex items-center gap-1">
            <NavLink href="/account" active={on("/account")} onNavigate={onNavigate}>
              <Avatar name={user.display_name} email={user.email} size={20} />
              <span className="min-w-0 truncate">{name}</span>
            </NavLink>
            <button
              type="button"
              aria-label="Sign out"
              title="Sign out"
              className="ml-auto rounded-md p-1.5 hover:bg-[var(--panel)] hover:text-[var(--fg)]"
              style={{ color: "var(--muted)" }}
              onClick={async () => {
                await logout();
                router.replace("/login");
              }}
            >
              <svg aria-hidden width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <path d="M16 17l5-5-5-5" />
                <path d="M21 12H9" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [drawer, setDrawer] = useState(false);
  useEffect(() => setDrawer(false), [path]);
  useEffect(() => {
    if (!drawer) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawer(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawer]);

  return (
    <div className="min-h-screen md:flex">
      {/* wide screens: a sticky sidebar */}
      <aside className="sticky top-0 hidden h-screen w-[var(--sidebar-w)] shrink-0 overflow-x-hidden overflow-y-auto border-r md:block" style={{ borderColor: "var(--line)" }}>
        <Sidebar />
      </aside>

      {/* phones: a top bar with a menu button that opens the same sidebar as a drawer */}
      <div className="sticky top-0 z-20 flex items-center gap-3 border-b px-4 py-3 md:hidden" style={{ borderColor: "var(--line)", background: "var(--bg)" }}>
        <button type="button" className="btn-ghost px-2" aria-label="Open navigation" aria-expanded={drawer} onClick={() => setDrawer(true)}>
          <svg aria-hidden width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <Link href="/home" className="text-lg font-semibold tracking-tight">
          envforge
        </Link>
      </div>
      {drawer && (
        <div className="fixed inset-0 z-40 md:hidden" style={{ background: "var(--scrim)" }} onMouseDown={() => setDrawer(false)}>
          <aside
            aria-label="Navigation"
            className="h-full w-[min(var(--sidebar-w),85vw)] overflow-x-hidden overflow-y-auto border-r"
            style={{ borderColor: "var(--line)", background: "var(--bg)" }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <Sidebar onNavigate={() => setDrawer(false)} />
          </aside>
        </div>
      )}

      <main className="min-w-0 flex-1 p-4 sm:p-6">{children}</main>
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
      <ToastProvider>
        <ConfirmProvider>
          <Shell>{children}</Shell>
        </ConfirmProvider>
      </ToastProvider>
    </WorkspaceProvider>
  );
}
