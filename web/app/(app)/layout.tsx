"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { AvatarMenu } from "@/components/AvatarMenu";
import { ConfirmProvider } from "@/components/ConfirmDialog";
import { WorkspaceSwitcher } from "@/components/WorkspaceSwitcher";
import { useAuthState } from "@/lib/auth";
import { WorkspaceProvider } from "@/lib/workspace";

const NAV = [
  { href: "/runs", label: "Runs" },
  { href: "/environments", label: "Environments" },
  { href: "/builder", label: "Builder" },
  { href: "/usage", label: "Usage" },
];

function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();

  return (
    <div className="min-h-screen">
      <header className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b px-4 py-3 sm:px-6 lg:px-10" style={{ borderColor: "var(--line)" }}>
        <Link href="/runs" className="font-semibold tracking-tight">
          envforge
        </Link>
        <nav className="flex gap-4 text-sm">
          {NAV.map((n) => {
            const on = path.startsWith(n.href) || (n.href === "/runs" && path.startsWith("/compare"));
            return (
              <Link key={n.href} href={n.href} className={on ? "font-medium" : ""} style={{ color: on ? "var(--fg)" : "var(--muted)" }}>
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-sm">
          <WorkspaceSwitcher />
          <AvatarMenu />
        </div>
      </header>
      <main className="w-full px-4 py-6 sm:px-6 lg:px-10">{children}</main>
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
      <ConfirmProvider>
        <Shell>{children}</Shell>
      </ConfirmProvider>
    </WorkspaceProvider>
  );
}
