"use client";

// The account menu in the header: an initials avatar that opens a card with the
// user's name and email, links to their profile and workspace settings, and sign out.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useAuthState } from "@/lib/auth";

function initials(name: string, email: string): string {
  const src = name.trim() || email.split("@")[0];
  const parts = src.split(/[\s._-]+/).filter(Boolean);
  return ((parts[0]?.[0] ?? "?") + (parts[1]?.[0] ?? "")).toUpperCase();
}

export function Avatar({ name, email, size = 32 }: { name: string; email: string; size?: number }) {
  return (
    <span
      aria-hidden
      className="inline-flex shrink-0 items-center justify-center rounded-full font-semibold"
      style={{ width: size, height: size, fontSize: size * 0.38, background: "var(--accent)", color: "#06201c" }}
    >
      {initials(name, email)}
    </span>
  );
}

export function AvatarMenu() {
  const { user, logout } = useAuthState();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!user) return null;
  const name = user.display_name || user.email.split("@")[0];

  const item = "flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm hover:bg-[var(--line)]";
  return (
    <div ref={root} className="relative">
      <button onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open} aria-label="Account menu" className="rounded-full">
        <Avatar name={user.display_name} email={user.email} />
      </button>
      {open && (
        <div role="menu" className="panel absolute right-0 z-30 mt-2 w-72 overflow-hidden shadow-2xl">
          <div className="flex items-center gap-3 border-b px-4 py-3" style={{ borderColor: "var(--line)" }}>
            <Avatar name={user.display_name} email={user.email} size={40} />
            <div className="min-w-0">
              <div className="truncate font-medium">{name}</div>
              <div className="truncate text-xs" style={{ color: "var(--muted)" }}>
                {user.email}
              </div>
            </div>
          </div>
          <Link role="menuitem" href="/account" className={item} onClick={() => setOpen(false)}>
            <span aria-hidden>👤</span> Profile &amp; provider keys
          </Link>
          <Link role="menuitem" href="/settings" className={item} onClick={() => setOpen(false)}>
            <span aria-hidden>⚙</span> Workspace settings
          </Link>
          <div className="border-t" style={{ borderColor: "var(--line)" }} />
          <button
            role="menuitem"
            className={item}
            onClick={async () => {
              setOpen(false);
              await logout();
              router.replace("/login");
            }}
          >
            <span aria-hidden>⎋</span> Sign out
          </button>
        </div>
      )}
    </div>
  );
}
