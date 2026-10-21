"use client";

// The two workspace-settings pages as a tab bar: General (defaults, workspaces) and
// GPU boxes (Setup for serving and training). Links, so each page has its own URL.

import Link from "next/link";

const TABS = [
  { id: "general", label: "General", href: "/settings" },
  { id: "gpu", label: "GPU boxes", href: "/settings/gpu-boxes" },
] as const;

export function SettingsTabs({ active }: { active: (typeof TABS)[number]["id"] }) {
  return (
    <nav aria-label="Settings sections" className="flex gap-1 overflow-x-auto border-b" style={{ borderColor: "var(--line)" }}>
      {TABS.map((t) => {
        const on = t.id === active;
        return (
          <Link
            key={t.id}
            href={t.href}
            aria-current={on ? "page" : undefined}
            className="-mb-px border-b-2 px-3 py-2 text-sm whitespace-nowrap"
            style={{ borderColor: on ? "var(--accent)" : "transparent", color: on ? "var(--fg)" : "var(--muted)" }}
          >
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}
