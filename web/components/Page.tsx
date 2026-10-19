"use client";

// The building blocks every page under /(app) uses, so they all line up:
//
//   <Page>                                   vertical rhythm: space-y-6 between blocks
//     <PageHeader title="Runs" description="…" actions={<button…/>} />
//     <Panel title="New run" hint="…">…</Panel>          padded panel (p-5)
//     <Panel title="History" flush>…<table/></Panel>     a table edge to edge
//   </Page>
//
// Inside a panel, form grids use `items-end` so buttons sit on the inputs' baseline.

import Link from "next/link";

export function Page({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`space-y-6 ${className}`}>{children}</div>;
}

export function PageHeader({
  title,
  description,
  back,
  meta,
  actions,
  children,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  back?: { href: string; label: string }; // a "← Runs" link above the title
  meta?: React.ReactNode; // pills / toggles right after the title
  actions?: React.ReactNode; // buttons on the right
  children?: React.ReactNode; // extra lines under the description
}) {
  return (
    <header className="space-y-1.5">
      {back && (
        <Link href={back.href} className="text-sm" style={{ color: "var(--muted)" }}>
          ← {back.label}
        </Link>
      )}
      <div className="flex min-h-9 flex-wrap items-center gap-x-3 gap-y-2">
        <h1 className="min-w-0 break-words text-lg font-semibold">{title}</h1>
        {meta}
        {actions && <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {description && (
        <div className="max-w-4xl text-sm" style={{ color: "var(--muted)" }}>
          {description}
        </div>
      )}
      {children}
    </header>
  );
}

export function PanelTitle({ title, hint, aside }: { title: React.ReactNode; hint?: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
      <div className="min-w-0 flex-1">
        <h2 className="text-sm font-semibold">{title}</h2>
        {hint && (
          <div className="mt-0.5 max-w-4xl text-xs" style={{ color: "var(--muted)" }}>
            {hint}
          </div>
        )}
      </div>
      {aside}
    </div>
  );
}

/**
 * A bordered panel. `title`/`hint`/`aside` render a PanelTitle. `flush` is for tables:
 * the content runs edge to edge (cells bring their own padding) and the title still
 * lines up with the first column. `tone="danger"` gives a red border (Danger zone).
 */
export function Panel({
  title,
  hint,
  aside,
  flush = false,
  tone = "default",
  className = "",
  children,
}: {
  title?: React.ReactNode;
  hint?: React.ReactNode;
  aside?: React.ReactNode;
  flush?: boolean;
  tone?: "default" | "danger";
  className?: string;
  children: React.ReactNode;
}) {
  const border = tone === "danger" ? { borderColor: "color-mix(in srgb, var(--bad) 45%, var(--line))" } : undefined;
  const head = title !== undefined && <PanelTitle title={title} hint={hint} aside={aside} />;
  if (flush)
    return (
      <section className={`panel min-w-0 ${className}`} style={border}>
        {head && <div className="px-5 pt-5">{head}</div>}
        <div className={`overflow-x-auto px-3 pb-3 ${head ? "pt-2" : "pt-3"}`}>{children}</div>
      </section>
    );
  return (
    <section className={`panel min-w-0 space-y-4 p-5 ${className}`} style={border}>
      {head}
      {children}
    </section>
  );
}

/** Muted one-liner for "Loading…" / "Nothing yet" inside a flush (table) panel. */
export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 py-3 text-sm" style={{ color: "var(--muted)" }}>
      {children}
    </div>
  );
}
