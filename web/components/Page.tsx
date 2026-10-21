"use client";

// The building blocks every page under /(app) uses, so they all line up:
//
//   <Page>                                   vertical rhythm: 24px between blocks
//     <PageHeader title="Runs" description="…" actions={<button…/>} />
//     <StatGrid><StatCard label="Active" value="1" note="of 3 slots" /></StatGrid>
//     <Panel title="New run" hint="…">…</Panel>          padded panel (20px)
//     <Panel title="History" flush>…<table/></Panel>     header bar + a table edge to edge
//   </Page>
//
// Inside a panel, form grids use `items-end` so buttons sit on the inputs' baseline.
// Feedback for an action goes to a toast (useToast), confirmations to useConfirm.

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
  leading,
  children,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  back?: { href: string; label: string }; // a "← Runs" link above the title
  meta?: React.ReactNode; // pills / a muted summary right after the title
  actions?: React.ReactNode; // buttons on the right
  leading?: React.ReactNode; // an avatar or mark before the title
  children?: React.ReactNode; // extra content under the description (e.g. a facts grid)
}) {
  return (
    <header className="space-y-3">
      {back && (
        <Link href={back.href} className="inline-block text-xs hover:text-[var(--fg)]" style={{ color: "var(--muted)" }}>
          ← {back.label}
        </Link>
      )}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {leading}
        <div className="min-w-0 flex-1 basis-80 space-y-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <h1 className="min-w-0 text-2xl font-semibold break-words">{title}</h1>
            {meta}
          </div>
          {description && (
            <div className="max-w-4xl" style={{ color: "var(--muted)" }}>
              {description}
            </div>
          )}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </header>
  );
}

/** A section heading inside a padded panel: 18px title, optional muted hint, optional controls on the right. */
export function PanelTitle({ title, hint, aside }: { title: React.ReactNode; hint?: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
      <div className="min-w-0 flex-1">
        <h2 className="text-lg font-semibold">{title}</h2>
        {hint && <div className="hint mt-0.5 max-w-4xl">{hint}</div>}
      </div>
      {aside}
    </div>
  );
}

/**
 * A bordered panel. `title`/`hint`/`aside` render its heading. `flush` is for tables
 * and lists: the heading becomes a bar with a bottom border and the content runs edge
 * to edge (table cells bring their own padding). `footer` adds a muted bottom line to
 * a flush panel. `tone="danger"` gives a red border (Danger zone).
 */
export function Panel({
  title,
  hint,
  aside,
  footer,
  flush = false,
  tone = "default",
  className = "",
  children,
}: {
  title?: React.ReactNode;
  hint?: React.ReactNode;
  aside?: React.ReactNode;
  footer?: React.ReactNode;
  flush?: boolean;
  tone?: "default" | "danger";
  className?: string;
  children: React.ReactNode;
}) {
  const border = tone === "danger" ? { borderColor: "color-mix(in srgb, var(--bad) 45%, var(--line))" } : undefined;
  if (flush)
    return (
      <section className={`panel flex min-w-0 flex-col ${className}`} style={border}>
        {title !== undefined && (
          <div className="flex min-h-14 flex-wrap items-center gap-x-3 gap-y-1 border-b px-4 py-3" style={{ borderColor: "var(--line)" }}>
            <h2 className="text-lg font-semibold">{title}</h2>
            {hint && <span className={`hint ${aside ? "" : "ml-auto"}`}>{hint}</span>}
            {aside && <div className="ml-auto flex flex-wrap items-center gap-2">{aside}</div>}
          </div>
        )}
        <div className="flush-body min-w-0 overflow-x-auto">{children}</div>
        {footer && (
          <div className="hint flex flex-wrap justify-between gap-2 border-t px-4 py-3" style={{ borderColor: "var(--line)" }}>
            {footer}
          </div>
        )}
      </section>
    );
  return (
    <section className={`panel min-w-0 space-y-4 p-5 ${className}`} style={border}>
      {title !== undefined && <PanelTitle title={title} hint={hint} aside={aside} />}
      {children}
    </section>
  );
}

/** Muted one-liner for "Loading…" / "Nothing yet" inside a flush (table) panel. */
export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-4 py-4 text-sm" style={{ color: "var(--muted)" }}>
      {children}
    </div>
  );
}

const TONE = { good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)", accent: "var(--accent)" } as const;
export type Tone = keyof typeof TONE;

/** The CSS colour for a tone (undefined stays the default text colour). */
export function toneColor(tone: Tone | undefined): string | undefined {
  return tone ? TONE[tone] : undefined;
}

/** Colour for a 0–1 rate: ≥ 80% good, 40–80% warn, below bad (the same thresholds as MetricsTable). */
export function rateTone(x: number | null | undefined): Tone | undefined {
  if (x === null || x === undefined || Number.isNaN(x)) return undefined;
  return x >= 0.8 ? "good" : x >= 0.4 ? "warn" : "bad";
}

/** One headline number: an uppercase label, a 24px value, and a muted note. */
export function StatCard({ label, value, unit, note, tone, mono }: { label: string; value: React.ReactNode; unit?: React.ReactNode; note?: React.ReactNode; tone?: Tone; mono?: boolean }) {
  return (
    <div className="panel flex min-w-0 flex-col gap-1 p-4">
      <div className="label mb-0">{label}</div>
      <div className={`num truncate text-2xl font-semibold ${mono ? "mono" : ""}`} style={{ color: tone ? TONE[tone] : undefined }}>
        {value}
        {unit && <span className="ml-1.5 text-sm font-normal" style={{ color: "var(--muted)", fontFamily: "inherit" }}>{unit}</span>}
      </div>
      {note && <div className="hint truncate">{note}</div>}
    </div>
  );
}

/** A responsive row of StatCards: 2 across on phones, `cols` across on wide screens. */
export function StatGrid({ cols = 4, children }: { cols?: 3 | 4 | 5; children: React.ReactNode }) {
  const wide = cols === 5 ? "xl:grid-cols-5" : cols === 3 ? "lg:grid-cols-3" : "lg:grid-cols-4";
  return <div className={`grid grid-cols-2 gap-4 ${wide}`}>{children}</div>;
}

/** Label/value pairs in a bordered grid — the "facts" block under a detail page's title. */
export function Facts({ items }: { items: { label: string; value: React.ReactNode }[] }) {
  return (
    <dl className="panel grid grid-cols-2 gap-x-4 gap-y-3 p-4 md:grid-cols-3 xl:grid-cols-6">
      {items.map((f) => (
        <div key={f.label} className="min-w-0">
          <dt className="label mb-0.5">{f.label}</dt>
          <dd className="[overflow-wrap:anywhere]">{f.value}</dd>
        </div>
      ))}
    </dl>
  );
}
