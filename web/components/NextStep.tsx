"use client";

// "What next?" — a small callout at the end of a step (a finished run, a comparison)
// that points at the step that usually follows, so the pages read as one pipeline.

export function NextStep({ title, children, actions }: { title: React.ReactNode; children?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <section
      aria-label="Next step"
      className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-[var(--radius-panel)] border px-5 py-4"
      style={{ borderColor: "color-mix(in srgb, var(--accent) 40%, var(--line))", background: "var(--accent-tint)" }}
    >
      <div className="min-w-0 flex-1 basis-80">
        <div className="label mb-0.5" style={{ color: "var(--accent)" }}>
          next step
        </div>
        <div className="font-medium">{title}</div>
        {children && (
          <div className="mt-0.5" style={{ color: "var(--muted)" }}>
            {children}
          </div>
        )}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </section>
  );
}
