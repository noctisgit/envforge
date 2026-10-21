"use client";

// Toasts: short, transient feedback for something the user just did ("Saved",
// "Key verified", "Couldn't queue the run"). One stack, bottom-right, for the whole app:
//
//   const toast = useToast();
//   toast.success("Defaults saved");
//   toast.error("Couldn't queue the run", err);          // err: ApiError | Error | string
//   toast({ tone: "info", title: "Run queued", message: "…", action: { label: "Open", onClick } });
//
// info/success fade after 5 s; warning/error stay until dismissed (they usually need reading).
// Errors that belong to one form field still show next to that field, not here.

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import { ApiError } from "@/lib/api";

export type ToastTone = "info" | "success" | "warning" | "error";

export interface ToastOptions {
  tone?: ToastTone;
  title: string;
  message?: React.ReactNode;
  action?: { label: string; onClick: () => void };
  durationMs?: number; // 0 = stay until dismissed
}

interface Item extends ToastOptions {
  id: number;
}

type ToastFn = ((o: ToastOptions) => void) & {
  info: (title: string, message?: React.ReactNode) => void;
  success: (title: string, message?: React.ReactNode) => void;
  warning: (title: string, message?: React.ReactNode) => void;
  error: (title: string, err?: unknown) => void;
};

/** The readable part of whatever was thrown. */
export function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return err === undefined || err === null ? "" : String(err);
}

const noop = Object.assign(() => {}, { info: () => {}, success: () => {}, warning: () => {}, error: () => {} }) as ToastFn;
const Ctx = createContext<ToastFn>(noop);

export function useToast() {
  return useContext(Ctx);
}

const GLYPH: Record<ToastTone, string> = { info: "i", success: "✓", warning: "!", error: "×" };
const COLOR: Record<ToastTone, string> = { info: "var(--accent)", success: "var(--good)", warning: "var(--warn)", error: "var(--bad)" };

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<Item[]>([]);
  const next = useRef(1);

  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);

  const push = useCallback(
    (o: ToastOptions) => {
      const id = next.current++;
      const tone = o.tone ?? "info";
      const ms = o.durationMs ?? (tone === "info" || tone === "success" ? 5000 : 0);
      setItems((xs) => [...xs.slice(-3), { ...o, tone, id }]); // at most 4 on screen
      if (ms > 0) setTimeout(() => dismiss(id), ms);
    },
    [dismiss],
  );

  const toast = useMemo(
    () =>
      Object.assign(push, {
        info: (title: string, message?: React.ReactNode) => push({ tone: "info", title, message }),
        success: (title: string, message?: React.ReactNode) => push({ tone: "success", title, message }),
        warning: (title: string, message?: React.ReactNode) => push({ tone: "warning", title, message }),
        error: (title: string, err?: unknown) => push({ tone: "error", title, message: errorText(err) || undefined }),
      }) as ToastFn,
    [push],
  );

  return (
    <Ctx.Provider value={toast}>
      {children}
      <div className="pointer-events-none fixed right-4 bottom-4 z-[60] flex w-[360px] max-w-[calc(100vw-2rem)] flex-col gap-2">
        {items.map((t) => (
          <ToastView key={t.id} item={t} onDismiss={() => dismiss(t.id)} />
        ))}
      </div>
    </Ctx.Provider>
  );
}

function ToastView({ item, onDismiss }: { item: Item; onDismiss: () => void }) {
  const tone = item.tone ?? "info";
  const loud = tone === "error" || tone === "warning";
  return (
    <div
      role={loud ? "alert" : "status"}
      aria-live={loud ? "assertive" : "polite"}
      className="panel pointer-events-auto flex items-start gap-3 rounded-md px-4 py-3 text-sm"
      style={{ boxShadow: "var(--shadow-dialog)", borderRadius: 6 }}
    >
      <span
        aria-hidden
        className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-bold"
        style={{ background: COLOR[tone], color: tone === "error" ? "var(--on-bad)" : "var(--on-accent)" }}
      >
        {GLYPH[tone]}
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="font-semibold" style={{ color: COLOR[tone] }}>
          {item.title}
        </div>
        {item.message && <div className="[overflow-wrap:anywhere]">{item.message}</div>}
        {item.action && (
          <div className="mt-1">
            <button
              className="btn-ghost btn-sm"
              onClick={() => {
                item.action!.onClick();
                onDismiss();
              }}
            >
              {item.action.label}
            </button>
          </div>
        )}
      </div>
      <button type="button" aria-label="Dismiss" className="shrink-0 rounded px-1 text-base leading-5 hover:text-[var(--fg)]" style={{ color: "var(--muted)" }} onClick={onDismiss}>
        ×
      </button>
    </div>
  );
}
