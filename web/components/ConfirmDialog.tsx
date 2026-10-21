"use client";

// One confirmation dialog for the whole app. Anything destructive or that needs an
// explicit "yes" goes through it, so they all look and behave the same:
//
//   const confirm = useConfirm();
//   if (await confirm({ title: "Delete 3 runs?", body: "…", confirmLabel: "Delete", tone: "danger" })) { … }
//
// Optional `requireText` makes the user type a word (e.g. a workspace name) first.
// Esc / clicking outside cancels; Enter confirms when allowed. For danger actions the
// Cancel button gets focus, so a stray Enter never deletes anything.

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

export interface ConfirmOptions {
  title: string;
  body?: React.ReactNode;
  confirmLabel?: string; // default "Confirm"
  cancelLabel?: string; // default "Cancel"
  tone?: "default" | "danger";
  requireText?: string; // the user must type this exactly to enable the confirm button
}

type Pending = ConfirmOptions & { resolve: (ok: boolean) => void };

const Ctx = createContext<(o: ConfirmOptions) => Promise<boolean>>(async () => false);

export function useConfirm() {
  return useContext(Ctx);
}

export function ConfirmProvider({ children }: { children: React.ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null);

  const confirm = useCallback(
    (o: ConfirmOptions) =>
      new Promise<boolean>((resolve) => {
        setPending({ ...o, resolve });
      }),
    [],
  );

  const close = (ok: boolean) => {
    pending?.resolve(ok);
    setPending(null);
  };

  return (
    <Ctx.Provider value={confirm}>
      {children}
      {pending && <Dialog {...pending} onClose={close} />}
    </Ctx.Provider>
  );
}

function Dialog({ title, body, confirmLabel = "Confirm", cancelLabel = "Cancel", tone = "default", requireText, onClose }: Pending & { onClose: (ok: boolean) => void }) {
  const [typed, setTyped] = useState("");
  const cancelRef = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const danger = tone === "danger";
  const allowed = !requireText || typed === requireText;

  useEffect(() => {
    (requireText ? inputRef : danger ? cancelRef : confirmRef).current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [danger, requireText, onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "var(--scrim)" }} onMouseDown={() => onClose(false)}>
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        className="panel w-full max-w-md space-y-4 p-5" style={{ boxShadow: "var(--shadow-dialog)" }}
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          // keep Tab inside the dialog
          if (e.key !== "Tab") return;
          const f = Array.from(e.currentTarget.querySelectorAll<HTMLElement>("button, input")).filter((el) => !el.hasAttribute("disabled"));
          if (!f.length) return;
          const first = f[0];
          const last = f[f.length - 1];
          if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last.focus();
          } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }}
      >
        <h2 id="confirm-title" className="text-base font-semibold">
          {title}
        </h2>
        {body && (
          <div className="text-sm" style={{ color: "var(--muted)" }}>
            {body}
          </div>
        )}
        {requireText && (
          <div>
            <label className="label" htmlFor="confirm-text">
              Type <span className="mono normal-case">{requireText}</span> to confirm
            </label>
            <input
              id="confirm-text"
              ref={inputRef}
              className="input mono"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && allowed) onClose(true);
              }}
              autoComplete="off"
            />
          </div>
        )}
        <div className="flex justify-end gap-2">
          <button ref={cancelRef} className="btn-ghost" onClick={() => onClose(false)}>
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            className="btn"
            disabled={!allowed}
            onClick={() => onClose(true)}
            style={danger ? { background: "var(--bad)", color: "var(--on-bad)" } : undefined}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
