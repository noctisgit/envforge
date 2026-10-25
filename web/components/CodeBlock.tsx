"use client";

// A shell / code snippet with a Copy button. Used for the command-line steps of the
// training guide and the "use your model" examples.

import { useState } from "react";

export function CodeBlock({ code, label, wrap = false }: { code: string; label?: string; wrap?: boolean }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the text is still selectable */
    }
  }
  return (
    <div className="min-w-0">
      {label && <div className="label">{label}</div>}
      <div className="relative">
        <pre className="code-block overflow-x-auto pr-16 leading-5" style={wrap ? { whiteSpace: "pre-wrap", overflowWrap: "anywhere" } : { whiteSpace: "pre", overflowWrap: "normal" }}>
          {code}
        </pre>
        <button type="button" className="btn-ghost btn-sm absolute top-2 right-2" style={{ background: "var(--code-bg)" }} onClick={copy} aria-label={label ? `Copy ${label}` : "Copy"}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </div>
  );
}
