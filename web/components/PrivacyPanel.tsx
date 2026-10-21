"use client";

// Builder Data tab: privacy settings, the measured privacy review, and the corpus overview.
//
// The review: a person reads a random sample of *redacted* conversations and ticks each
// one where they can still see personal data. The server turns that into a leak rate
// with a 95% confidence interval; the review passes only if the interval's upper end is
// within the project's threshold (envforge/builder/review.py).

import { Fragment, useCallback, useEffect, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { errorText, useToast } from "@/components/Toast";
import {
  builderCorpus,
  clusterExamples,
  getPrivacyReview,
  setBuilderPrivacy,
  startPrivacyReview,
  submitPrivacyReview,
  type BuilderProject,
  type CorpusConversation,
  type CorpusOverview,
  type PrivacyPatch,
  type PrivacyProfile,
  type PrivacyReview,
} from "@/lib/api";

const pct = (x: number) => `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;

const PROFILES: { id: PrivacyProfile; label: string; hint: string }[] = [
  { id: "off", label: "off", hint: "public or synthetic data: nothing is redacted" },
  { id: "standard", label: "standard", hint: "emails, phones, cards, IDs, addresses, names after cues…" },
  { id: "strict", label: "strict (HIPAA-style)", hint: "Safe Harbor identifiers; extraction waits for a passed review" },
];

// ---- settings + review status --------------------------------------------------------------

export function PrivacyPanel({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const confirm = useConfirm();
  const toast = useToast();
  const pv = project.privacy;
  const [reviewing, setReviewing] = useState(false);
  const hasData = project.uploads.some((u) => u.status === "ready");

  async function patch(p: PrivacyPatch, done?: string) {
    try {
      await setBuilderPrivacy(wsId, project.id, p);
      if (done) toast.success(done, "Applies to uploads processed from now on.");
      onChange();
    } catch (e) {
      toast.error("Couldn't change the privacy settings", e);
    }
  }

  async function changeProfile(profile: PrivacyProfile) {
    if (profile === pv.profile) return;
    const bodies: Record<PrivacyProfile, string> = {
      off: "Nothing in future uploads will be redacted: names, emails, phone numbers and IDs go to the builder model and into the environment as they are. Only choose this for public or synthetic data (e.g. a published research dataset).",
      standard: "Best-effort redaction of emails, phone numbers, cards, IDs, addresses and names after cues. Applies to uploads processed from now on — re-upload existing files to re-redact them.",
      strict: "Aims at HIPAA Safe Harbor's identifier list: all dates except the year, ZIP codes truncated, ages over 89, record/account/licence numbers, serials, URLs, all IPs and more. Applies to uploads processed from now on. Extraction waits until a privacy review passes.",
    };
    const ok = await confirm({
      title: profile === "off" ? "Turn redaction off?" : `Switch to the ${profile} profile?`,
      body: bodies[profile],
      confirmLabel: profile === "off" ? "Turn off redaction" : "Switch",
      tone: profile === "off" ? "danger" : "default",
      ...(profile === "off" ? { requireText: "public data" } : {}),
    });
    if (ok) await patch({ profile, confirm_off: profile === "off" }, profile === "off" ? "Redaction turned off" : `Switched to the ${profile} profile`);
  }

  const rv = pv.review;
  const res = rv?.result;
  const border = pv.blocked ? "var(--bad)" : pv.needs_signoff ? "var(--warn)" : "var(--line)";

  return (
    <section className="panel space-y-3 p-4" style={{ borderColor: border, background: "var(--input-bg)" }} aria-label="Privacy">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="font-semibold">Privacy</h3>
        <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Redaction profile">
          {PROFILES.map((p) => (
            <button
              key={p.id}
              role="radio"
              aria-checked={pv.profile === p.id}
              title={p.hint}
              onClick={() => changeProfile(p.id)}
              className="rounded-full border px-2.5 py-1 text-xs outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent)]"
              style={{
                borderColor: pv.profile === p.id ? (p.id === "off" ? "var(--warn)" : "var(--accent)") : "var(--line)",
                color: pv.profile === p.id ? (p.id === "off" ? "var(--warn)" : "var(--accent)") : "var(--muted)",
                background: pv.profile === p.id ? (p.id === "off" ? "transparent" : "var(--accent-tint)") : "transparent",
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
        <span className="hint">{PROFILES.find((p) => p.id === pv.profile)?.hint}</span>
      </div>

      {pv.profile !== "off" && (
        <div className="grid grid-cols-1 gap-x-6 gap-y-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
          <label className="flex items-start gap-2" title="Replace personal data with realistic fake values instead of <NAME_1>-style placeholders.">
            <input type="checkbox" className="mt-0.5" checked={pv.surrogates} onChange={(e) => patch({ surrogates: e.target.checked })} />
            <span>
              <span className="font-medium">realistic stand-ins</span>
              <span className="block" style={{ color: "var(--muted)" }}>
                “Maria Lopez” instead of &lt;NAME_1&gt; — closer to real conversations for training
              </span>
            </span>
          </label>
          <label className="flex items-start gap-2" title="Local name/location detection (Presidio + spaCy). Runs on this server; no data leaves it.">
            <input type="checkbox" className="mt-0.5" checked={pv.ner} onChange={(e) => patch({ ner: e.target.checked })} />
            <span>
              <span className="font-medium">name detection (NER)</span>
              <span className="block" style={{ color: "var(--muted)" }}>
                finds names and places without cues; local, fast
              </span>
            </span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-medium">second pass (PII model)</span>
            <select className="input !py-1 text-xs" aria-label="Second pass (PII model)" value={pv.tier2} onChange={(e) => patch({ tier2: e.target.value as "off" | "auto" | "all" })}>
              <option value="off">off</option>
              <option value="auto">risky conversations only (recommended)</option>
              <option value="all">every conversation (slow)</option>
            </select>
            <span style={{ color: "var(--muted)" }}>a local span model re-checks what the rules may have missed</span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-medium">review threshold</span>
            <select className="input !py-1 text-xs" aria-label="Review threshold" value={String(pv.threshold)} onChange={(e) => patch({ threshold: Number(e.target.value) })}>
              {[0.01, 0.02, 0.05, 0.1].map((t) => (
                <option key={t} value={String(t)}>
                  at most {pct(t)} of conversations leak
                </option>
              ))}
            </select>
            <span style={{ color: "var(--muted)" }}>a clean review needs about {pv.review_size} conversations</span>
          </label>
        </div>
      )}

      {pv.profile !== "off" && (
        <div className="flex flex-wrap items-center gap-3 border-t pt-3 text-xs" style={{ borderColor: "var(--line)" }}>
          <ReviewStatus review={rv} />
          <button className={`${pv.needs_signoff || pv.blocked ? "btn" : "btn-ghost"} ml-auto`} disabled={!hasData} title={hasData ? "" : "upload some data first"} onClick={() => setReviewing(true)}>
            {rv?.status === "open" ? "Continue review" : res ? "New review" : "Review a sample"}
          </button>
        </div>
      )}
      {pv.blocked && (
        <div className="text-xs" style={{ color: "var(--bad)" }}>
          The last review measured too much personal data left. Extraction is blocked: tighten the settings above (stricter profile, name
          detection, second pass), delete and re-upload the files, then review again.
        </div>
      )}
      {!pv.blocked && pv.needs_signoff && (
        <div className="text-xs" style={{ color: "var(--warn)" }}>
          Strict profile: extraction is blocked until a privacy review passes.
        </div>
      )}
      {res && res.flagged > 0 && rv?.status !== "open" && (
        <div className="text-xs" style={{ color: "var(--muted)" }}>
          Flagged in the last review: <span className="mono">{rv?.flagged?.slice(0, 8).join(", ")}</span>
          {(rv?.flagged?.length ?? 0) > 8 && " …"}
        </div>
      )}
      {reviewing && (
        <ReviewDialog
          wsId={wsId}
          project={project}
          onClose={(changed) => {
            setReviewing(false);
            if (changed) onChange();
          }}
        />
      )}
    </section>
  );
}

function ReviewStatus({ review }: { review: PrivacyReview | null }) {
  if (!review) return <span style={{ color: "var(--muted)" }}>Not reviewed yet: check a random sample of redacted conversations for anything personal left.</span>;
  const r = review.result;
  if (review.status === "open") return <span style={{ color: "var(--warn)" }}>Review in progress: {review.n} conversations sampled.</span>;
  if (!r) return <span style={{ color: "var(--muted)" }}>Review {review.status}.</span>;
  const colour = review.status === "passed" ? "var(--good)" : review.status === "failed" ? "var(--bad)" : "var(--muted)";
  return (
    <span>
      <span style={{ color: colour }}>
        {review.status === "passed" ? "✓ passed" : review.status === "failed" ? "✗ failed" : "outdated (settings or data changed)"}
      </span>{" "}
      <span style={{ color: "var(--muted)" }}>
        · {r.flagged} of {r.reviewed} flagged · leak rate {pct(r.leak_rate)}
        {r.exact ? " (whole corpus checked)" : ` · 95% CI ${pct(r.ci_low)}–${pct(r.ci_high)}`} · threshold {pct(r.threshold)}
      </span>
    </span>
  );
}

// ---- the review itself -------------------------------------------------------------------

const PLACEHOLDER = /(<[A-Z_]+_\d+>)/g;

function Highlighted({ text }: { text: string }) {
  return (
    <>
      {text.split(PLACEHOLDER).map((part, i) =>
        PLACEHOLDER.test(part) ? (
          <span key={i} className="mono rounded px-1 text-[11px]" style={{ background: "var(--pill-running-bg)", color: "var(--accent)" }}>
            {part}
          </span>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

function ReviewDialog({ wsId, project, onClose }: { wsId: string; project: BuilderProject; onClose: (changed: boolean) => void }) {
  const toast = useToast();
  const [review, setReview] = useState<PrivacyReview | null>(null);
  const [convs, setConvs] = useState<CorpusConversation[]>([]);
  const [flagged, setFlagged] = useState<Set<string>>(new Set());
  const [i, setI] = useState(0);
  const [busy, setBusy] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [changed, setChanged] = useState(false);
  const draftKey = review ? `envforge-review-${review.id}` : null;

  useEffect(() => {
    (async () => {
      try {
        let r = await getPrivacyReview(wsId, project.id);
        if (r.review?.status !== "open") {
          const s = await startPrivacyReview(wsId, project.id);
          r = { review: s.review, conversations: s.conversations };
          setChanged(true);
        }
        setReview(r.review);
        setConvs(r.conversations);
        try {
          const saved = localStorage.getItem(`envforge-review-${r.review!.id}`);
          if (saved) setFlagged(new Set(JSON.parse(saved)));
        } catch {}
      } catch (e) {
        setErr(errorText(e));
      } finally {
        setBusy(false);
      }
    })();
  }, [wsId, project.id]);

  const toggle = useCallback(
    (id: string) => {
      setFlagged((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        try {
          if (draftKey) localStorage.setItem(draftKey, JSON.stringify([...next]));
        } catch {}
        return next;
      });
    },
    [draftKey],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "Escape") onClose(changed);
      else if (e.key === "ArrowRight" || e.key === "j") setI((x) => Math.min(x + 1, convs.length - 1));
      else if (e.key === "ArrowLeft" || e.key === "k") setI((x) => Math.max(x - 1, 0));
      else if (e.key === "f" && convs[i]) toggle(convs[i].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [convs, i, toggle, onClose, changed]);

  async function submit() {
    if (!review) return;
    setBusy(true);
    setErr(null);
    try {
      const p = await submitPrivacyReview(wsId, project.id, review.id, [...flagged]);
      try {
        if (draftKey) localStorage.removeItem(draftKey);
      } catch {}
      const r = p.privacy.review;
      if (r?.status === "passed") toast.success("Privacy review passed", `${r.result?.flagged ?? 0} of ${r.result?.reviewed ?? convs.length} conversations flagged.`);
      else if (r?.status === "failed") toast.warning("Privacy review failed", "Too much personal data is left; extraction is blocked until a review passes.");
      else toast.success("Review submitted");
      onClose(true);
    } catch (e) {
      toast.error("Couldn't submit the review", e);
      setBusy(false);
    }
  }

  const c = convs[i];
  const seen = Math.min(convs.length, i + 1);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "var(--scrim)" }} role="dialog" aria-modal aria-labelledby="review-title">
      <div className="panel flex max-h-[92vh] w-full max-w-4xl flex-col" style={{ boxShadow: "var(--shadow-dialog)" }}>
        <div className="flex flex-wrap items-center gap-3 border-b px-5 py-3" style={{ borderColor: "var(--line)" }}>
          <h2 id="review-title" className="text-base font-semibold">
            Privacy review
          </h2>
          {review && (
            <div className="hint">
              conversation {seen} of {convs.length} · {flagged.size} flagged · random sample of {review.population.toLocaleString()} unique
              conversations
            </div>
          )}
          <button className="btn-ghost btn-sm ml-auto" onClick={() => onClose(changed)}>
            Close (keeps progress)
          </button>
        </div>
        <div className="border-b px-5 py-2 text-xs" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>
          Read each conversation and flag it if you can still see <strong>anything that identifies a real person</strong>: a name, email,
          phone, address, account or order number tied to someone, date of birth… Highlighted tokens like{" "}
          <span className="mono" style={{ color: "var(--accent)" }}>
            &lt;NAME_1&gt;
          </span>{" "}
          were already removed. Keys: ← → to move, F to flag.
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {busy && !c && <div style={{ color: "var(--muted)" }}>Drawing a sample…</div>}
          {err && !c && <div style={{ color: "var(--bad)" }}>{err}</div>}
          {c && (
            <div className="space-y-2 text-sm">
              <div className="mono text-xs" style={{ color: "var(--muted)" }}>
                {c.source_id ?? c.id}
                {c.cluster !== null && ` · topic ${c.cluster}`}
              </div>
              {c.turns.map((t, k) => (
                <div key={k} className="flex gap-3">
                  <span className="w-20 shrink-0 text-xs uppercase tracking-wide" style={{ color: t.role === "customer" || t.role === "user" ? "var(--warn)" : "var(--muted)" }}>
                    {t.role}
                  </span>
                  <span className="min-w-0 whitespace-pre-wrap [overflow-wrap:anywhere]">
                    <Highlighted text={t.text} />
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t px-5 py-3" style={{ borderColor: "var(--line)" }}>
          <button className="btn-ghost" disabled={i === 0} onClick={() => setI(i - 1)}>
            ← Previous
          </button>
          {c && (
            <button
              className="btn-ghost"
              aria-pressed={flagged.has(c.id)}
              style={{
                borderColor: flagged.has(c.id) ? "var(--bad)" : undefined,
                color: flagged.has(c.id) ? "var(--bad)" : undefined,
                background: flagged.has(c.id) ? "var(--bad-tint)" : undefined,
              }}
              onClick={() => toggle(c.id)}
            >
              {flagged.has(c.id) ? "⚑ Flagged: personal data left" : "Flag: personal data left"}
            </button>
          )}
          <button className="btn-ghost" disabled={i >= convs.length - 1} onClick={() => setI(i + 1)}>
            Next →
          </button>
          <div className="ml-auto flex items-center gap-3">
            <button className="btn" disabled={busy || !review || seen < convs.length} title={seen < convs.length ? "Read every conversation first" : ""} onClick={submit}>
              Submit review ({flagged.size} flagged)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---- corpus overview ------------------------------------------------------------------------

export function CorpusPanel({ wsId, project }: { wsId: string; project: BuilderProject }) {
  const [ov, setOv] = useState<CorpusOverview | null | undefined>(undefined);
  const [open, setOpen] = useState<number | null>(null);
  const [examples, setExamples] = useState<Record<number, CorpusConversation[]>>({});
  const readyKey = project.uploads.map((u) => `${u.id}:${u.status}`).join(",");

  useEffect(() => {
    builderCorpus(wsId, project.id)
      .then((r) => setOv(r.corpus))
      .catch(() => setOv(null));
  }, [wsId, project.id, readyKey]);

  async function toggle(cid: number) {
    setOpen(open === cid ? null : cid);
    if (!examples[cid]) {
      const r = await clusterExamples(wsId, project.id, cid, 3);
      setExamples((e) => ({ ...e, [cid]: r.conversations }));
    }
  }

  if (!ov) return null;
  return (
    <section className="inset-card space-y-3 p-4" aria-label="Corpus">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h3 className="font-semibold">Corpus</h3>
        <span>{ov.conversations.toLocaleString()} conversations</span>
        <span style={{ color: "var(--muted)" }}>
          {ov.unique.toLocaleString()} unique · {ov.exact_duplicates.toLocaleString()} exact and {ov.near_duplicates.toLocaleString()} near duplicates
          skipped · {ov.clusters.length} topic clusters
        </span>
        {!!ov.stale?.length && <span className="text-xs" style={{ color: "var(--warn)" }}>updating {ov.stale.join(", ")}…</span>}
      </div>
      <p className="hint">
        Similar conversations are grouped by their wording; the builder samples across these topics so a small extraction still covers the
        whole corpus. Click a topic to read examples (redacted).
      </p>
      <div className="overflow-x-auto">
        <table className="data inset">
          <thead>
            <tr>
              <th className="w-12">#</th>
              <th>topic (top terms)</th>
              <th className="w-28 text-right">conversations</th>
              <th className="w-40">share</th>
            </tr>
          </thead>
          <tbody>
            {ov.clusters.map((cl) => (
              <Fragment key={cl.id}>
                <tr className="cursor-pointer" aria-expanded={open === cl.id} onClick={() => toggle(cl.id)}>
                  <td className="mono text-xs" style={{ color: "var(--muted)" }}>
                    {cl.id}
                  </td>
                  <td>
                    <span className="font-medium">{cl.label}</span>
                    {cl.terms.length > 0 && <span style={{ color: "var(--muted)" }}> · {cl.terms.join(", ")}</span>}
                  </td>
                  <td className="num text-right">{cl.size.toLocaleString()}</td>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: "var(--line)" }}>
                        <div className="h-full rounded-full" style={{ width: `${Math.max(2, cl.share * 100)}%`, background: "var(--accent)" }} />
                      </div>
                      <span className="num w-10 text-right text-xs" style={{ color: "var(--muted)" }}>
                        {pct(cl.share)}
                      </span>
                    </div>
                  </td>
                </tr>
                {open === cl.id && (
                  <tr>
                    <td />
                    <td colSpan={3}>
                      {!examples[cl.id] && <span style={{ color: "var(--muted)" }}>Loading…</span>}
                      <div className="space-y-3">
                        {examples[cl.id]?.map((c) => (
                          <div key={c.id} className="rounded-md border p-3 text-xs" style={{ borderColor: "var(--line)", background: "var(--panel)" }}>
                            <div className="mono mb-1" style={{ color: "var(--muted)" }}>
                              {c.source_id ?? c.id}
                            </div>
                            {c.turns.slice(0, 8).map((t, k) => (
                              <div key={k} className="[overflow-wrap:anywhere]">
                                <span style={{ color: "var(--muted)" }}>{t.role}:</span> <Highlighted text={t.text.slice(0, 400)} />
                              </div>
                            ))}
                            {c.turns.length > 8 && <div style={{ color: "var(--muted)" }}>… {c.turns.length - 8} more turns</div>}
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
