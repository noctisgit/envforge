// Initials marks: a round one for people (Avatar) and a square one for workspaces
// (WorkspaceMark, a stable colour per name so workspaces are easy to tell apart).

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
      style={{ width: size, height: size, fontSize: Math.max(10, size * 0.38), background: "var(--accent)", color: "var(--on-accent)" }}
    >
      {initials(name, email)}
    </span>
  );
}

export function WorkspaceMark({ name, size = 20 }: { name: string; size?: number }) {
  const hue = [...name].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);
  return (
    <span
      aria-hidden
      className="inline-flex shrink-0 items-center justify-center font-semibold"
      style={{ width: size, height: size, borderRadius: size >= 28 ? 6 : 4, fontSize: Math.max(11, size * 0.45), background: `hsl(${hue} 45% 28%)`, color: `hsl(${hue} 80% 85%)` }}
    >
      {(name.trim()[0] ?? "?").toUpperCase()}
    </span>
  );
}
