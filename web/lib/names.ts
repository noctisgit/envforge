// Names that become directories, container names and vLLM model ids (datasets, adapters).
// The server's rule (train/remote.py check_name): ^[a-z0-9][a-z0-9._-]{0,62}$.

/** What a name field holds while you type: lower case, other characters become "-", no
 *  leading separator. A trailing "-" stays, or you could never type one; finalName trims it. */
export function typingName(s: string, max = 63): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[^a-z0-9]+/, "")
    .slice(0, max);
}

/** The name to save: typingName without a trailing separator. */
export function finalName(s: string, max = 63): string {
  return typingName(s, max).replace(/[._-]+$/, "");
}
