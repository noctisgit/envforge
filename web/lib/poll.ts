"use client";

import { useEffect, useRef } from "react";

/** Call `fn` every `ms` while `active` is true (the latest `fn`, without restarting the timer). */
export function usePoll(fn: () => void, active: boolean, ms = 3000) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => ref.current(), ms);
    return () => clearInterval(t);
  }, [active, ms]);
}
