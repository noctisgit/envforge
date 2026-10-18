"use client";

// The active workspace, remembered per browser. Everything under /(app) reads it.

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { Api, type Workspace } from "./api";
import { useAuthState } from "./auth";

interface WsState {
  workspaces: Workspace[];
  current: Workspace | null;
  select: (id: string) => void;
  reload: () => Promise<void>;
}

const Ctx = createContext<WsState>({ workspaces: [], current: null, select: () => {}, reload: async () => {} });
const KEY = "envforge.workspace";

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuthState();
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!user) return;
    const list = await Api.workspaces();
    setWorkspaces(list);
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(KEY);
    } catch {}
    const pick = list.find((w) => w.id === saved) ?? list[0] ?? null;
    setCurrentId(pick?.id ?? null);
  }, [user]);

  useEffect(() => {
    reload();
  }, [reload]);

  const select = (id: string) => {
    setCurrentId(id);
    try {
      localStorage.setItem(KEY, id);
    } catch {}
  };

  const current = workspaces.find((w) => w.id === currentId) ?? null;
  return <Ctx.Provider value={{ workspaces, current, select, reload }}>{children}</Ctx.Provider>;
}

export function useWorkspace() {
  return useContext(Ctx);
}
