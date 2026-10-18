"use client";

// One auth context for the whole app. Underneath it is either:
//   local  -> cookie session issued by FastAPI (/api/auth/login)
//   clerk  -> Clerk session; we forward its JWT as a Bearer token to FastAPI
// Pages only ever use useAuthState() and never know which one is active.

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { Api, ApiError, setTokenGetter, type User } from "./api";

export const AUTH_PROVIDER = (process.env.NEXT_PUBLIC_AUTH_PROVIDER ?? "local") as "local" | "clerk";

interface AuthState {
  user: User | null;
  loading: boolean;
  refresh: () => Promise<void>;
  logout: () => Promise<void>;
}

const Ctx = createContext<AuthState>({ user: null, loading: true, refresh: async () => {}, logout: async () => {} });

function BaseProvider({ children, getToken, clerkSignOut }: { children: React.ReactNode; getToken?: () => Promise<string | null>; clerkSignOut?: () => Promise<void> }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTokenGetter(getToken ?? null);
  }, [getToken]);

  const refresh = useCallback(async () => {
    try {
      setUser(await Api.me());
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setUser(null);
      else throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const logout = useCallback(async () => {
    if (clerkSignOut) await clerkSignOut();
    else await Api.logout();
    setUser(null);
  }, [clerkSignOut]);

  return <Ctx.Provider value={{ user, loading, refresh, logout }}>{children}</Ctx.Provider>;
}

function ClerkBridge({ children }: { children: React.ReactNode }) {
  // Imported lazily so the local build doesn't need Clerk configured.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { useAuth } = require("@clerk/nextjs") as typeof import("@clerk/nextjs");
  const { getToken, signOut, isLoaded } = useAuth();
  const tokenGetter = useCallback(() => getToken(), [getToken]);
  const signOutFn = useCallback(async () => {
    await signOut();
  }, [signOut]);
  if (!isLoaded) return null;
  return (
    <BaseProvider getToken={tokenGetter} clerkSignOut={signOutFn}>
      {children}
    </BaseProvider>
  );
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  if (AUTH_PROVIDER === "clerk") {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { ClerkProvider } = require("@clerk/nextjs") as typeof import("@clerk/nextjs");
    return (
      <ClerkProvider>
        <ClerkBridge>{children}</ClerkBridge>
      </ClerkProvider>
    );
  }
  return <BaseProvider>{children}</BaseProvider>;
}

export function useAuthState() {
  return useContext(Ctx);
}
