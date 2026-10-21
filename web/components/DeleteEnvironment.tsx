"use client";

// Deleting a published (builder-made) environment: one confirm + toast flow shared by the
// Environments list, an environment's page and the builder's published-versions table.

import { useConfirm } from "@/components/ConfirmDialog";
import { useToast } from "@/components/Toast";
import { Api } from "@/lib/api";

/** Returns `remove(wsId, packName)`, which resolves true once the environment is deleted. */
export function useDeleteEnvironment() {
  const confirm = useConfirm();
  const toast = useToast();
  return async function remove(wsId: string, name: string): Promise<boolean> {
    const ok = await confirm({
      title: `Delete ${name}?`,
      body: "It leaves the Environments list and can't start new runs. Runs already made on it keep their results, replays and exports. The builder project is untouched, so you can publish again any time.",
      confirmLabel: "Delete environment",
      tone: "danger",
    });
    if (!ok) return false;
    try {
      await Api.deletePack(wsId, name);
      toast.success(`${name} deleted`);
      return true;
    } catch (e) {
      toast.error(`Couldn't delete ${name}`, e);
      return false;
    }
  };
}
