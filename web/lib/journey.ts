// The user's journey through envforge: what they want to do (a goal) and the steps that
// get them there, each with a status worked out from real data (keys, runs, packs,
// builder projects). Powers the Home page, the onboarding chooser and "next step" cards.
//
// The GPU, dataset, training, test and use steps of distillation happen on the GPU boxes,
// Datasets, Fine-tune and Models pages, so their status comes from those records (a box
// serving, a saved dataset, a trained adapter, a run on it). Work done on the command line
// still counts where it shows up (a vLLM key, a run on an adapter), and "use it" can be
// marked done by hand; that mark, and the chosen goal, live in this browser, per workspace.

import type { AdapterRow, BuilderProjectSummary, DatasetRow, GpuBox, KeyOut, Pack, Run, TrainingJob } from "@/lib/api";

export type GoalId = "compare" | "build" | "distill" | "explore";

export const GOALS: { id: GoalId; title: string; blurb: string; outcome: string }[] = [
  {
    id: "compare",
    title: "Compare models on a task",
    blurb: "Run two or more models on the same environment and see which one solves more, at what cost.",
    outcome: "a side-by-side scorecard",
  },
  {
    id: "build",
    title: "Build an environment from my data",
    blurb: "Describe the job and upload real chat logs; the builder turns them into a runnable, scored environment.",
    outcome: "your own environment, ready to run",
  },
  {
    id: "distill",
    title: "Train a small open model",
    blurb: "Let a strong model show how it's done, keep its best attempts, and fine-tune an open model (LoRA) to copy them.",
    outcome: "a trained model you can serve or download",
  },
  {
    id: "explore",
    title: "Just look around",
    blurb: "Run the offline scripted model on a built-in environment and replay what it did. No key needed.",
    outcome: "a feel for runs, episodes and replays",
  },
];

/** The two seed ranges: evaluate on 0–99, collect training data on 10000+, never overlapping. */
export const EVAL_SEED = 0;
export const TRAIN_SEED = 10000;
export const isTrainingRun = (r: Run) => r.seed_start >= TRAIN_SEED;
export const isEvalRun = (r: Run) => r.seed_start < 100;

export type ModelRole = "teacher" | "student" | "student + adapter" | "demo";

/**
 * What a model is in the distillation story. API models are teachers; a vLLM model with a
 * Hugging Face id (org/name) is an untrained student; a vLLM model without one is an adapter
 * served by name (`envforge compute vllm-up … --lora <name>`).
 */
export function modelRole(model: string): ModelRole {
  if (model === "scripted" || model.startsWith("scripted:")) return "demo";
  if (model.startsWith("vllm:")) return model.slice(5).includes("/") ? "student" : "student + adapter";
  return "teacher";
}

export type StepStatus = "done" | "current" | "todo" | "optional";

export interface Step {
  id: string;
  title: string;
  why: string; // one line: what this step produces / why it matters
  status: StepStatus;
  detail?: string; // what we found, e.g. "2 runs on netops"
  optional?: boolean;
  cli?: boolean; // done on the command line for now
  manual?: boolean; // can be marked done by hand
}

export interface JourneyData {
  keys: KeyOut[];
  runs: Run[];
  packs: Pack[];
  projects: BuilderProjectSummary[];
  marks: Record<string, boolean>; // steps marked done by hand
  boxes?: GpuBox[];
  datasets?: DatasetRow[];
  trainings?: TrainingJob[];
  adapters?: AdapterRow[];
}

const completed = (rs: Run[]) => rs.filter((r) => r.status === "completed");
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;

/** The steps for a goal, in order, with the first unfinished required one marked current. */
export function stepsFor(goal: GoalId, d: JourneyData): Step[] {
  const done = completed(d.runs);
  const verifiedKey = d.keys.find((k) => k.verified_at || (k.provider === "vllm" && k.base_url));
  const apiKey = d.keys.find((k) => k.provider !== "vllm");
  const vllm = d.keys.find((k) => k.provider === "vllm" && k.base_url);
  const evalRuns = done.filter(isEvalRun);
  const steps: Step[] = [];

  const keyStep = (required: boolean): Step => ({
    id: "key",
    title: "Connect a model provider",
    why: "Add an Anthropic, OpenAI or Vercel AI Gateway key so runs can call a model. Keys are encrypted and only used by the worker.",
    status: verifiedKey ? "done" : required ? "todo" : "optional",
    detail: verifiedKey ? `${verifiedKey.provider} key saved` : required ? undefined : "optional: the scripted model needs no key",
    optional: !required,
  });

  if (goal === "explore") {
    steps.push(
      {
        id: "demo-run",
        title: "Run the scripted model",
        why: "It solves every task with the environment's reference solution, offline and free, so you can see what a run produces.",
        status: done.length ? "done" : "todo",
        detail: done.length ? plural(done.length, "finished run") : undefined,
      },
      {
        id: "replay",
        title: "Open a run and replay an episode",
        why: "Step through each tool call the model made and how the world changed.",
        status: d.marks.replay ? "done" : "todo",
        manual: true,
      },
      {
        id: "envs",
        title: "Browse the environments",
        why: "Each one has tools, tasks at three difficulty tiers and a verifier that scores every attempt.",
        status: d.marks.envs ? "done" : "todo",
        manual: true,
      },
      keyStep(false),
    );
  }

  if (goal === "compare") {
    const byPack = new Map<string, Set<string>>();
    for (const r of evalRuns) byPack.set(r.pack, (byPack.get(r.pack) ?? new Set()).add(r.model));
    const best = [...byPack.entries()].sort((a, b) => b[1].size - a[1].size)[0];
    const models = best?.[1].size ?? 0;
    steps.push(
      keyStep(true),
      {
        id: "first-model",
        title: "Run the first model",
        why: "Pick an environment and a model, on evaluation seeds (0–99), so every model sees the same tasks.",
        status: models >= 1 ? "done" : "todo",
        detail: best ? `${best[0]}: ${plural(models, "model")}` : undefined,
      },
      {
        id: "second-model",
        title: "Run another model on the same tasks",
        why: "Same environment, same seeds, different model: the only fair comparison.",
        status: models >= 2 ? "done" : "todo",
      },
      {
        id: "compare",
        title: "Compare them",
        why: "Solve rate, reward, steps, mistakes and cost side by side, per difficulty tier.",
        status: d.marks.compare ? "done" : "todo",
        manual: true,
      },
    );
  }

  if (goal === "build") {
    const custom = d.packs.filter((p) => p.custom);
    const withTasks = d.projects.filter((p) => p.n_tasks > 0);
    const ranCustom = done.some((r) => custom.some((p) => p.name === r.pack));
    steps.push(
      keyStep(true),
      {
        id: "describe",
        title: "Describe the job in the Builder",
        why: "Say what the agent should learn; the builder drafts the world, the tools and the rules.",
        status: d.projects.length ? "done" : "todo",
        detail: d.projects.length ? plural(d.projects.length, "project") : undefined,
      },
      {
        id: "tasks",
        title: "Upload chat logs and extract tasks",
        why: "Real conversations (redacted first) become tasks the verifier can score.",
        status: withTasks.length ? "done" : "todo",
        detail: withTasks.length ? `${withTasks[0].n_tasks} tasks in ${withTasks[0].title || withTasks[0].name}` : undefined,
      },
      {
        id: "publish",
        title: "Publish it",
        why: "A published version shows up under Environments and can be run like any other.",
        status: custom.length ? "done" : "todo",
        detail: custom.length ? custom.map((p) => p.name).join(", ") : undefined,
      },
      {
        id: "run-custom",
        title: "Run a model on it",
        why: "See how well a model does on your own job before you rely on it.",
        status: ranCustom ? "done" : "todo",
      },
    );
  }

  if (goal === "distill") {
    const teacherEval = evalRuns.filter((r) => modelRole(r.model) === "teacher");
    const studentEval = evalRuns.filter((r) => modelRole(r.model) === "student");
    const teacherTrain = done.filter((r) => isTrainingRun(r) && modelRole(r.model) === "teacher");
    const adapterEval = evalRuns.filter((r) => modelRole(r.model) === "student + adapter");
    const solved = teacherTrain.reduce((n, r) => n + Math.round((r.metrics?.overall.solve_rate ?? 0) * r.progress), 0);
    const adapters = d.adapters ?? [];
    const trained = adapters.length > 0 || adapterEval.length > 0 || done.some((r) => modelRole(r.model) === "student + adapter");
    const serving = (d.boxes ?? []).find((b) => b.serve_state === "serving");
    const box = serving ?? (d.boxes ?? [])[0];
    const datasets = d.datasets ?? [];
    const training = (d.trainings ?? []).find((t) => ["queued", "starting", "running", "fetching"].includes(t.status));
    steps.push(
      keyStep(true),
      {
        id: "teacher-eval",
        title: "Score a strong model (the teacher)",
        why: "Run e.g. Claude Sonnet on evaluation seeds. Its score is the target the small model is chasing.",
        status: teacherEval.length ? "done" : "todo",
        detail: teacherEval.length ? `${teacherEval[0].model} on ${teacherEval[0].pack}` : undefined,
      },
      {
        id: "gpu",
        title: "Connect a GPU box with vLLM",
        why: "Rent a GPU (RunPod, Vast, Lambda), add it under Settings → GPU boxes, check it, and serve the open model with vLLM.",
        // stays done once the small model has been scored on it: vLLM is stopped on purpose for training
        status: serving || vllm || (box && studentEval.length) ? "done" : "todo",
        detail: serving
          ? `${serving.name} serving ${serving.served_model}`
          : vllm
            ? (vllm.base_url ?? undefined)
            : box
              ? `${box.name} added, vLLM ${studentEval.length ? "stopped (training needs the GPU)" : "not serving yet"}`
              : undefined,
      },
      {
        id: "student-eval",
        title: "Score the small model before training",
        why: "Run the untrained open model on the same evaluation seeds: the baseline you will beat.",
        status: studentEval.length ? "done" : "todo",
        detail: studentEval.length ? `${studentEval[0].model.slice(5)}` : undefined,
      },
      {
        id: "teacher-train",
        title: "Collect the teacher's solutions",
        why: "Run the teacher on training seeds (10000+), never the evaluation ones. A few hundred clean solves is a good first dataset.",
        status: teacherTrain.length ? "done" : "todo",
        detail: teacherTrain.length ? `≈ ${solved} solved episodes in ${plural(teacherTrain.length, "run")}` : undefined,
      },
      {
        id: "dataset",
        title: "Build the training dataset",
        why: "Keep only the clean solves, one per task, with a held-out validation split.",
        status: datasets.length || trained || d.marks.dataset ? "done" : "todo",
        detail: datasets.length ? `${datasets[0].name}: ${datasets[0].n_train} training examples` : undefined,
      },
      {
        id: "train",
        title: "Fine-tune a LoRA adapter",
        why: "Train a small add-on to the open model on the GPU box (about 20–60 min for a 7B model). Starting it stops vLLM there: serving and training can't share the GPU's memory.",
        status: trained || d.marks.train ? "done" : "todo",
        detail: adapters.length ? `${adapters[0].name} on ${adapters[0].base_model}` : training ? `${training.adapter_name}: ${training.status}` : undefined,
      },
      {
        id: "test",
        title: "Test the trained model",
        why: "Serve base + adapter and run the same evaluation seeds. The gain over the baseline is what it learned.",
        status: adapterEval.length ? "done" : "todo",
        detail: adapterEval.length ? adapterEval[0].model : undefined,
      },
      {
        id: "use",
        title: "Use it",
        why: "Call it through vLLM's OpenAI-compatible API from your own apps, or download the adapter.",
        status: d.marks.use ? "done" : "todo",
        manual: true,
      },
      {
        id: "beyond",
        title: "Go past imitation: DPO or GRPO",
        why: "SFT copies the teacher. DPO learns from pairs of solved and failed attempts at the same task; GRPO from rounds of the model's own attempts, scored by the environment.",
        status: adapters.some((a) => a.method === "dpo" || a.method === "grpo") ? "done" : "todo",
        detail: adapters.find((a) => a.method === "dpo" || a.method === "grpo")?.name,
        optional: true,
      },
    );
  }

  // the first unfinished required step is the one to do now; unfinished optional ones stay optional
  let found = false;
  for (const s of steps) {
    if (s.status === "done") continue;
    if (s.optional) {
      s.status = "optional";
      continue;
    }
    s.status = found ? "todo" : "current";
    found = true;
  }
  return steps;
}

/** API keys that let a run call a teacher model (any non-vLLM provider). */
export const hasApiKey = (keys: KeyOut[]) => keys.some((k) => k.provider !== "vllm");

// ---- per-workspace choices kept in this browser ----

const goalKey = (ws: string) => `envforge.goal.${ws}`;
const marksKey = (ws: string) => `envforge.marks.${ws}`;

export function loadGoal(ws: string): GoalId | null {
  try {
    const g = localStorage.getItem(goalKey(ws));
    return GOALS.some((x) => x.id === g) ? (g as GoalId) : null;
  } catch {
    return null;
  }
}

export function saveGoal(ws: string, goal: GoalId | null) {
  try {
    if (goal) localStorage.setItem(goalKey(ws), goal);
    else localStorage.removeItem(goalKey(ws));
  } catch {
    /* storage blocked: the choice lasts for this page only */
  }
}

export function loadMarks(ws: string): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(marksKey(ws)) || "{}");
  } catch {
    return {};
  }
}

export function saveMark(ws: string, id: string, on: boolean) {
  const m = loadMarks(ws);
  if (on) m[id] = true;
  else delete m[id];
  try {
    localStorage.setItem(marksKey(ws), JSON.stringify(m));
  } catch {
    /* ignore */
  }
}
