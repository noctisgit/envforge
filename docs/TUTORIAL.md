# Tutorial: your first fine-tune, in the browser

This walks through the whole loop in the web app: score a strong model, score a small open
model, teach the small one with the strong one's solutions, and prove it learned. The second
half is a **worked example with real numbers**: a customer-service environment built from the
ABCD dataset, Claude Sonnet and Fable as teachers, Qwen2.5-3B and 7B as students.

[docs/TRAINING.md](TRAINING.md) explains the same loop from the command line and the ML behind
each step. The app's **Train → Guide** page tracks your progress through the same steps.

**You need:** envforge running (see the [README](../README.md)), a provider key for the teacher
(Anthropic, OpenAI or Vercel) under **Profile**, and a rented GPU for the student. Budget a few
dollars of API calls and a few hours of GPU time.

## The loop at a glance

```
1 score the teacher ──► 2 rent a GPU, serve the student ──► 3 score the student (baseline)
                                                                       │
7 test: same tasks, base vs adapter ◄── 6 fine-tune ◄── 5 dataset ◄── 4 teacher solves training tasks
         │
         └──► 8 optional: DPO / GRPO, or more training tasks
```

Two rules make the result trustworthy:

- **Evaluation and training never share a task.** Evaluation runs use seeds 0 and up, training
  runs use seeds 10000 and up, and the environment keeps the two sets apart. An environment built
  in the Builder splits its fixed task list by task (about a quarter held out), and a task stays
  on its side in every later version.
- **Every model is scored on the same tasks.** For an environment with a fixed task list, tick
  **every task once** on the Runs page: n episodes then cover each of the n held-out tasks
  exactly once. More episodes over the same tasks don't make a score more reliable; more tasks do.

Before you start, set a monthly spending limit under **Settings → General**. Runs stop between
episodes when they reach it.

## 1. Score the teacher

**Runs → new run.** Pick the environment, a strong model (e.g. `anthropic:claude-sonnet-5-5`),
purpose **Evaluation · seeds 0+**, and tick **every task once** if it's offered. Label it, e.g.
`sonnet-eval`.

When it finishes, open the run and **read replays**: three that passed and every failure. Look at
the order of tool calls, where identity gets verified, how refusals are handled. A strong model
failing a task is often a broken task (a goal that contradicts its checks, a check on text the
privacy pass rewrote). Fix those in the Builder and publish a new version before you trust any
score.

## 2. Rent a GPU and serve the open model

- **Pick a card.** LoRA training of a 3B model fits a 24 GB card. A 7B model with long
  conversations (16k tokens) wants 40–48 GB (A6000, L40S, A100).
- **Start a pod** from the image `ghcr.io/noctisgit/envforge-gpu` (RunPod template in
  [docs/COMPUTE.md](COMPUTE.md)). A network volume at `/root/.cache/huggingface` keeps
  downloaded weights across restarts.
- **Let envforge in.** Copy the workspace public key from **Settings → GPU boxes** into
  **RunPod → Settings → SSH Public Keys** *before* starting the pod. A key added inside a running
  pod is lost when the pod restarts.
- **Add the box** under **Settings → GPU boxes**:
  - over **Tailscale**: the pod's tailnet name or `100.x` address, port 22;
  - over **plain SSH**: the pod's public IP and its mapped SSH port (RunPod shows it as
    "SSH over exposed TCP", e.g. `194.68.245.94 -p 22012`). envforge then reaches vLLM through
    an SSH tunnel, so vLLM is never exposed to the internet.

  Click **Check** until it's green, then **Serve** the base model, e.g. `Qwen/Qwen2.5-7B-Instruct`.

If the box drops off the network during a run, the run waits up to 3 minutes for it, then stops
with "stopped after N episode(s)" and keeps what finished. Queue it again once the box is back.

## 3. Score the student before training

Same environment, same purpose, **every task once**, model `vllm:Qwen/Qwen2.5-7B-Instruct`.
Expect a low score. This is the "before" a fine-tune has to beat; without it a fine-tune proves
nothing.

## 4. Collect the teacher's solutions

A run with purpose **Training data · seeds 10000+**, 2–3 passes over the training tasks (for 84
training tasks, 168–252 episodes). A cheaper teacher for the bulk and a stronger one for the hard
tier works well. Only episodes with reward 1.0 will become training examples.

## 5. Build the dataset

**Datasets**: pick the training runs, **Minimum reward** 1.0, **best per task** on (or off to keep
every passing episode; the validation split still stays by task). The preview must say
**Eval overlap: none**. If it doesn't, test tasks got into training: stop and find out why.

## 6. Fine-tune

**Fine-tune**: the dataset, the base model, the box. Run a smoke test first with **Max steps**
20; it must finish cleanly. Then the full run: LoRA rank 16, learning rate 2e-4, 2–3 epochs.
Starting it stops vLLM on the box (serving and training can't share the GPU's memory).

Watch the loss curve. Training loss should fall. If the **validation loss turns up** in the
last epoch, the model has started memorising its training examples: train again with fewer
epochs. When the job finishes, the adapter is copied back to the control plane, so stopping
the pod loses nothing.

## 7. Test it

**Models → your adapter → Test on every held-out task.** It serves base + adapter on the box and
scores the adapter on the newest version of its environment, every held-out task once. If the
base model's score is from an older version, score it again the same way (the box serves the base
next to the adapter, so `vllm:<base model>` works too).

Then **Compare** teacher, base and adapter. Look at each difficulty tier separately, and read
five failed adapter replays: what did it pick up from the teacher, and where does it still go
wrong?

## 8. Past imitation

- **More training tasks.** SFT can only copy what it has seen. Growing the training side (more
  conversations extracted in the Builder, more teacher runs) is often the biggest single gain.
- **DPO** learns from pairs of the student's own attempts at the same task, one solved and one
  failed. Build them in **Datasets** (preference pairs) from a sampling run: several attempts per
  training task at temperature 0.7–1.
- **GRPO** needs no dataset. Each round, the current adapter attempts training tasks several
  times, the environment scores every attempt, and the update favours the attempts that beat
  their group's average. **Fine-tune → GRPO**, starting from your SFT adapter; 3 rounds, 16 tasks
  per round, 8 attempts per task, temperature 0.8, learning rate 1e-5. Then test it like any
  adapter.

## Reading your results

| you see | it means |
|---|---|
| solve rate up on held-out tasks | the student learned the skill; gains usually show on easy tasks first |
| collateral or broken policies go up | it learned to reach the outcome by skipping steps (refunding before verifying); read those replays |
| training loss falls, validation loss rises | memorising: fewer epochs, more training tasks, or a lower learning rate |
| student close to the teacher | the ceiling for imitation; going past it needs reward-based training (DPO, GRPO) |
| two runs a few points apart | probably noise: on n tasks a score moves by about ±(100/√n) points by chance (±18 on 30 tasks, ±7 on 181) |
| GRPO rounds score higher, then lower | the model has nothing left to learn from (attempts at a task all pass or all fail); stop at the best round |

## Sanity checks

- The **eval overlap** in Datasets says none.
- The teacher's failures were read, and the broken tasks among them fixed.
- Teacher, base and adapter were scored on the **same environment version** with **every task
  once**.
- A 20-step smoke test finished before the full fine-tune.
- **Outside the exam:** write a few tasks unlike the training data in the Builder (a customer who
  pushes back after a refusal, two requests in one chat) and run base vs adapter. Run both on a
  different environment (e.g. `netops`) too: a drop there means fine-tuning cost general skill.

## Experiments worth running

Change one thing, keep everything else, compare on the same held-out tasks:

- **data scaling**: 25%, 50%, 100% of the examples;
- **model size**: 3B vs 7B on the same dataset;
- **filtering**: minimum reward 1.0 vs keeping every episode (why rejection sampling matters);
- **epochs**: 1 vs 2 vs 3, and what overfitting looks like on the loss curve and in the score;
- **more tasks** vs more epochs on the same tasks;
- **mixing environments**: one adapter on two environments, without losing either.

---

# Worked example: ABCD customer service

[ABCD](https://github.com/asappresearch/abcd) (Action-Based Conversations Dataset) is about
10,000 customer-service chats for an online clothing store, with the store's agent guidelines:
verify identity, then refund, cancel, update an order, recover a username, and so on.

## The environment

Built in the Builder from an upload of the ABCD conversations, redacted before any model saw
them. Each task is one conversation turned into a goal, a simulated customer who answers the
agent's questions, store records, and checks on the end state (for example "a refund of $69 on
order 6416384753 exists" and "resolved as order_refunded"). A task only counts if its reference
solution earns reward 1.0 in the simulator.

| version | held-out tasks | training tasks | what changed |
|---|---|---|---|
| v3 | 30 | 84 | the first held-out split |
| v4 | 178 | 84 | about 150 more held-out tasks, extracted with Claude Opus from held-out conversations, 3 per kind of request ($5.35), then audited (27 fixes) |
| v5 | **181** (50 easy, 103 medium, 28 hard) | 84 | 6 broken tasks fixed after reading Sonnet's v4 failures; 4 tasks the privacy pass had dropped restored |

The training side never changed, so one dataset served every version.

## Results on v5: 181 held-out tasks, every task once

| model | solved | easy | medium | hard | cost / episode |
|---|---|---|---|---|---|
| Claude Fable 5.1 | **99.4%** | 100% | 99% | 100% | $0.096 |
| Claude Sonnet 5.5 | **96.1%** | 100% | 96% | 89% | $0.018 |
| Qwen2.5-7B + LoRA SFT | **55.8%** | 70% | 51% | 46% | GPU time |
| Qwen2.5-7B, untrained | **5.0%** | 6% | 5% | 4% | GPU time |

Fine-tuning took the 7B model from 9 to 101 of the 181 held-out tasks.

**The fine-tune.** Dataset: 216 training and 27 validation examples, the best solves from Sonnet
and Fable on the 84 training tasks (seeds 10000+), about $7 of teacher runs. LoRA rank 16
(alpha 32), learning rate 2e-4, 3 epochs, max length 16384, batch 1 with 8 gradient
accumulation steps: 81 steps, 1 h 44 min on an RTX A6000 (48 GB). Training loss went from 0.57
to 0.08. Validation loss went 0.290 → **0.268** → 0.283 over the three epochs: the third epoch
had started to memorise, so 2 epochs is the obvious next experiment.

## The earlier round: Qwen2.5-3B on v3

| model (v3, 30 held-out tasks) | solved |
|---|---|
| Claude Fable 5.1 | 100% |
| Claude Sonnet 5.5 | 92% |
| Qwen2.5-3B + SFT | 49% |
| + DPO: the teacher's solve vs its own failure | 49% |
| + DPO: its own solve vs its own failure | 39% |
| + GRPO, 4 rounds | 45% |
| Qwen2.5-3B, untrained | 2% |

On 30 tasks a score moves by about ±18 points by chance, so nothing after SFT is a clear
change. GRPO's training-task score went 49% → 57% → 65% → 53% over the rounds while the number
of tasks with mixed outcomes (the only ones it learns from) fell from 13 to 6. These 3B scores
come from a smaller, easier test set, so they don't compare directly with the 7B on v5.

## What we learned

1. **Fix the test first.** 30 held-out tasks couldn't tell 45% from 49%. Running more episodes over
   them didn't help; extracting 150 more did.
2. **Read the teacher's failures.** 6 of Sonnet's 13 failures on v4 were broken tasks: a cancel
   reason the guidelines handle differently, a goal that asked for one thing while the checks
   expected another. After fixing them, every task a teacher failed on v5 was solved by the other
   teacher.
3. **Watch validation loss, not training loss.** Training loss kept falling through epoch 3;
   validation loss had already turned.
4. **Imitation from 84 tasks goes a long way, then stops.** DPO and GRPO on the same 84 tasks didn't
   beat SFT for the 3B. More training tasks is the next lever.
5. **Rented GPUs drop off the network.** One pod lost its Tailscale connection, then its public SSH
   port, for minutes at a time. Connecting over plain SSH with a tunnel, and runs that wait for
   the box and then stop cleanly, made that survivable.
