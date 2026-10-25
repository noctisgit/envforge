"""Request bodies for GPU boxes, datasets, training jobs and adapters.

Responses are plain dicts built in compute_service / training_service, which decide
what may leave the server (never a key, only masked forms).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoxIn(BaseModel):
    name: str = Field(min_length=1, max_length=63)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    user: str = Field(default="root", min_length=1, max_length=64)
    # empty: log in with the workspace's SSH key (Settings → GPU boxes shows its public half)
    ssh_private_key: str | None = Field(default=None, max_length=20_000)
    runtime: Literal["docker", "pod", "auto"] = "auto"
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    endpoint_host: str | None = Field(default=None, max_length=255)
    bind: str | None = Field(default=None, max_length=64)
    # how the control plane reaches vLLM; None picks from the host (tailnet: direct, public: tunnel)
    vllm_access: Literal["direct", "tunnel"] | None = None
    vllm_api_key: str | None = Field(default=None, max_length=500)
    hf_token: str | None = Field(default=None, max_length=500)
    hourly_usd: float | None = Field(default=None, ge=0, le=1000)


class VllmKeyIn(BaseModel):
    key: str = Field(min_length=16, max_length=500)


class ServeIn(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    adapters: list[str] = Field(default_factory=list, max_length=16)
    max_model_len: int | None = Field(default=None, ge=256, le=1_048_576)
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=0.98)


class DatasetFilterIn(BaseModel):
    run_ids: list[str] = Field(min_length=1, max_length=50)
    min_reward: float = Field(default=1.0, ge=0, le=1)
    val_frac: float = Field(default=0.05, ge=0, le=0.5)
    best_per_task: bool = True
    # sft: solved episodes; preference: chosen/rejected pairs of the same task for DPO
    kind: Literal["sft", "preference"] = "sft"
    max_pairs_per_task: int = Field(default=4, ge=1, le=32)


class DatasetIn(DatasetFilterIn):
    name: str = Field(min_length=1, max_length=63)


class TrainIn(BaseModel):
    method: Literal["sft", "dpo", "grpo"] = "sft"
    dataset_id: str | None = Field(default=None, max_length=32)  # sft / dpo; GRPO makes its own
    box_id: str = Field(max_length=32)
    base_model: str = Field(min_length=1, max_length=200)
    adapter_name: str | None = Field(default=None, max_length=63)
    init_adapter: str | None = Field(default=None, max_length=63)  # dpo / grpo start from it
    pack: str | None = Field(default=None, max_length=80)  # grpo: the environment to train on
    # dpo / grpo hyper-parameters (ranges are checked in training_service)
    beta: float | None = None
    kl_beta: float | None = None
    clip_eps: float | None = None
    rounds: int | None = None
    tasks_per_round: int | None = None
    group_size: int | None = None
    temperature: float | None = None
    epochs: float | None = None
    lr: float | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    max_len: int | None = None
    batch_size: int | None = None
    grad_accum: int | None = None
    max_steps: int | None = None


class AdapterBoxIn(BaseModel):
    box_id: str = Field(max_length=32)
    episodes: int = Field(default=50, ge=1, le=1000)
