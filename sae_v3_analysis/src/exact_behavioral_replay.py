#!/usr/bin/env python3
"""
Exact behavioral replay helpers for steering experiments.

These utilities replay the original behavioral experiment conditions as closely
as possible by:
  1. sampling condition profiles from the actual behavioral JSON catalogs,
  2. reusing the original prompt builders, parsers, retry logic, and game
     mechanics from the behavioral runners, and
  3. routing generation through the local hooked model used by steering.

This module is the canonical bridge between:
  - raw behavioral data under data/behavioral/*
  - hidden-state extraction under data/sae_features_v3/*
  - steering follow-up experiments under sae_v3_analysis/src/*
"""

from __future__ import annotations

import json
import random
import sys
import types
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import os

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIORAL_ROOT = Path(
    os.environ.get(
        "LLM_ADDICTION_BEHAVIORAL_ROOT",
        "/home/v-seungplee/data/llm-addiction/behavioral",
    )
)

SLOT_MACHINE_SRC = REPO_ROOT / "paper_experiments" / "slot_machine_6models" / "src"
ALT_PARADIGMS_SRC = REPO_ROOT / "exploratory_experiments" / "alternative_paradigms" / "src"

for path in (str(SLOT_MACHINE_SRC), str(ALT_PARADIGMS_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass(frozen=True)
class BehavioralCondition:
    task: str
    model_name: str
    prompt_condition: str
    bet_type: str
    bet_constraint: Optional[str]


def _normalize_prompt_condition(task: str, prompt_condition: str) -> str:
    prompt_condition = (prompt_condition or "BASE").strip().upper()
    if task in {"sm", "mw"}:
        # Older runs sometimes used R instead of H for "hidden patterns".
        prompt_condition = prompt_condition.replace("R", "H")
    return prompt_condition


def _iter_behavioral_games(task: str, model_name: str):
    if task == "sm":
        data_dir = BEHAVIORAL_ROOT / "slot_machine" / f"{model_name}_v4_role"
        files = sorted(data_dir.glob("final_*.json"))
    elif task == "ic":
        data_dir = BEHAVIORAL_ROOT / "investment_choice" / f"v2_role_{model_name}"
        files = sorted(data_dir.glob(f"{model_name}_investment_*.json"))
    elif task == "mw":
        data_dir = BEHAVIORAL_ROOT / "mystery_wheel" / f"{model_name}_v2_role"
        files = sorted(data_dir.glob(f"{model_name}_mysterywheel_*.json"))
    else:
        raise ValueError(f"Unknown task: {task}")

    for path in files:
        if path.suffix != ".json":
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        games = payload.get("results", payload.get("games", []))
        for game in games:
            yield game


@lru_cache(maxsize=None)
def get_behavioral_catalog(task: str, model_name: str) -> tuple[BehavioralCondition, ...]:
    catalog: list[BehavioralCondition] = []
    for game in _iter_behavioral_games(task, model_name):
        prompt_condition = game.get("prompt_condition", game.get("prompt_combo", "BASE"))
        catalog.append(
            BehavioralCondition(
                task=task,
                model_name=model_name,
                prompt_condition=_normalize_prompt_condition(task, prompt_condition),
                bet_type=game.get("bet_type", "fixed"),
                bet_constraint=(
                    None if task == "sm" else str(game.get("bet_constraint", "unknown"))
                ),
            )
        )
    if not catalog:
        raise FileNotFoundError(f"No behavioral catalog found for task={task}, model={model_name}")
    return tuple(catalog)


@lru_cache(maxsize=None)
def _catalog_permutation(task: str, model_name: str) -> tuple[int, ...]:
    catalog = get_behavioral_catalog(task, model_name)
    seed = sum(ord(ch) for ch in f"{task}:{model_name}:behavioral-replay")
    rng = np.random.RandomState(seed)
    order = np.arange(len(catalog))
    rng.shuffle(order)
    return tuple(int(i) for i in order)


def select_behavioral_condition(
    task: str, model_name: str, game_index: int
) -> BehavioralCondition:
    catalog = get_behavioral_catalog(task, model_name)
    order = _catalog_permutation(task, model_name)
    return catalog[order[game_index % len(catalog)]]


class HookedModelLoader:
    """Minimal ModelLoader-compatible shim with steering hooks."""

    def __init__(self, model, tokenizer, device: str, hook_fn, layer_module):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_fn = hook_fn
        self.layer_module = layer_module
        self.config = {"chat_template": True}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_retries: int = 3,
        stop_strings: Optional[list] = None,
    ) -> str:
        del top_p, max_retries, stop_strings
        messages = [{"role": "user", "content": prompt}]
        import torch
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        handle = self.layer_module.register_forward_hook(self.hook_fn) if self.hook_fn else None
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if handle:
                handle.remove()
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()


def _compute_mean_iba(decisions: list[dict]) -> float:
    ibas = []
    for dec in decisions:
        if dec.get("skipped"):
            continue
        bet_amount = dec.get("bet_amount")
        if bet_amount is None:
            bet_amount = dec.get("bet")
        balance_before = dec.get("balance_before")
        if balance_before and bet_amount is not None and bet_amount > 0:
            ibas.append(float(bet_amount) / float(balance_before))
    return float(np.mean(ibas)) if ibas else 0.0


def _summarize_result(result: dict, condition: BehavioralCondition) -> dict:
    decisions = result.get("decisions", [])
    final_outcome = result.get("final_outcome", result.get("outcome", ""))
    final_outcome = str(final_outcome).lower()

    return {
        "stopped": bool(
            result.get("stopped_voluntarily", False)
            or result.get("outcome") == "voluntary_stop"
            or final_outcome == "voluntary_stop"
        ),
        "bk": bool(
            result.get("bankruptcy", False)
            or result.get("outcome") == "bankruptcy"
            or final_outcome == "bankrupt"
            or final_outcome == "bankruptcy"
        ),
        "terminal_wealth": int(result.get("final_balance", 0)),
        "mean_iba": _compute_mean_iba(decisions),
        "n_rounds": int(result.get("rounds_completed", result.get("total_rounds", 0))),
        "parse_failures": sum(1 for dec in decisions if dec.get("skipped")),
        "round_history": decisions,
        "behavioral_condition": asdict(condition),
    }


def _make_sm_generate_response(model, tokenizer, device, hook_fn, layer_module):
    def generate_response(self, prompt: str) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        handle = layer_module.register_forward_hook(hook_fn) if hook_fn else None
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    min_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            if handle:
                handle.remove()
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate_response


def _make_slot_machine_experiment(
    model_name: str,
    model,
    tokenizer,
    device: str,
    hook_fn,
    layer_module,
):
    from llama_gemma_experiment import RestartExperiment

    exp = RestartExperiment.__new__(RestartExperiment)
    exp.model_name = model_name
    exp.gpu_id = 0
    exp.device = device
    exp.model = model
    exp.tokenizer = tokenizer
    exp.max_rounds = 100
    exp.generate_response = types.MethodType(
        _make_sm_generate_response(model, tokenizer, device, hook_fn, layer_module),
        exp,
    )
    return exp


def _make_open_paradigm_experiment(
    exp_cls,
    model_name: str,
    condition: BehavioralCondition,
    model,
    tokenizer,
    device: str,
    hook_fn,
    layer_module,
):
    exp_cls = {
        "InvestmentChoiceExperiment": __import__(
            "investment_choice.run_experiment",
            fromlist=["InvestmentChoiceExperiment"],
        ).InvestmentChoiceExperiment,
        "MysteryWheelExperiment": __import__(
            "mystery_wheel.run_experiment",
            fromlist=["MysteryWheelExperiment"],
        ).MysteryWheelExperiment,
    }[exp_cls]
    exp = exp_cls.__new__(exp_cls)
    exp.model_name = model_name
    exp.gpu_id = 0
    exp.bet_type = condition.bet_type
    exp.bet_constraint = condition.bet_constraint
    exp.initial_balance = 100
    exp.max_rounds = 100
    exp.max_retries = 5
    exp.model_loader = HookedModelLoader(model, tokenizer, device, hook_fn, layer_module)
    return exp


def validate_behavioral_catalog(task: str, model_name: str) -> dict:
    catalog = get_behavioral_catalog(task, model_name)
    prompt_conditions = sorted({c.prompt_condition for c in catalog})
    bet_types = sorted({c.bet_type for c in catalog})
    bet_constraints = sorted({str(c.bet_constraint) for c in catalog if c.bet_constraint is not None})
    return {
        "task": task,
        "model_name": model_name,
        "n_games": len(catalog),
        "prompt_conditions": prompt_conditions,
        "bet_types": bet_types,
        "bet_constraints": bet_constraints,
    }


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def play_exact_behavioral_game(
    model,
    tokenizer,
    device: str,
    hook_fn,
    layer_module,
    model_name: str,
    task: str,
    game_index: int,
    seed: int,
) -> dict:
    """Replay one exact behavioral game profile under steering."""
    condition = select_behavioral_condition(task, model_name, game_index)
    _set_random_seed(seed)

    if task == "sm":
        exp = _make_slot_machine_experiment(
            model_name, model, tokenizer, device, hook_fn, layer_module
        )
        result = exp.play_game(condition.bet_type, condition.prompt_condition, rep=game_index)
    elif task == "ic":
        exp = _make_open_paradigm_experiment(
            "InvestmentChoiceExperiment",
            model_name,
            condition,
            model,
            tokenizer,
            device,
            hook_fn,
            layer_module,
        )
        result = exp.play_game(condition.prompt_condition, game_index + 1, seed)
    elif task == "mw":
        exp = _make_open_paradigm_experiment(
            "MysteryWheelExperiment",
            model_name,
            condition,
            model,
            tokenizer,
            device,
            hook_fn,
            layer_module,
        )
        result = exp.play_game(condition.prompt_condition, game_index + 1, seed)
    else:
        raise ValueError(f"Unknown task: {task}")

    return _summarize_result(result, condition)
