# -*- coding: utf-8 -*-
"""Behavior-clone a PPO MLP actor from the verified turn-expert dataset."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle
from dogfight.ai.rl_module_action_provider import _build_module_from_metadata


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument(
        "--dataset", default="artifacts/expert/habfm_turn_expert_v1.npz"
    )
    parser.add_argument(
        "--output", default="artifacts/models/codex/candidates/habfm_turn_bc_v1"
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--seed", type=int, default=260822)
    args = parser.parse_args()

    import torch
    from ray.rllib.core.columns import Columns

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source = _resolve(args.source_bundle)
    dataset_path = _resolve(args.dataset)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    payload = np.load(dataset_path)
    observations = np.asarray(payload["observations"], dtype=np.float32)
    actions = np.asarray(payload["actions"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 26:
        raise ValueError(f"expected observations [N,26], got {observations.shape}")
    if actions.shape != (len(observations), 4):
        raise ValueError(f"expected actions [N,4], got {actions.shape}")

    metadata, weights = load_lightweight_policy_bundle(source)
    if metadata.get("algorithm_class") != "PPO":
        raise ValueError("turn behavior cloning currently supports PPO bundles only")
    module = _build_module_from_metadata(metadata, weights)
    module.train()
    for name, parameter in module.named_parameters():
        parameter.requires_grad = not name.startswith("vf.")
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    indices = np.arange(len(observations))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    validation_size = max(1, int(round(len(indices) * args.validation_fraction)))
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    obs_tensor = torch.as_tensor(observations)
    action_tensor = torch.as_tensor(actions)

    def loss_for(batch_indices: np.ndarray) -> torch.Tensor:
        output_dict = module.forward_train({Columns.OBS: obs_tensor[batch_indices]})
        logits = output_dict[Columns.ACTION_DIST_INPUTS]
        action_mean = logits[..., :4]
        return torch.nn.functional.mse_loss(action_mean, action_tensor[batch_indices])

    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_indices)
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(train_indices), args.batch_size):
            batch_indices = train_indices[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(batch_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_indices)
            total_count += len(batch_indices)

        module.eval()
        with torch.no_grad():
            validation_loss = float(loss_for(validation_indices))
        module.train()
        train_loss = total_loss / max(total_count, 1)
        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": validation_loss,
        })
        if validation_loss < best_validation - 1.0e-7:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(module.get_state())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_mse={train_loss:.6f} "
                f"validation_mse={validation_loss:.6f}"
            )
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("behavior cloning did not produce a checkpoint")
    module.set_state(best_state)
    cloned_weights = module.get_state()

    cloned_metadata = copy.deepcopy(metadata)
    env_config = cloned_metadata.setdefault("algorithm_config", {}).setdefault(
        "env_config", {}
    )
    env_config["action_postprocess"] = {}
    bundle_metadata = cloned_metadata.setdefault("metadata", {})
    bundle_metadata["action_postprocess"] = {}
    bundle_metadata["behavior_cloning"] = {
        "source_bundle": str(source.resolve()),
        "dataset": str(dataset_path.resolve()),
        "samples": int(len(observations)),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "actor_encoder_trained": True,
        "value_head_frozen": True,
    }

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(cloned_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as file:
        pickle.dump(cloned_weights, file, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "bc_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"saved={output}")
    print(f"samples={len(observations)} best_epoch={best_epoch} best_val_mse={best_validation:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
