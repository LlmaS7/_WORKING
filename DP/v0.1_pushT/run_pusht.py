"""Windows entry point for the official low-dimensional PushT environment/U-Net.

Run with E:\\_WORKING\\.venv\\Scripts\\python.exe. This deliberately avoids the
upstream Linux/Hydra/Ray training stack. See README_LOCAL.md for commands.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import urllib.request
import zipfile

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import zarr

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/pusht/pusht_cchi_v7_replay.zarr"
DATA_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
OBS_STEPS, HORIZON, ACTION_STEPS, DIFFUSION_STEPS = 2, 16, 8, 100


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    return device


def make_env(seed=42, keypoint_map=None):
    # Match upstream evaluation's legacy random-reset convention. Recorded
    # body poses, in contrast, must be replayed with legacy=False.
    env = PushTKeypointsEnv(legacy=True, render_size=256, keypoint_visible_rate=1.0,
                            local_keypoint_map=keypoint_map)
    env.seed(seed)
    return env


def read_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path}\nRun run_pusht.py download first.")
    group = zarr.open_group(str(path), mode="r")
    state = np.asarray(group["data/state"][:], dtype=np.float32)
    keypoints = np.asarray(group["data/keypoint"][:], dtype=np.float32)
    action = np.asarray(group["data/action"][:], dtype=np.float32)
    ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
    obs = np.concatenate([keypoints.reshape(len(state), -1), state[:, :2]], axis=-1)
    if obs.shape != (len(action), 20) or action.shape[1:] != (2,):
        raise ValueError(f"Unexpected data shapes: obs={obs.shape}, action={action.shape}")
    if not len(ends) or ends[0] <= 0 or ends[-1] != len(action) or np.any(np.diff(ends) <= 0):
        raise ValueError("Invalid episode boundaries.")
    if not np.isfinite(obs).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite observation/action in dataset.")
    return obs, action, state, ends


def dataset_keypoint_map(obs, state):
    """Recover the dataset's local keypoint definition, independent of rendering.

    Upstream derives local points from a rasterized object. Pymunk/SDL version
    changes can shift that raster, so reconstruct the rigid transform instead.
    """
    angles = state[:, 4].astype(np.float64)
    cos, sin = np.cos(angles), np.sin(angles)
    rotations = np.stack([np.stack([cos, -sin], axis=1), np.stack([sin, cos], axis=1)], axis=1)
    world_offsets = obs[:, :18].reshape(-1, 9, 2) - state[:, None, 2:4]
    local_points = np.einsum("nki,nij->nkj", world_offsets, rotations)
    # Contact frames in the published data contain small pose/keypoint offsets.
    # Use a robust estimate rather than allowing an outlier to define the map.
    local = np.median(local_points, axis=0)
    frame_errors = np.max(np.abs(local_points - local), axis=(1, 2))
    if np.quantile(frame_errors, 0.95) > 0.25:
        raise ValueError("Dataset keypoints do not match a consistent rigid object.")
    return {"block": local, "agent": np.zeros((3, 2))}


def split_episodes(ends, seed):
    if len(ends) < 2:
        raise ValueError("At least two episodes are required for train/validation split.")
    ids = np.random.default_rng(seed).permutation(len(ends))
    n_val = max(1, round(len(ends) * 0.1))
    return np.sort(ids[n_val:]), np.sort(ids[:n_val])


def fit_stats(obs, action, ends, train_ids):
    starts = np.r_[0, ends[:-1]]
    indices = np.concatenate([np.arange(starts[i], ends[i]) for i in train_ids])
    result = {}
    for key, values in (("obs", obs), ("action", action)):
        selected = values[indices]
        result[key + "_min"] = torch.from_numpy(selected.min(axis=0).copy())
        result[key + "_range"] = torch.from_numpy(np.maximum(np.ptp(selected, axis=0), 1e-6))
    return result


def normalize(x, key, stats):
    return 2 * (x - stats[key + "_min"].to(x.device)) / stats[key + "_range"].to(x.device) - 1


def unnormalize(x, key, stats):
    return (x + 1) / 2 * stats[key + "_range"].to(x.device) + stats[key + "_min"].to(x.device)


class PushTSequences(Dataset):
    """Episode-local windows with endpoint padding; observation t predicts action t."""
    def __init__(self, obs, action, ends, episode_ids):
        self.obs = torch.from_numpy(obs)
        self.action = torch.from_numpy(action)
        starts = np.r_[0, ends[:-1]]
        self.windows = []
        for i in episode_ids:
            start, end = int(starts[i]), int(ends[i])
            # Same padding limits as the upstream task: before=1, after=7.
            for offset in range(-(OBS_STEPS - 1), end - start - HORIZON + ACTION_STEPS):
                self.windows.append((start, end, start + offset))
        if not self.windows:
            raise ValueError("No sequence windows; check episode lengths.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        start, end, offset = self.windows[index]
        positions = torch.arange(offset, offset + HORIZON).clamp(start, end - 1)
        return self.obs[positions[:OBS_STEPS]], self.action[positions]


def make_model(widths):
    return ConditionalUnet1D(input_dim=2, global_cond_dim=20 * OBS_STEPS,
                             diffusion_step_embed_dim=128, down_dims=widths,
                             kernel_size=5, n_groups=8, cond_predict_scale=True)


def make_scheduler():
    return DDPMScheduler(num_train_timesteps=DIFFUSION_STEPS,
                         beta_schedule="squaredcos_cap_v2", clip_sample=True,
                         prediction_type="epsilon", variance_type="fixed_small")


def noise_loss(model, batch, scheduler, stats, device):
    obs, actions = (x.to(device) for x in batch)
    actions = normalize(actions, "action", stats)
    cond = normalize(obs, "obs", stats).flatten(start_dim=1)
    noise = torch.randn_like(actions)
    t = torch.randint(0, DIFFUSION_STEPS, (len(actions),), device=device)
    noisy_actions = scheduler.add_noise(actions, noise, t)
    return nn.functional.mse_loss(model(noisy_actions, t, global_cond=cond), noise)


@torch.inference_mode()
def predict(model, obs, scheduler, stats, device, sampling_steps):
    cond = normalize(torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)[None], "obs", stats)
    sample = torch.randn((1, HORIZON, 2), device=device)
    scheduler.set_timesteps(sampling_steps, device=device)
    for t in scheduler.timesteps:
        predicted_noise = model(sample, t, global_cond=cond.flatten(start_dim=1))
        sample = scheduler.step(predicted_noise, t, sample).prev_sample
    actions = unnormalize(sample, "action", stats)[0]
    # The last observed state is at index OBS_STEPS-1, matching upstream.
    return actions[OBS_STEPS - 1: OBS_STEPS - 1 + ACTION_STEPS].clamp(0, 512).cpu().numpy()


def download(args):
    if args.data.exists():
        print(f"Dataset already exists: {args.data}")
        return
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    archive = folder / "pusht.zip"
    partial = folder / "pusht.zip.part"
    print(f"Downloading official demonstrations: {DATA_URL}", flush=True)
    if not archive.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                received += len(chunk)
                if received % (16 * 1024 * 1024) == 0:
                    print(f"  {received / 2**20:.0f} / {total / 2**20:.0f} MiB", flush=True)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        destination = folder.resolve()
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe archive member.")
        zipped.extractall(folder)
    if not DATA.exists():
        raise FileNotFoundError(f"Archive extracted but expected data missing: {DATA}")
    print(f"Ready: {DATA}")


def inspect_data(args):
    obs, action, state, ends = read_data(args.data)
    train_ids, val_ids = split_episodes(ends, args.seed)
    ds = PushTSequences(obs, action, ends, train_ids)
    batch = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    report = {"episodes": len(ends), "frames": len(obs), "train_episodes": len(train_ids),
              "val_episodes": len(val_ids), "obs_shape": list(obs.shape),
              "action_shape": list(action.shape), "batch_obs": list(batch[0].shape),
              "batch_action": list(batch[1].shape), "action_min": action.min(0).tolist(),
              "action_max": action.max(0).tolist()}
    print(json.dumps(report, indent=2), flush=True)
    out = ROOT / "outputs/inspect"
    out.mkdir(parents=True, exist_ok=True)
    # Render recorded states, rather than claiming this is a learned rollout.
    keypoint_map = dataset_keypoint_map(obs, state)
    env = make_env(args.seed, keypoint_map)
    env.legacy = False
    first_end = int(ends[0])
    with imageio.get_writer(out / "demonstration.mp4", fps=10) as writer:
        for i in range(first_end):
            env.reset_to_state = state[i].astype(float)
            env_obs = env.reset()
            if i == 0:
                np.testing.assert_allclose(env_obs[:20], obs[i], atol=0.05)
            env.latest_action = action[i]
            writer.append_data(env.render("rgb_array"))
            if i == 0:
                imageio.imwrite(out / "first_frame.png", env.render("rgb_array"))
    env.close()
    (out / "data_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Demonstration video: {out / 'demonstration.mp4'}")


def train(args):
    device = get_device(args.device)
    obs, action, state, ends = read_data(args.data)
    keypoint_map = dataset_keypoint_map(obs, state)
    train_ids, val_ids = split_episodes(ends, args.seed)
    stats = fit_stats(obs, action, ends, train_ids)
    train_ds = PushTSequences(obs, action, ends, train_ids)
    val_ds = PushTSequences(obs, action, ends, val_ids)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    widths = [32, 64, 128] if args.small else [128, 256, 512]
    model = make_model(widths).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    scheduler = make_scheduler()
    total_steps = args.epochs * len(loader)
    if args.max_train_steps:
        total_steps = min(total_steps, args.max_train_steps)
    warmup = min(500, max(1, total_steps // 10))
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    out = args.output or ROOT / "outputs" / datetime.now().strftime("train_%Y%m%d_%H%M%S")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {"widths": widths, "obs_dim": 20, "obs_steps": OBS_STEPS, "horizon": HORIZON,
              "action_steps": ACTION_STEPS, "diffusion_steps": DIFFUSION_STEPS, "seed": args.seed,
              "epochs": args.epochs, "batch_size": args.batch_size, "device": str(device),
              "max_train_steps": args.max_train_steps, "data": str(args.data.resolve()),
              "train_episode_ids": train_ids.tolist(), "val_episode_ids": val_ids.tolist()}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Output: {out}", flush=True)
    step, best = 0, float("inf")
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = noise_loss(model, batch, scheduler, stats, device)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                step += 1
                decay = min(0.9999, 1 - (1 + step) ** -0.75)
                with torch.no_grad():
                    for averaged, current in zip(ema.parameters(), model.parameters()):
                        averaged.lerp_(current, 1 - decay)
                losses.append(float(loss.detach()))
                if step == 1 or step % 100 == 0:
                    print(f"epoch={epoch + 1} step={step} loss={losses[-1]:.5f}", flush=True)
                if step >= total_steps:
                    break
            # A fixed validation noise stream makes epochs more comparable.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(args.seed + 10000)
                with torch.inference_mode():
                    val_losses = [float(noise_loss(ema, b, scheduler, stats, device)) for b in val_loader]
            metrics = {"epoch": epoch + 1, "step": step,
                       "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses))}
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            payload = {"config": config, "model": ema.state_dict(), "stats": stats, "metrics": metrics,
                       "keypoint_map": {k: torch.as_tensor(v) for k, v in keypoint_map.items()}}
            torch.save(payload, out / "latest.pt")
            if metrics["val_loss"] < best:
                best = metrics["val_loss"]
                torch.save(payload, out / "best.pt")
            print(json.dumps(metrics), flush=True)
            if step >= total_steps:
                break
    print(f"Checkpoint: {out / 'best.pt'}", flush=True)


def evaluate(args):
    if args.checkpoint is None:
        raise ValueError("eval requires --checkpoint PATH")
    device = get_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_model(payload["config"]["widths"]).to(device).eval()
    model.load_state_dict(payload["model"])
    out = args.output or ROOT / "outputs" / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scheduler = make_scheduler()
    # Warm up outside the measured region, especially for CUDA's first invocation.
    predict(model, np.zeros((OBS_STEPS, 20), dtype=np.float32), scheduler,
            payload["stats"], device, args.sampling_steps)
    rows = []
    env = make_env(args.seed, {k: v.numpy() for k, v in payload["keypoint_map"].items()})
    try:
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            obs = env.reset()[:20]
            history = deque([obs.copy()] * OBS_STEPS, maxlen=OBS_STEPS)
            score, success, count, latencies = 0.0, False, 0, []
            with imageio.get_writer(out / f"rollout_{episode:02d}.mp4", fps=10) as writer:
                writer.append_data(env.render("rgb_array"))
                while count < args.max_steps and not success:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    actions = predict(model, history, scheduler, payload["stats"], device, args.sampling_steps)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    latencies.append((time.perf_counter() - start) * 1000)
                    for action in actions:
                        obs, reward, done, _ = env.step(action)
                        history.append(obs[:20].copy())
                        score = max(score, float(reward))
                        success = bool(done)
                        count += 1
                        writer.append_data(env.render("rgb_array"))
                        if success or count >= args.max_steps:
                            break
            row = {"seed": args.seed + episode, "max_score": score, "success": success,
                   "steps": count, "mean_action_chunk_ms": float(np.mean(latencies))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    summary = {"checkpoint": str(args.checkpoint.resolve()), "device": str(device),
               "sampling_steps": args.sampling_steps,
               "mean_score": float(np.mean([r["max_score"] for r in rows])),
               "success_rate": float(np.mean([r["success"] for r in rows])),
               "success_definition": "coverage > 0.95 at any step",
               "episodes": rows}
    (out / "eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Evaluation: {out / 'eval.json'}")


def demo(args):
    import pygame
    env = make_env(args.seed)
    env.reset()
    try:
        while True:
            env.render("human")
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.seed(env._seed + 1)
                    env.reset()
            # Mouse coordinates are in the original 512x512 window.
            _, _, done, _ = env.step(np.asarray(pygame.mouse.get_pos(), dtype=float))
            if done:
                pygame.display.set_caption("Success! Press R to reset; Esc to quit")
            env.clock.tick(10)
    finally:
        env.close()


def smoke(args):
    device = get_device(args.device)
    out = ROOT / "outputs/smoke"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.seed)
    try:
        obs = env.reset()
        assert obs.shape == (40,), obs.shape  # 20 coordinates + 20 visibility flags
        for _ in range(10):
            obs, reward, done, _ = env.step(np.asarray(env.agent.position) + [1.0, -1.0])
        frame = env.render("rgb_array")
        assert frame.shape == (256, 256, 3), frame.shape
        imageio.imwrite(out / "environment.png", frame)
        model = make_model([32, 64, 128]).to(device)
        stats = {"obs_min": torch.zeros(20), "obs_range": torch.full((20,), 512.),
                 "action_min": torch.zeros(2), "action_range": torch.full((2,), 512.)}
        batch = (torch.rand(2, OBS_STEPS, 20) * 512, torch.rand(2, HORIZON, 2) * 512)
        optimizer = torch.optim.AdamW(model.parameters())
        loss = noise_loss(model, batch, make_scheduler(), stats, device)
        loss.backward()
        optimizer.step()
        model.eval()
        actions = predict(model, [obs[:20], obs[:20]], make_scheduler(), stats, device, 2)
        assert actions.shape == (ACTION_STEPS, 2) and np.isfinite(actions).all()
        for action in actions:
            env.step(action)
        report = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                  "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "policy_obs_shape": [20], "raw_env_obs_shape": list(obs.shape),
                  "action_chunk_shape": list(actions.shape), "training_loss": float(loss.detach()),
                  "status": "passed", "note": "Synthetic-batch check, not a trained successful policy."}
        (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "inspect", "demo", "smoke", "train", "eval"])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--small", action="store_true", help="Small network for quick pipeline checks")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sampling-steps", type=int, default=100)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.episodes, args.max_steps) < 1:
        parser.error("epochs, batch-size, episodes and max-steps must be positive")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("max-train-steps must be positive")
    if not 1 <= args.sampling_steps <= DIFFUSION_STEPS:
        parser.error("sampling-steps must be between 1 and 100")
    if args.command == "download" and args.data != DATA:
        parser.error("download uses the default data location; omit --data")
    # Avoid excessive CPU thread overhead for this small temporal network.
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all(args.seed)
    handlers = {"download": download, "inspect": inspect_data, "demo": demo,
                "smoke": smoke, "train": train, "eval": evaluate}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
