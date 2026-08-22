#!/usr/bin/env python3
"""Train 18-output GRU students with or without causal TabPFN distillation."""

from __future__ import annotations

import argparse
from importlib import metadata as package_metadata
import random

import numpy as np

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io, models
from scripts.tabpfn_distillation.teacher import select_tasks


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in _parse_csv(value))
    if not seeds:
        raise ValueError("至少需要一个随机种子。")
    unknown = set(seeds).difference(config.STUDENT_SEEDS)
    if unknown:
        raise ValueError(f"随机种子不在冻结协议中: {sorted(unknown)}")
    return seeds


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_inputs(
    train: dict[str, np.ndarray], evaluation: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit all input scaling on training history only."""

    def continuous(split: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate((split["x_raw"], split["x_diff"]), axis=-1)

    def masks(split: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate((split["x_raw_mask"], split["x_diff_mask"]), axis=-1).astype(float)

    sequence_scaler = models.FeatureScaler.fit(continuous(train))
    current_scaler = models.FeatureScaler.fit(np.asarray(train["current"], dtype=float))

    def sequence(split: dict[str, np.ndarray]) -> np.ndarray:
        scaled = sequence_scaler.transform(continuous(split))
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate((scaled, masks(split)), axis=-1).astype(np.float32)

    def current(split: dict[str, np.ndarray]) -> np.ndarray:
        scaled = current_scaler.transform(split["current"])
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate(
            (scaled, np.asarray(split["current_mask"], dtype=bool).astype(float)), axis=-1
        ).astype(np.float32)

    return sequence(train), current(train), sequence(evaluation), current(evaluation)


def teacher_targets(
    train: dict[str, np.ndarray], station: str, target: str, mode: str
) -> tuple[np.ndarray, np.ndarray, str]:
    path = io.teacher_cache_path("训练OOF", station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少因果OOF教师缓存: {path}")
    arrays, metadata = io.load_archive(path)
    if metadata.get("experiment") != config.EXPERIMENT_ID:
        raise RuntimeError(f"教师缓存实验身份不一致: {path}")
    if metadata.get("kind") != "causal_oof":
        raise RuntimeError(f"不是训练OOF教师缓存: {path}")
    if metadata.get("station") != station or metadata.get("target") != target:
        raise RuntimeError(f"教师缓存任务身份不一致: {path}")
    if not np.array_equal(
        arrays.get("target_start"), np.asarray(train["target_start"], dtype="datetime64[ns]")
    ):
        raise RuntimeError(f"教师缓存时间轴与学生训练集不一致: {path}")
    completed = np.asarray(arrays.get("completed"), dtype=bool)
    if completed.shape != (len(config.OOF_FOLDS), config.OUTPUT_STEPS) or not completed.all():
        raise RuntimeError(f"教师OOF缓存尚未完成全部18个时距: {path}")
    teacher_delta = np.asarray(arrays["pred_delta"], dtype=float)
    teacher_mask = np.asarray(arrays["pred_mask"], dtype=bool) & np.isfinite(teacher_delta)
    if mode == "delta":
        values = teacher_delta
    elif mode == "absolute":
        values = data.to_absolute(teacher_delta, train["current"], "delta")
    else:
        raise ValueError(f"Unknown target mode: {mode}")
    return values, teacher_mask & np.isfinite(values), io.file_sha256(path)


def masked_mse(torch, prediction, target, mask):
    """Differentiable masked mean that is safe for an empty mini-batch mask."""
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    if bool(valid.any()):
        return torch.square(prediction[valid] - target[valid]).mean()
    return prediction.sum() * 0.0


def _student_metadata(
    variant: str,
    seed: int,
    station: str,
    target: str,
    teacher_sha256: str | None,
) -> dict[str, object]:
    return {
        "experiment": config.EXPERIMENT_ID,
        "kind": "student_validation_prediction",
        "variant": variant,
        "target_mode": config.student_target_mode(variant),
        "distilled": config.is_distilled(variant),
        "seed": int(seed),
        "station": station,
        "target": target,
        "input_steps": config.INPUT_STEPS,
        "horizon_hours": list(config.HORIZON_HOURS),
        "gru_hidden_size": config.GRU_HIDDEN_SIZE,
        "gru_current_hidden_size": config.GRU_CURRENT_HIDDEN_SIZE,
        "gru_epochs": config.GRU_EPOCHS,
        "gru_learning_rate": config.GRU_LEARNING_RATE,
        "distillation_weight": config.DISTILLATION_WEIGHT if config.is_distilled(variant) else 0.0,
        "teacher_cache_sha256": teacher_sha256,
        "torch_version": package_metadata.version("torch"),
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": io.code_sha256(
            ("config.py", "data.py", "io.py", "models.py", "student.py")
        ),
    }


def train_student(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    variant: str,
    seed: int,
    station: str,
    target: str,
) -> np.ndarray:
    import torch

    mode = config.student_target_mode(variant)
    distilled = config.is_distilled(variant)
    true_values = data.target_values(train, mode)
    true_mask = np.asarray(train["y_mask"], dtype=bool) & np.isfinite(true_values)
    target_scaler = models.MaskedScaler.fit(true_values, true_mask)
    true_scaled = np.nan_to_num(
        target_scaler.transform(true_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    if distilled:
        teacher_values, teacher_mask, _ = teacher_targets(train, station, target, mode)
        teacher_scaled = np.nan_to_num(
            target_scaler.transform(teacher_values), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
    else:
        teacher_scaled = np.zeros_like(true_scaled, dtype=np.float32)
        teacher_mask = np.zeros_like(true_mask, dtype=bool)

    usable_rows = true_mask.any(axis=1) | teacher_mask.any(axis=1)
    if not usable_rows.any():
        raise ValueError(f"学生没有可用训练标签: {station}/{target}/{variant}")

    train_sequence, train_current, val_sequence, val_current = prepare_inputs(train, validation)
    _set_seed(torch, seed)

    class LongHorizonStudent(torch.nn.Module):
        def __init__(self, sequence_dim: int) -> None:
            super().__init__()
            self.sequence_encoder = torch.nn.GRU(
                input_size=sequence_dim,
                hidden_size=config.GRU_HIDDEN_SIZE,
                batch_first=True,
            )
            self.current_encoder = torch.nn.Sequential(
                torch.nn.Linear(2, config.GRU_CURRENT_HIDDEN_SIZE),
                torch.nn.ReLU(),
                torch.nn.Linear(config.GRU_CURRENT_HIDDEN_SIZE, config.GRU_HIDDEN_SIZE),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(config.GRU_HIDDEN_SIZE * 2, config.OUTPUT_STEPS)

        def forward(self, sequence_x, current_x):
            encoded, _ = self.sequence_encoder(sequence_x)
            current_state = self.current_encoder(current_x)
            return self.head(torch.cat((encoded[:, -1, :], current_state), dim=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongHorizonStudent(train_sequence.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GRU_LEARNING_RATE)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_sequence[usable_rows]),
        torch.as_tensor(train_current[usable_rows]),
        torch.as_tensor(true_scaled[usable_rows]),
        torch.as_tensor(true_mask[usable_rows]),
        torch.as_tensor(teacher_scaled[usable_rows]),
        torch.as_tensor(teacher_mask[usable_rows]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.GRU_BATCH_SIZE, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    model.train()
    for _ in range(config.GRU_EPOCHS):
        for sequence_x, current_x, truth_y, truth_mask, teacher_y, teacher_mask_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(sequence_x.to(device), current_x.to(device))
            truth_loss = masked_mse(
                torch, predicted, truth_y.to(device), truth_mask.to(device)
            )
            teacher_loss = masked_mse(
                torch, predicted, teacher_y.to(device), teacher_mask_batch.to(device)
            )
            loss = truth_loss + (
                config.DISTILLATION_WEIGHT * teacher_loss if distilled else 0.0
            )
            loss.backward()
            optimizer.step()

    model.eval()
    batches = []
    with torch.no_grad():
        for begin in range(0, len(val_sequence), config.GRU_BATCH_SIZE):
            end = begin + config.GRU_BATCH_SIZE
            batches.append(
                model(
                    torch.as_tensor(val_sequence[begin:end], device=device),
                    torch.as_tensor(val_current[begin:end], device=device),
                )
                .cpu()
                .numpy()
            )
    predicted_scaled = (
        np.concatenate(batches, axis=0)
        if batches
        else np.empty((0, config.OUTPUT_STEPS), dtype=float)
    )
    predicted_target = target_scaler.inverse_transform(predicted_scaled)
    return data.to_absolute(predicted_target, validation["current"], mode)


def run_task(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    variant: str,
    seed: int,
    station: str,
    target: str,
    force: bool,
) -> None:
    teacher_sha256 = None
    if config.is_distilled(variant):
        _, _, teacher_sha256 = teacher_targets(
            train, station, target, config.student_target_mode(variant)
        )
    expected = _student_metadata(variant, seed, station, target, teacher_sha256)
    path = io.student_prediction_path("val", variant, seed, station, target)
    existing = None if force else io.load_exact(path, expected)
    if existing is not None:
        if existing.get("pred", np.empty(0)).shape != (
            len(validation["target_start"]),
            config.OUTPUT_STEPS,
        ):
            raise RuntimeError(f"已有学生预测形状不正确: {path}")
        console.info("resume", variant=variant, seed=seed, status="already complete")
        return
    prediction = train_student(
        train,
        validation,
        variant=variant,
        seed=seed,
        station=station,
        target=target,
    )
    io.save_archive(path, data.prediction_arrays(validation, prediction), expected)
    console.info("saved", variant=variant, seed=seed, rows=len(prediction))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", *config.STUDENT_KEYS), default="all")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets")
    target_group.add_argument("--all-targets", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, config.STUDENT_SEEDS)))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = data.load_v2_panel()
    try:
        stations, targets = select_tasks(
            panel, args.stations, args.targets, args.all_stations, args.all_targets
        )
        seeds = _parse_seeds(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    variants = config.STUDENT_KEYS if args.variant == "all" else (args.variant,)
    total = len(stations) * len(targets)
    current = 0
    for station in stations:
        for target in targets:
            current += 1
            console.phase(f"{station} / {target}", current=current, total=total)
            splits = data.split_by_time(data.build_station_target_dataset(panel, station, target))
            for variant in variants:
                for seed in seeds:
                    run_task(
                        splits["train"],
                        splits["val"],
                        variant=variant,
                        seed=seed,
                        station=station,
                        target=target,
                        force=args.force,
                    )
    console.done(config.output_dir_for_split("val"))


if __name__ == "__main__":
    main()
