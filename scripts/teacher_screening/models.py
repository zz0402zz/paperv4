"""Candidate teacher implementations under one feature/target protocol."""

from __future__ import annotations

import gc
from importlib import metadata
import time

import numpy as np

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import models as tabpfn_models
from scripts.teacher_screening import config
from scripts.teacher_screening.data import PreparedFold


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def require_candidate(model: str) -> str:
    if model == "tabpfn":
        return tabpfn_models.require_tabpfn()
    package = {
        "xgboost": "xgboost",
        "catboost_joint": "catboost",
        "patch_transformer": "torch",
    }[model]
    version = package_version(package)
    if version == "not-installed":
        hint = (
            "uv pip install --python .venv-tabpfn\\Scripts\\python.exe catboost"
            if package == "catboost"
            else f"请在.venv-tabpfn环境安装{package}。"
        )
        raise SystemExit(f"缺少教师候选依赖 {package}；{hint}")
    return version


def candidate_identity(model: str) -> dict[str, object]:
    return {
        "model": model,
        "model_label": config.MODEL_LABELS[model],
        "package_version": require_candidate(model),
    }


def _release(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _tabpfn_predict(
    fold: PreparedFold, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    fit_x = np.asarray(fold.fit_flat, dtype=float)
    prediction_x = np.asarray(fold.prediction_flat, dtype=float)
    output = np.full(
        (len(prediction_x), *fold.fit_scaled_target.shape[1:]), np.nan, dtype=float
    )
    training_seconds = 0.0
    inference_seconds = 0.0
    fitted_outputs = 0
    for horizon_index in range(output.shape[1]):
        for target_index in range(output.shape[2]):
            fit_rows = (
                fold.fit_target_mask[:, horizon_index, target_index]
                & np.isfinite(fold.fit_scaled_target[:, horizon_index, target_index])
            )
            if int(fit_rows.sum()) < config.MIN_TRAIN_ROWS:
                raise ValueError(
                    "TabPFN教师训练样本不足: "
                    f"horizon_index={horizon_index}, target_index={target_index}, "
                    f"rows={int(fit_rows.sum())}"
                )
            regressor = tabpfn_models.make_teacher(seed)
            begin = time.perf_counter()
            regressor.fit(
                fit_x[fit_rows],
                fold.fit_scaled_target[fit_rows, horizon_index, target_index],
            )
            training_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            batches = []
            batch_size = 16
            for offset in range(0, len(prediction_x), batch_size):
                batches.append(
                    np.asarray(
                        regressor.predict(prediction_x[offset : offset + batch_size]),
                        dtype=float,
                    )
                )
            output[:, horizon_index, target_index] = np.concatenate(batches)
            inference_seconds += time.perf_counter() - begin
            fitted_outputs += 1
            _release(regressor)
    return output, {
        "training_seconds": np.asarray(training_seconds),
        "inference_seconds": np.asarray(inference_seconds),
        "fitted_models": np.asarray(fitted_outputs, dtype=np.int64),
    }


def _xgboost_predict(
    fold: PreparedFold, seed: int, device: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from xgboost import XGBRegressor

    output = np.full(
        (len(fold.prediction_flat), *fold.fit_scaled_target.shape[1:]),
        np.nan,
        dtype=float,
    )
    training_seconds = 0.0
    inference_seconds = 0.0
    fitted_outputs = 0
    for horizon_index in range(output.shape[1]):
        for target_index in range(output.shape[2]):
            fit_rows = (
                fold.fit_target_mask[:, horizon_index, target_index]
                & np.isfinite(fold.fit_scaled_target[:, horizon_index, target_index])
            )
            if int(fit_rows.sum()) < config.MIN_TRAIN_ROWS:
                raise ValueError(
                    "XGBoost教师训练样本不足: "
                    f"horizon_index={horizon_index}, target_index={target_index}, "
                    f"rows={int(fit_rows.sum())}"
                )
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=config.XGBOOST_ESTIMATORS,
                max_depth=config.XGBOOST_MAX_DEPTH,
                learning_rate=config.XGBOOST_LEARNING_RATE,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                device=device,
                n_jobs=-1,
                random_state=int(seed),
                verbosity=0,
            )
            begin = time.perf_counter()
            model.fit(
                fold.fit_flat[fit_rows],
                fold.fit_scaled_target[fit_rows, horizon_index, target_index],
            )
            training_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            output[:, horizon_index, target_index] = np.asarray(
                model.predict(fold.prediction_flat), dtype=float
            )
            inference_seconds += time.perf_counter() - begin
            fitted_outputs += 1
            _release(model)
    return output, {
        "training_seconds": np.asarray(training_seconds),
        "inference_seconds": np.asarray(inference_seconds),
        "fitted_models": np.asarray(fitted_outputs, dtype=np.int64),
    }


def _catboost_predict(
    fold: PreparedFold, seed: int, device: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from catboost import CatBoostRegressor

    output_shape = fold.fit_scaled_target.shape[1:]
    flat_target = fold.fit_scaled_target.reshape(len(fold.fit_flat), -1)
    flat_mask = fold.fit_target_mask.reshape(len(fold.fit_flat), -1)
    fit_rows = flat_mask.all(axis=1) & np.isfinite(flat_target).all(axis=1)
    if int(fit_rows.sum()) < config.MIN_TRAIN_ROWS:
        raise ValueError(
            "CatBoost联合教师的完整多输出训练行不足: "
            f"rows={int(fit_rows.sum())}。GPU MultiRMSE不能直接使用缺失标签。"
        )
    model = CatBoostRegressor(**catboost_parameters(seed, device))
    begin = time.perf_counter()
    model.fit(fold.fit_flat[fit_rows], flat_target[fit_rows])
    training_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    prediction = np.asarray(model.predict(fold.prediction_flat), dtype=float)
    inference_seconds = time.perf_counter() - begin
    _release(model)
    return prediction.reshape(len(prediction), *output_shape), {
        "training_seconds": np.asarray(training_seconds),
        "inference_seconds": np.asarray(inference_seconds),
        "fitted_models": np.asarray(1, dtype=np.int64),
        "complete_joint_rows": np.asarray(int(fit_rows.sum()), dtype=np.int64),
    }


def catboost_parameters(seed: int, device: str) -> dict[str, object]:
    parameters: dict[str, object] = dict(
        loss_function="MultiRMSE",
        # CatBoost may automatically choose Ordered boosting for small data,
        # while its CUDA MultiRMSE trainer only supports Plain boosting.
        boosting_type="Plain",
        iterations=config.CATBOOST_ITERATIONS,
        depth=config.CATBOOST_DEPTH,
        learning_rate=config.CATBOOST_LEARNING_RATE,
        random_seed=int(seed),
        task_type="GPU" if device == "cuda" else "CPU",
        verbose=False,
        allow_writing_files=False,
    )
    if device == "cuda":
        parameters["devices"] = "0"
    return parameters


def _patch_transformer_predict(
    fold: PreparedFold, seed: int, device_name: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Train a compact patch encoder control; this is not an official PatchTST port."""

    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 已指定，但PyTorch未检测到CUDA。")
    device = torch.device(device_name)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    output_shape = fold.fit_scaled_target.shape[1:]
    output_count = int(np.prod(output_shape))
    sequence_steps = fold.fit_sequence.shape[1]
    patch_count = sequence_steps - config.PATCH_KERNEL + 1

    class PatchTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch = torch.nn.Conv1d(
                fold.fit_sequence.shape[2],
                config.PATCH_DIMENSION,
                kernel_size=config.PATCH_KERNEL,
                stride=1,
            )
            self.context = torch.nn.Linear(
                fold.fit_context.shape[1], config.PATCH_DIMENSION
            )
            self.position = torch.nn.Parameter(
                torch.zeros(1, patch_count + 1, config.PATCH_DIMENSION)
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=config.PATCH_DIMENSION,
                nhead=config.PATCH_HEADS,
                dim_feedforward=config.PATCH_DIMENSION * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = torch.nn.TransformerEncoder(
                layer,
                num_layers=config.PATCH_LAYERS,
                enable_nested_tensor=False,
            )
            self.norm = torch.nn.LayerNorm(config.PATCH_DIMENSION)
            self.output = torch.nn.Linear(config.PATCH_DIMENSION, output_count)

        def forward(self, sequence, context):
            patches = self.patch(sequence.transpose(1, 2)).transpose(1, 2)
            context_token = self.context(context).unsqueeze(1)
            tokens = torch.cat((context_token, patches), dim=1) + self.position
            encoded = self.encoder(tokens)
            return self.output(self.norm(encoded[:, 0]))

    sequence = torch.as_tensor(fold.fit_sequence, dtype=torch.float32)
    context = torch.as_tensor(fold.fit_context, dtype=torch.float32)
    target = torch.as_tensor(
        fold.fit_scaled_target.reshape(len(sequence), -1), dtype=torch.float32
    )
    mask = torch.as_tensor(
        fold.fit_target_mask.reshape(len(sequence), -1), dtype=torch.bool
    )

    def masked_huber(prediction, truth, valid):
        valid = valid & torch.isfinite(truth) & torch.isfinite(prediction)
        absolute = torch.abs(prediction - truth)
        loss = torch.where(
            absolute <= 1.0,
            0.5 * absolute.square(),
            absolute - 0.5,
        )
        counts = valid.sum(dim=0)
        available = counts > 0
        per_output = torch.where(valid, loss, 0.0).sum(dim=0) / counts.clamp_min(1)
        return per_output[available].mean()

    def make_loader(indices: np.ndarray, shuffle: bool):
        dataset = torch.utils.data.TensorDataset(
            sequence[indices], context[indices], target[indices], mask[indices]
        )
        generator = torch.Generator().manual_seed(int(seed))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.PATCH_BATCH_SIZE,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    selection_boundary = max(config.MIN_TRAIN_ROWS, int(len(sequence) * 0.8))
    selection_boundary = min(selection_boundary, len(sequence) - 1)
    train_indices = np.arange(selection_boundary)
    validation_indices = np.arange(selection_boundary, len(sequence))
    if not len(validation_indices):
        raise ValueError("补丁Transformer没有可用的训练期内部验证行。")

    def train_epochs(network, loader, epochs: int, evaluate: bool):
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.PATCH_LEARNING_RATE
        )
        best_epoch = epochs
        best_loss = np.inf
        stale = 0
        for epoch in range(1, epochs + 1):
            network.train()
            for seq_batch, ctx_batch, y_batch, mask_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                predicted = network(seq_batch.to(device), ctx_batch.to(device))
                loss = masked_huber(
                    predicted, y_batch.to(device), mask_batch.to(device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
            if not evaluate or epoch % config.PATCH_EVALUATION_EVERY:
                continue
            network.eval()
            losses = []
            with torch.no_grad():
                for seq_batch, ctx_batch, y_batch, mask_batch in make_loader(
                    validation_indices, False
                ):
                    losses.append(
                        float(
                            masked_huber(
                                network(seq_batch.to(device), ctx_batch.to(device)),
                                y_batch.to(device),
                                mask_batch.to(device),
                            ).cpu()
                        )
                    )
            validation_loss = float(np.mean(losses))
            improved = np.isfinite(validation_loss) and validation_loss < best_loss - 1e-4
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            console.info(
                "patch_internal_validation",
                epoch=epoch,
                val_loss=validation_loss,
                best_epoch=best_epoch,
            )
            if epoch >= config.PATCH_MIN_EPOCHS and stale >= config.PATCH_PATIENCE:
                break
        return best_epoch, best_loss

    begin = time.perf_counter()
    selection_model = PatchTeacher().to(device)
    selected_epoch, best_loss = train_epochs(
        selection_model,
        make_loader(train_indices, True),
        config.PATCH_MAX_EPOCHS,
        True,
    )
    del selection_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(int(seed))
    final_model = PatchTeacher().to(device)
    train_epochs(
        final_model,
        make_loader(np.arange(len(sequence)), True),
        selected_epoch,
        False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - begin

    final_model.eval()
    begin = time.perf_counter()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(fold.prediction_sequence), config.PATCH_BATCH_SIZE):
            end = offset + config.PATCH_BATCH_SIZE
            predictions.append(
                final_model(
                    torch.as_tensor(
                        fold.prediction_sequence[offset:end],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        fold.prediction_context[offset:end],
                        dtype=torch.float32,
                        device=device,
                    ),
                ).cpu().numpy()
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - begin
    prediction = np.concatenate(predictions).reshape(-1, *output_shape)
    parameter_count = sum(parameter.numel() for parameter in final_model.parameters())
    _release(final_model)
    return prediction, {
        "training_seconds": np.asarray(training_seconds),
        "inference_seconds": np.asarray(inference_seconds),
        "fitted_models": np.asarray(1, dtype=np.int64),
        "selected_epoch": np.asarray(selected_epoch, dtype=np.int64),
        "best_internal_loss": np.asarray(best_loss, dtype=float),
        "parameter_count": np.asarray(parameter_count, dtype=np.int64),
    }


def fit_predict(
    model: str,
    fold: PreparedFold,
    *,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    require_candidate(model)
    if model == "tabpfn":
        prediction, diagnostics = _tabpfn_predict(fold, seed)
    elif model == "xgboost":
        prediction, diagnostics = _xgboost_predict(fold, seed, device)
    elif model == "catboost_joint":
        prediction, diagnostics = _catboost_predict(fold, seed, device)
    elif model == "patch_transformer":
        prediction, diagnostics = _patch_transformer_predict(fold, seed, device)
    else:
        raise ValueError(f"未知教师候选: {model}")
    return fold.to_absolute(prediction), diagnostics
