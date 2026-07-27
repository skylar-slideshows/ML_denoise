############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks

# Skylar Denno
# July 26, 2026

# 4_train.py

# Train the speech denoising neural network using the prepared dataset.

############################################################

# USAGE:
#    python src/4_train.py

############################################################


import os
import argparse
import itertools
import importlib
import numpy as np
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import istft as scipy_istft
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

try:
    pesq_module = importlib.import_module("pesq")
    pesq_fn = getattr(pesq_module, "pesq", None)
    HAS_PESQ = pesq_fn is not None
except Exception:
    pesq_fn = None
    HAS_PESQ = False

try:
    pystoi_module = importlib.import_module("pystoi")
    stoi_fn = getattr(pystoi_module, "stoi", None)
    HAS_STOI = stoi_fn is not None
except Exception:
    stoi_fn = None
    HAS_STOI = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "4_train.yaml")

# init/utilty functions:

def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

# test mode overrides to make it run really fast (only if test_mode enabled)
def apply_test_overrides(cfg):
    overrides = cfg.get("test_overrides", {})
    merged = dict(cfg)
    merged.update(overrides)
    return merged

def read_split_list(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r") as fh:
        names = [line.strip() for line in fh if line.strip()]
    if len(names) == 0:
        raise ValueError(f"Split file is empty: {path}")
    return names

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM speech denoiser")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable fast test mode with lightweight settings",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning defined in config",
    )
    return parser.parse_args()


############################################################

# CLASS: For loading sequences from the cached STFT NPZ files (/data/encode/cache)
class SpectrogramDataset(Dataset):
    
    def __init__(self, cache_dir, file_names, max_frames=None, random_crop=False):
        self.cache_dir = cache_dir
        self.file_names = file_names
        self.max_frames = max_frames
        self.random_crop = random_crop

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        path = os.path.join(self.cache_dir, name)

        # failsafes
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cached feature file not found: {path}")

        data = np.load(path)
        if "noisy_stft" not in data or "clean_stft" not in data:
            raise KeyError(f"Expected keys clean_stft and noisy_stft in {path}")

        noisy_stft = data["noisy_stft"]
        clean_stft = data["clean_stft"]

        # model takes [T, F]: log-magnitude spectrums
        noisy_mag = np.log1p(np.abs(noisy_stft)).T.astype(np.float32)
        clean_mag = np.log1p(np.abs(clean_stft)).T.astype(np.float32)

        if self.max_frames is not None and noisy_mag.shape[0] > self.max_frames:
            if self.random_crop:
                start = np.random.randint(0, noisy_mag.shape[0] - self.max_frames + 1)
            else:
                start = 0
            end = start + self.max_frames
            noisy_mag = noisy_mag[start:end]
            clean_mag = clean_mag[start:end]
            noisy_stft = noisy_stft[:, start:end]
            clean_stft = clean_stft[:, start:end]

        return {
            "x": torch.from_numpy(noisy_mag),
            "y": torch.from_numpy(clean_mag),
            "length": noisy_mag.shape[0],
            "noisy_phase": np.angle(noisy_stft).T.astype(np.float32),
            "clean_stft": clean_stft.T.astype(np.complex64),
        }

############################################################

# CLASS: for instantiating the LSTMs 
class LSTMDenoiser(nn.Module):
    def __init__(self, n_freq, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_freq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, n_freq),
            nn.Softplus(),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.head(out)


############################################################


# Outer functions

# batched tensor - need to pad STFT sequences to the same length.
def collate_batch(batch):
    xs = [item["x"] for item in batch]
    ys = [item["y"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    noisy_phase = [item["noisy_phase"] for item in batch]
    clean_stft = [item["clean_stft"] for item in batch]

    x_padded = pad_sequence(xs, batch_first=True)
    y_padded = pad_sequence(ys, batch_first=True)

    return {
        "x": x_padded,
        "y": y_padded,
        "lengths": lengths,
        "noisy_phase": noisy_phase,
        "clean_stft": clean_stft,
    }


# compute loss only on valid frames and ignore right-padding
def masked_mse(pred, target, lengths):
    batch_size, max_t, _ = pred.shape
    time_ids = torch.arange(max_t, device=pred.device).unsqueeze(0).expand(batch_size, -1)
    mask = (time_ids < lengths.unsqueeze(1)).unsqueeze(2).float()
    sq = (pred - target) ** 2
    return (sq * mask).sum() / mask.sum().clamp(min=1.0)


def make_valid_mask(lengths, max_t, n_freq):
    batch_size = lengths.shape[0]
    time_ids = torch.arange(max_t, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    valid_frames = time_ids < lengths.unsqueeze(1)
    return valid_frames.unsqueeze(2).expand(batch_size, max_t, n_freq)


# reporting performance while training
def compute_binary_roc_auc(labels, scores):
    if labels.size == 0:
        return float("nan")

    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]

    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tpr = np.concatenate(([0.0], tps / positives, [1.0]))
    fpr = np.concatenate(([0.0], fps / negatives, [1.0]))

    return float(np.trapz(tpr, fpr))


def stft_to_waveform(stft_ft, stft_params):
    _, wav = scipy_istft(
        stft_ft,
        fs=stft_params["target_sr"],
        window=stft_params["win"],
        nperseg=stft_params["n_fft"],
        noverlap=stft_params["n_fft"] - stft_params["n_hop"],
        nfft=stft_params["n_fft"],
        input_onesided=True,
        boundary=True,
    )
    return wav.astype(np.float32)


def compute_si_sdr(reference, estimate, eps=1e-8):
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)

    ref = ref - ref.mean()
    est = est - est.mean()

    ref_energy = np.sum(ref * ref) + eps
    projection = (np.sum(est * ref) / ref_energy) * ref
    noise = est - projection

    ratio = (np.sum(projection * projection) + eps) / (np.sum(noise * noise) + eps)
    return float(10.0 * np.log10(ratio))


def average_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def format_metrics(metrics):
    def _fmt(value):
        if value is None or np.isnan(value):
            return "nan"
        return f"{value:.6f}"

    return (
        f"loss={_fmt(metrics.get('loss'))}, "
        f"mse={_fmt(metrics.get('mse'))}, "
        f"r2={_fmt(metrics.get('r2'))}, "
        f"acc={_fmt(metrics.get('accuracy'))}, "
        f"f1={_fmt(metrics.get('f1'))}, "
        f"roc_auc={_fmt(metrics.get('roc_auc'))}, "
        f"pesq={_fmt(metrics.get('pesq'))}, "
        f"stoi={_fmt(metrics.get('stoi'))}, "
        f"si_sdr={_fmt(metrics.get('si_sdr'))}"
    )


# instantiate data loaders
def make_loader(cache_dir, split_path, batch_size, shuffle, num_workers, max_files=None, max_frames=None):
    names = read_split_list(split_path)
    if max_files is not None:
        names = names[:max_files]
    dataset = SpectrogramDataset(
        cache_dir,
        names,
        max_frames=max_frames,
        random_crop=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

# infer the number of frequency bins from the first sample in the split
def infer_n_freq(cache_dir, split_path):
    first_name = read_split_list(split_path)[0]
    path = os.path.join(cache_dir, first_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cached feature file not found: {path}")
    sample = np.load(path)
    if "noisy_stft" not in sample:
        raise KeyError(f"Expected noisy_stft key in {path}")
    return int(sample["noisy_stft"].shape[0])


############################################################


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip=None,
    metric_threshold=0.1,
    metric_max_points=250000,
    compute_speech_metrics=False,
    speech_metrics_max_utterances=0,
    sample_rate=16000,
    pesq_mode="wb",
    stft_params=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running = 0.0
    n_batches = 0

    sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n_points = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    roc_scores = []
    roc_labels = []

    pesq_scores = []
    stoi_scores = []
    si_sdr_scores = []
    speech_eval_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)

            pred = model(x, lengths)
            loss = masked_mse(pred, y, lengths)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            with torch.no_grad():
                pred_eval = pred.detach()
                y_eval = y.detach()
                valid_mask = make_valid_mask(lengths, pred_eval.shape[1], pred_eval.shape[2])

                valid_pred = pred_eval[valid_mask]
                valid_target = y_eval[valid_mask]
                diff = valid_pred - valid_target

                sse += float((diff ** 2).sum().item())
                sum_y += float(valid_target.sum().item())
                sum_y2 += float((valid_target ** 2).sum().item())
                n_points += int(valid_target.numel())

                target_bin = valid_target > metric_threshold
                pred_bin = valid_pred > metric_threshold

                tp += int((pred_bin & target_bin).sum().item())
                tn += int((~pred_bin & ~target_bin).sum().item())
                fp += int((pred_bin & ~target_bin).sum().item())
                fn += int((~pred_bin & target_bin).sum().item())

                remaining = metric_max_points - len(roc_scores)
                if remaining > 0:
                    score_np = valid_pred.detach().cpu().numpy()
                    label_np = target_bin.detach().cpu().numpy().astype(np.int32)

                    if score_np.size > remaining:
                        choose = np.random.choice(score_np.size, size=remaining, replace=False)
                        score_np = score_np[choose]
                        label_np = label_np[choose]

                    roc_scores.extend(score_np.tolist())
                    roc_labels.extend(label_np.tolist())

                if (
                    compute_speech_metrics
                    and stft_params is not None
                    and speech_eval_count < speech_metrics_max_utterances
                ):
                    for sample_idx in range(pred_eval.shape[0]):
                        if speech_eval_count >= speech_metrics_max_utterances:
                            break

                        t_len = int(lengths[sample_idx].item())
                        if t_len <= 1:
                            continue

                        pred_log_mag_tf = pred_eval[sample_idx, :t_len, :].detach().cpu().numpy()
                        pred_mag_ft = np.maximum(np.expm1(pred_log_mag_tf).T, 0.0)

                        noisy_phase_tf = batch["noisy_phase"][sample_idx][:t_len, :]
                        clean_stft_tf = batch["clean_stft"][sample_idx][:t_len, :]

                        z_hat = pred_mag_ft * np.exp(1j * noisy_phase_tf.T)
                        est_wav = stft_to_waveform(z_hat, stft_params)
                        ref_wav = stft_to_waveform(clean_stft_tf.T, stft_params)

                        m = min(est_wav.shape[0], ref_wav.shape[0])
                        if m <= 1:
                            continue

                        est_wav = est_wav[:m]
                        ref_wav = ref_wav[:m]

                        si_sdr_scores.append(compute_si_sdr(ref_wav, est_wav))

                        if HAS_STOI:
                            try:
                                stoi_scores.append(float(stoi_fn(ref_wav, est_wav, sample_rate, extended=False)))
                            except Exception:
                                pass

                        if HAS_PESQ:
                            try:
                                pesq_scores.append(float(pesq_fn(sample_rate, ref_wav, est_wav, pesq_mode)))
                            except Exception:
                                pass

                        speech_eval_count += 1

            running += float(loss.item())
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"Processed {n_batches} batches, running loss: {running / max(n_batches, 1)}")

    loss = running / max(n_batches, 1)
    mse = sse / max(n_points, 1)

    sst = sum_y2 - ((sum_y ** 2) / max(n_points, 1))
    r2 = float("nan") if sst <= 1e-12 else 1.0 - (sse / sst)

    total_cls = tp + tn + fp + fn
    accuracy = float("nan") if total_cls == 0 else (tp + tn) / total_cls

    precision = float("nan") if (tp + fp) == 0 else tp / (tp + fp)
    recall = float("nan") if (tp + fn) == 0 else tp / (tp + fn)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    roc_auc = compute_binary_roc_auc(np.array(roc_labels), np.array(roc_scores))
    pesq_value = average_or_nan(pesq_scores)
    stoi_value = average_or_nan(stoi_scores)
    si_sdr_value = average_or_nan(si_sdr_scores)

    return {
        "loss": loss,
        "mse": mse,
        "r2": r2,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "pesq": pesq_value,
        "stoi": stoi_value,
        "si_sdr": si_sdr_value,
        "speech_eval_count": speech_eval_count,
    }


############################################################


# Grid search hyperparameter tuning

def checkpoint_value(metrics, objective, lower_is_better):
    if objective in metrics and not np.isnan(metrics[objective]):
        return float(metrics[objective])
    if lower_is_better:
        return float("inf")
    return -float("inf")


def build_tuning_trials(cfg):
    tuning_cfg = cfg.get("hyperparameter_tuning", {})
    param_grid = tuning_cfg.get("param_grid", {})

    if not param_grid:
        return []

    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    random.shuffle(combos)
    max_trials = tuning_cfg.get("max_trials")
    if max_trials is not None:
        combos = combos[: int(max_trials)]

    return combos


def train_with_config(cfg, device, run_label="main"):
    cache_dir = cfg["cache_dir"]
    train_split = cfg["train_split"]
    val_split = cfg["val_split"]
    out_dir = cfg.get("out_dir", "artifacts/train")
    os.makedirs(out_dir, exist_ok=True)

    train_loader = make_loader(
        cache_dir=cache_dir,
        split_path=train_split,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        max_files=cfg.get("max_train_files"),
        max_frames=cfg.get("max_frames"),
    )

    val_loader = make_loader(
        cache_dir=cache_dir,
        split_path=val_split,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        max_files=cfg.get("max_val_files"),
        max_frames=cfg.get("max_frames"),
    )

    n_freq = infer_n_freq(cache_dir, train_split)

    model = LSTMDenoiser(
        n_freq=n_freq,
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    epochs = cfg.get("epochs", 30)
    grad_clip = cfg.get("grad_clip", 1.0)
    metric_threshold = cfg.get("metric_threshold", 0.1)
    metric_max_points = cfg.get("metric_max_points", 250000)
    checkpoint_metric = cfg.get("checkpoint_metric", "mse")
    lower_is_better = checkpoint_metric not in {"r2", "accuracy", "f1", "roc_auc", "pesq", "stoi", "si_sdr"}

    stft_params = {
        "target_sr": int(cfg.get("target_sr", 16000)),
        "n_fft": int(cfg.get("n_fft", 512)),
        "n_hop": int(cfg.get("n_hop", 128)),
        "win": cfg.get("win", "hann"),
    }
    speech_metrics_split = cfg.get("speech_metrics_split", "val")
    speech_metrics_max_utterances = int(cfg.get("speech_metrics_max_utterances", 0))
    pesq_mode = cfg.get("pesq_mode", "wb")

    compute_train_speech = speech_metrics_split in {"train", "both"}
    compute_val_speech = speech_metrics_split in {"val", "both"}

    if speech_metrics_max_utterances > 0 and (compute_train_speech or compute_val_speech):
        if not HAS_STOI:
            print("Warning: STOI metric unavailable. Install pystoi to enable STOI reporting.")
        if not HAS_PESQ:
            print("Warning: PESQ metric unavailable. Install pesq to enable PESQ reporting.")

    best_value = float("inf") if lower_is_better else -float("inf")
    best_epoch = 0

    print(f"[{run_label}] Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print(f"[{run_label}] Output directory: {out_dir}")
    print(f"[{run_label}] EPOCHS: {epochs}")
    print("\n#### Starting training loop ####\n")

    for epoch in range(1, epochs + 1):
        print(f"[{run_label}] STARTING epoch {epoch} of {epochs}")

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=grad_clip,
            metric_threshold=metric_threshold,
            metric_max_points=metric_max_points,
            compute_speech_metrics=compute_train_speech,
            speech_metrics_max_utterances=speech_metrics_max_utterances,
            sample_rate=stft_params["target_sr"],
            pesq_mode=pesq_mode,
            stft_params=stft_params,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            metric_threshold=metric_threshold,
            metric_max_points=metric_max_points,
            compute_speech_metrics=compute_val_speech,
            speech_metrics_max_utterances=speech_metrics_max_utterances,
            sample_rate=stft_params["target_sr"],
            pesq_mode=pesq_mode,
            stft_params=stft_params,
        )

        print(f"[{run_label}] COMPLETED epoch {epoch} of {epochs}")
        print(f"[{run_label}] TRAIN metrics: {format_metrics(train_metrics)}")
        print(f"[{run_label}] VAL   metrics: {format_metrics(val_metrics)}")

        ckpt_latest = os.path.join(out_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": cfg,
            },
            ckpt_latest,
        )

        current_value = checkpoint_value(val_metrics, checkpoint_metric, lower_is_better)
        improved = (current_value < best_value) if lower_is_better else (current_value > best_value)
        if improved:
            best_value = current_value
            best_epoch = epoch
            ckpt_best = os.path.join(out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_best,
            )
            print(f"[{run_label}] !! NEW BEST checkpoint !!: {ckpt_best}")

        print(f"[{run_label}] Checkpoint saved\n")

    print(f"\n[{run_label}] Training complete. Best {checkpoint_metric}: {best_value} at epoch {best_epoch}")

    return {
        "best_value": best_value,
        "best_epoch": best_epoch,
        "checkpoint_metric": checkpoint_metric,
    }


def run_hyperparameter_tuning(base_cfg, device):
    tuning_cfg = base_cfg.get("hyperparameter_tuning", {})
    trials = build_tuning_trials(base_cfg)
    if not trials:
        print("No hyperparameter tuning trials found. Add hyperparameter_tuning.param_grid in config.")
        return None

    tuning_out_dir = tuning_cfg.get(
        "out_dir",
        os.path.join(base_cfg.get("out_dir", "artifacts/train"), "tuning"),
    )
    os.makedirs(tuning_out_dir, exist_ok=True)

    tune_epochs = tuning_cfg.get("tune_epochs", min(5, base_cfg.get("epochs", 30)))
    objective = tuning_cfg.get("objective", "best_value")
    maximize = objective in {"r2", "accuracy", "f1", "roc_auc"}

    results = []
    for idx, params in enumerate(trials, start=1):
        trial_cfg = dict(base_cfg)
        trial_cfg.update(params)
        trial_cfg["epochs"] = tune_epochs
        trial_cfg["out_dir"] = os.path.join(tuning_out_dir, f"trial_{idx:03d}")

        print("\n============================================================")
        print(f"TUNING TRIAL {idx}/{len(trials)}")
        print(f"params: {params}")
        print("============================================================\n")

        summary = train_with_config(trial_cfg, device, run_label=f"tune-{idx:03d}")
        result = {
            "trial": idx,
            "params": params,
            "best_value": summary["best_value"],
            "best_epoch": summary["best_epoch"],
            "checkpoint_metric": summary["checkpoint_metric"],
        }
        results.append(result)

    if maximize:
        best_result = max(results, key=lambda item: item["best_value"])
    else:
        best_result = min(results, key=lambda item: item["best_value"])

    results_path = os.path.join(tuning_out_dir, "tuning_results.yaml")
    with open(results_path, "w") as fh:
        yaml.safe_dump(
            {
                "objective": objective,
                "tune_epochs": tune_epochs,
                "best_trial": best_result,
                "all_trials": results,
            },
            fh,
            sort_keys=False,
        )

    print("\n#### Hyperparameter tuning complete ####")
    print(f"Best trial: {best_result['trial']}")
    print(f"Best params: {best_result['params']}")
    print(f"Best value: {best_result['best_value']}")
    print(f"Saved tuning summary: {results_path}\n")

    if tuning_cfg.get("retrain_best", False):
        final_cfg = dict(base_cfg)
        final_cfg.update(best_result["params"])
        final_cfg["out_dir"] = tuning_cfg.get(
            "retrain_out_dir",
            os.path.join(base_cfg.get("out_dir", "artifacts/train"), "best_from_tuning"),
        )
        print("\n#### Retraining best hyperparameters on full epoch schedule ####\n")
        train_with_config(final_cfg, device, run_label="retrain-best")

    return best_result


############################################################


def main():

    print("#### Initializing training program ####\n")

    args = parse_args()
    cfg = load_config(CONFIG_PATH)

    # use test mode?
    if args.test or cfg.get("test_mode", False):
        cfg = apply_test_overrides(cfg)
        print("\n\n#### TEST MODE IS ENABLED PLEASE DON'T FORGET TO DISABLE IT!!!!!! ####\n\n")

    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    # use best avaliable device for training
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(cfg.get("seed", 42))
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    tune_requested = args.tune or cfg.get("hyperparameter_tuning", {}).get("enabled", False)
    if tune_requested:
        run_hyperparameter_tuning(cfg, device)
    else:
        train_with_config(cfg, device, run_label="main")


if __name__ == "__main__":
    main()
