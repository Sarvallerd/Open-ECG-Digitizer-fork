from collections import defaultdict
import math
import os
from pathlib import Path
import random
import signal
import optuna
import torch
import pandas as pd
import numpy as np

from oed.hp_optimizer.utils import (
    calculate_snr,
    digitize_image,
    get_slice,
    leads_names,
    load_png_file,
)


def get_pixel_size_finder_args(trial: optuna.trial.Trial) -> dict[str, int | float]:
    return {
        "samples": trial.suggest_int("samples", 100, 2000, step=5),
        "min_number_of_grid_lines": trial.suggest_int(
            "min_number_of_grid_lines", 5, 50, step=2
        ),
        "max_number_of_grid_lines": trial.suggest_int(
            "max_number_of_grid_lines", 50, 200, step=5
        ),
        "max_zoom": trial.suggest_int("max_zoom", 5, 20),
        "zoom_factor": trial.suggest_float("zoom_factor", 2.0, 20.0, step=0.5),
        "lower_grid_line_factor": trial.suggest_float(
            "lower_grid_line_factor", 0.05, 1.0, step=0.05
        ),
    }


def get_signal_extractor_args(trial: optuna.trial.Trial) -> dict[str, int | float]:
    return {
        "threshold_sum": trial.suggest_float("threshold_sum", 5.0, 30.0, step=0.5),
        "threshold_line_in_mask": trial.suggest_float(
            "threshold_line_in_mask", 0.5, 1.0, step=0.1
        ),
        "label_thresh": trial.suggest_float("label_thresh", 0.01, 0.5, step=0.01),
        "max_iterations": trial.suggest_int("max_iterations", 3, 10),
        "split_num_stripes": trial.suggest_int("split_num_stripes", 1, 10),
        "candidate_span": trial.suggest_int("candidate_span", 5, 20),
        "lam": trial.suggest_float("lam", 0.1, 0.8, step=0.1),
        "min_line_width": trial.suggest_int("min_line_width", 15, 50),
    }


def digitize(
    path: str,
    device: str,
    target_num_samples: int,
    model: torch.nn.Module,
    target_df: pd.DataFrame,
    pixel_size_finder_args: dict[str, int | float],
    signal_extractor_args: dict[str, int | float],
) -> dict[str, float]:
    input_img = load_png_file(path)
    if not os.path.exists(path):
        return
    _, _, _, lines = digitize_image(
        input_img,
        2500,
        target_num_samples,
        model=model,
        device=device,
        pixel_size_finder_args=pixel_size_finder_args,
        signal_extractor_args=signal_extractor_args,
    )
    lead_scores = {}
    for i, lead_name in enumerate(leads_names):
        number_of_rows_in_lead = target_df[lead_name].dropna().shape[0]
        lead_data = lines[i]
        lead_data = lead_data[get_slice(lead_name, number_of_rows_in_lead)].numpy()
        if lead_data.shape[0] < number_of_rows_in_lead:
            lead_data_ = np.empty(number_of_rows_in_lead)
            lead_data_[:lead_data.shape[0]] = lead_data
            lead_data = lead_data_
        mean_val = np.nanmean(lead_data)
        if np.isnan(mean_val):
            mean_val = 0.0
        lead_data = np.nan_to_num(lead_data, nan=mean_val)
        lead_scores[lead_name] = calculate_snr(
            target_df[lead_name].dropna().to_numpy(), lead_data
        )
    return lead_scores


def timeout_handler(signum, frame):
    raise TimeoutError("Функция выполняется слишком долго")


def obj(
    trial: optuna.trial.Trial,
    paths: list[Path],
    model: torch.nn.Module,
    pci_stats_df: pd.DataFrame,
    device: str = "cuda",
    paths_pct: float = 0.1,
    max_digitize_duration: int = 60,
):
    pixel_size_finder_args = get_pixel_size_finder_args(trial)
    signal_extractor_args = get_signal_extractor_args(trial)

    full_metrics = defaultdict(list[float])

    for path in paths:
        image_id = int(path.stem.split("-")[0])
        target_df = pd.read_csv(path.parent / f"{image_id}.csv")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_digitize_duration * 10)
        try:
            leads_metrics = digitize(
                path=path.as_posix(),
                device=device,
                target_num_samples=pci_stats_df[pci_stats_df["id"] == image_id][
                    "fs"
                ].iloc[0]
                * 10,
                model=model,
                target_df=target_df,
                pixel_size_finder_args=pixel_size_finder_args,
                signal_extractor_args=signal_extractor_args,
            )
            signal.alarm(0)
        except TimeoutError:
            leads_metrics = {lead_name: -1e5 for lead_name in leads_names}

        for k, v in leads_metrics.items():
            full_metrics[k].append(v)
    return tuple([np.mean(v) for v in full_metrics.values()])


def run_study(
    paths: list[Path],
    model: torch.nn.Module,
    pci_stats_df: pd.DataFrame,
    n_trials: int = 100,
    n_jobs: int = 5,
    paths_pct: float = 0.1,
    max_digitize_duration: int = 60,
) -> dict[str, int | float]:
    study = optuna.create_study(directions=["maximize" for _ in leads_names])
    study.optimize(
        lambda trial: obj(
            trial=trial,
            paths=paths,
            model=model,
            pci_stats_df=pci_stats_df,
            paths_pct=paths_pct,
            max_digitize_duration=max_digitize_duration,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    return study.best_trials
