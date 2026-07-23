"""CRG/WBC compliance plotting and quantitative QA helpers.

The public entry points in this module are used by ``plot_joint_data.py``.
They intentionally depend only on the self-describing compliance logs, so a
saved CRG/WBC experiment can be plotted without the legacy walking/EKF logs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
if not hasattr(np, "typeDict"):
    # Compatibility for the system SciPy bundled with the robot environment.
    np.typeDict = np.sctypeDict
for _np_alias, _np_type in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
}.items():
    if _np_alias not in np.__dict__:
        setattr(np, _np_alias, _np_type)
from scipy.spatial.transform import Rotation as Rotation


AXIS_COLORS = ("#D55E00", "#009E73", "#0072B2")
LEFT_COLOR = "#0072B2"
RIGHT_COLOR = "#D55E00"
REFERENCE_COLOR = "#202124"
TORSO_COLOR = "#0072B2"
ARM_COLOR = "#E69F00"
RECONSTRUCTION_COLOR = "#CC79A7"
FORCE_WINDOW_COLOR = "#F0E442"
GRID_COLOR = "#D9DCE1"

PLOT_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#4A4D52",
    "axes.labelcolor": "#202124",
    "xtick.color": "#4A4D52",
    "ytick.color": "#4A4D52",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}

LOG_FILENAMES = {
    "applied": "applied_external_wrist_force.txt",
    "torso": "compliance_torso_state.txt",
    "jacobian": "compliance_torso_jacobian.txt",
    "wbc_torso": "wbc_torso_orientation_tracking.txt",
    "wbc_hand": "wbc_hand_compliance_tracking.txt",
    "wbc_hand_position": "wbc_hand_compliance_position_tracking.txt",
    "wrist_validation": "wrist_force_validation.txt",
}

COMPLETE_EVIDENCE_LOGS = (
    "applied",
    "torso",
    "jacobian",
    "wbc_torso",
    "wbc_hand",
    "wbc_hand_position",
    "wrist_validation",
)

REQUIRED_COLUMNS = {
    "applied": (
        "l_enabled", "r_enabled",
        "l_fx", "l_fy", "l_fz", "l_tx", "l_ty", "l_tz",
        "r_fx", "r_fy", "r_fz", "r_tx", "r_ty", "r_tz",
    ),
    "torso": (
        *(f"l_w{index}" for index in range(6)),
        *(f"r_w{index}" for index in range(6)),
        *(f"l_dxf{index}" for index in range(3)),
        *(f"r_dxf{index}" for index in range(3)),
        *(f"xb{index}" for index in range(3, 6)),
        *(f"xbf{index}" for index in range(3, 6)),
        *(f"xbfinal{index}" for index in range(3, 6)),
        *(f"l_arm{index}" for index in range(3)),
        *(f"r_arm{index}" for index in range(3)),
        "qp_solved",
    ),
    "jacobian": (
        "valid",
        *(
            f"{side}_Ab_{row}_{column}"
            for side in ("l", "r")
            for row in range(6)
            for column in range(3)
        ),
    ),
    "wbc_torso": (
        "offset_roll", "offset_pitch", "offset_yaw",
        "nom_roll", "nom_pitch", "nom_yaw",
        "des_roll", "des_pitch", "des_yaw",
        "cur_roll", "cur_pitch", "cur_yaw",
        "err_rotvec_x", "err_rotvec_y", "err_rotvec_z",
    ),
    "wbc_hand": (
        *(f"l_ref{index}" for index in range(3)),
        *(f"l_ach{index}" for index in range(3)),
        *(f"r_ref{index}" for index in range(3)),
        *(f"r_ach{index}" for index in range(3)),
    ),
    "wbc_hand_position": (
        *(f"{side}_{kind}_{axis}"
          for side in ("l", "r")
          for kind in ("ref", "ach")
          for axis in ("x", "y", "z")),
    ),
    "wrist_validation": (
        *(
            f"{side}_{kind}_{axis}"
            for side in ("l", "r")
            for kind in ("gt", "est", "estf")
            for axis in ("fx", "fy", "fz")
        ),
    ),
}

APPLIED_WRENCH_COLUMNS = (
    "l_fx", "l_fy", "l_fz", "l_tx", "l_ty", "l_tz",
    "r_fx", "r_fy", "r_fz", "r_tx", "r_ty", "r_tz",
)

SINGLE_FIGURE_STEMS = (
    "01_mujoco_applied_external_force",
    "02_crg_total_torso_hand_allocation",
    "03_wbc_torso_orientation_tracking",
    "04_wbc_hand_position_tracking",
    "05_wbc_hand_acceleration_tracking",
    "06_estimated_external_force",
)

# Remove files produced by the previous seven-figure layout when committing a
# new run, otherwise an existing output directory would misleadingly contain
# both the old and the new evidence sets.
LEGACY_SINGLE_FIGURE_STEMS = (
    "04_wbc_hand_acceleration_tracking",
    "05_estimated_external_force",
    "01_external_wrench_to_crg",
    "02_crg_allocation_decomposition",
    "03_crg_torso_reference",
    "04_wbc_torso_orientation_tracking",
    "05_wbc_hand_acceleration_tracking",
    "06_end_to_end_compliance_summary",
    "07_wrist_force_estimator_diagnostic",
)

SINGLE_CLEANUP_FIGURE_STEMS = tuple(
    dict.fromkeys((*SINGLE_FIGURE_STEMS, *LEGACY_SINGLE_FIGURE_STEMS))
)

BATCH_FIGURE_STEMS = (
    "crg_wbc_run_comparison",
    "crg_wbc_hand_task_comparison",
)

SUPPORTED_FIGURE_FORMATS = ("png", "pdf", "svg")


@dataclass
class NamedLog:
    path: Path
    names: Tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        self._index = {name: idx for idx, name in enumerate(self.names)}

    @property
    def time(self) -> np.ndarray:
        return self.column("time")

    def has(self, *names: str) -> bool:
        return all(name in self._index for name in names)

    def column(self, name: str) -> np.ndarray:
        if name not in self._index:
            raise KeyError(f"{self.path.name} does not contain column '{name}'")
        return self.values[:, self._index[name]]

    def columns(self, names: Sequence[str]) -> np.ndarray:
        missing = [name for name in names if name not in self._index]
        if missing:
            raise KeyError(
                f"{self.path.name} is missing columns: {', '.join(missing)}"
            )
        return self.values[:, [self._index[name] for name in names]]


@dataclass
class ComplianceRun:
    folder: Path
    label: str
    logs: Dict[str, NamedLog]
    metadata: Dict[str, str]
    force_window: Optional[Tuple[float, float]]
    nonzero_wrench_window: Optional[Tuple[float, float]]
    causal_delay_s: float = 0.0


def parse_formats(value: str) -> Tuple[str, ...]:
    formats = tuple(
        item.strip().lower().lstrip(".")
        for item in value.split(",")
        if item.strip()
    )
    supported = set(SUPPORTED_FIGURE_FORMATS)
    if not formats:
        raise ValueError("At least one output format is required")
    unsupported = sorted(set(formats) - supported)
    if unsupported:
        raise ValueError(
            "Unsupported output format(s): " + ", ".join(unsupported)
        )
    return tuple(dict.fromkeys(formats))


def _clear_known_artifacts(
    output_dir: Path,
    figure_stems: Sequence[str],
    data_filenames: Sequence[str],
) -> None:
    if not output_dir.is_dir():
        return
    paths = [
        output_dir / f"{stem}.{extension}"
        for stem in figure_stems
        for extension in SUPPORTED_FIGURE_FORMATS
    ]
    paths.extend(output_dir / filename for filename in data_filenames)
    for path in paths:
        if path.is_file() or path.is_symlink():
            path.unlink()


def _commit_known_artifacts(
    staging_dir: Path,
    output_dir: Path,
    figure_stems: Sequence[str],
    data_filenames: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_known_artifacts(output_dir, figure_stems, data_filenames)
    for source in staging_dir.iterdir():
        if source.is_file():
            os.replace(str(source), str(output_dir / source.name))


def folder_has_compliance_logs(folder: os.PathLike[str] | str) -> bool:
    folder_path = Path(folder).expanduser()
    return (folder_path / LOG_FILENAMES["torso"]).is_file()


def _missing_required_columns(run: ComplianceRun) -> Dict[str, Tuple[str, ...]]:
    missing: Dict[str, Tuple[str, ...]] = {}
    for key in COMPLETE_EVIDENCE_LOGS:
        log = run.logs.get(key)
        if log is None:
            continue
        absent = tuple(
            name for name in REQUIRED_COLUMNS[key] if not log.has(name)
        )
        if absent:
            missing[key] = absent
    return missing


def _read_metadata(folder: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for filename in ("manifest.txt", "compliance_test_metadata.txt"):
        path = folder / filename
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def _load_named_log(path: Path) -> NamedLog:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty log: {path}")

    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    names = tuple(header.split())
    if not names or names[0] != "time":
        raise ValueError(f"{path} has no supported named header")
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate column names")

    values = np.loadtxt(path, skiprows=1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(
            f"{path} has {values.shape[1] if values.ndim == 2 else 'invalid'} "
            f"data columns but {len(names)} header columns"
        )
    if values.shape[0] == 0:
        raise ValueError(f"{path} contains no data rows")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains NaN or Inf")

    time = values[:, 0]
    if values.shape[0] > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} time must be strictly increasing")
    return NamedLog(path=path, names=names, values=values)


def _infer_force_windows(
    applied: Optional[NamedLog],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if applied is None:
        return None, None

    required = (
        "l_enabled",
        "r_enabled",
        "l_fx",
        "l_fy",
        "l_fz",
        "l_tx",
        "l_ty",
        "l_tz",
        "r_fx",
        "r_fy",
        "r_fz",
        "r_tx",
        "r_ty",
        "r_tz",
    )
    if not applied.has(*required):
        return None, None

    enabled = (applied.column("l_enabled") > 0.5) | (
        applied.column("r_enabled") > 0.5
    )
    wrench = applied.columns(
        (
            "l_fx",
            "l_fy",
            "l_fz",
            "l_tx",
            "l_ty",
            "l_tz",
            "r_fx",
            "r_fy",
            "r_fz",
            "r_tx",
            "r_ty",
            "r_tz",
        )
    )
    nonzero = np.linalg.norm(wrench, axis=1) > 1.0e-10

    def bounds(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return None
        return float(applied.time[indices[0]]), float(applied.time[indices[-1]])

    return bounds(enabled), bounds(nonzero)


def load_compliance_run(
    folder: os.PathLike[str] | str,
    label: Optional[str] = None,
    force_window_override: Optional[Sequence[float]] = None,
) -> ComplianceRun:
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Log folder does not exist: {folder_path}")

    logs: Dict[str, NamedLog] = {}
    load_errors: List[str] = []
    for key, filename in LOG_FILENAMES.items():
        path = folder_path / filename
        if not path.exists():
            continue
        try:
            logs[key] = _load_named_log(path)
        except (OSError, ValueError) as exc:
            load_errors.append(str(exc))

    if "torso" not in logs:
        detail = f" ({'; '.join(load_errors)})" if load_errors else ""
        raise ValueError(
            f"{folder_path} has no readable {LOG_FILENAMES['torso']}{detail}"
        )
    if load_errors:
        for error in load_errors:
            print(f"[WARN] {error}")

    force_window, nonzero_window = _infer_force_windows(logs.get("applied"))
    if force_window_override is not None:
        if len(force_window_override) != 2:
            raise ValueError("force window override requires START END")
        start, end = (float(force_window_override[0]), float(force_window_override[1]))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("force window must be finite and END > START")
        force_window = (start, end)
        nonzero_window = (start, end)

    metadata = _read_metadata(folder_path)
    run_label = label or folder_path.name or str(folder_path)
    run = ComplianceRun(
        folder=folder_path,
        label=run_label,
        logs=logs,
        metadata=metadata,
        force_window=force_window,
        nonzero_wrench_window=nonzero_window,
    )
    run.causal_delay_s = _detect_applied_to_crg_delay(run)
    return run


def _window_mask(
    time: np.ndarray,
    run: ComplianceRun,
    *,
    causal: bool = False,
) -> np.ndarray:
    window = run.nonzero_wrench_window or run.force_window
    if window is None:
        return np.ones(time.shape, dtype=bool)
    delay = run.causal_delay_s if causal and run.nonzero_wrench_window else 0.0
    tolerance = 1.0e-12
    if time.size > 1:
        tolerance = max(tolerance, abs(float(np.median(np.diff(time)))) * 1.0e-6)
    return (time >= window[0] + delay - tolerance) & (
        time <= window[1] + delay + tolerance
    )


def _plot_indices(
    time: np.ndarray,
    max_points: int,
    time_range: Optional[Sequence[float]],
) -> np.ndarray:
    mask = np.ones(time.shape, dtype=bool)
    if time_range is not None:
        if len(time_range) != 2:
            raise ValueError("plot time range requires START END")
        start, end = float(time_range[0]), float(time_range[1])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("plot time range must be finite and END > START")
        mask &= (time >= start) & (time <= end)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Requested plot time range contains no samples")
    if max_points > 0 and indices.size > max_points:
        selected = np.linspace(0, indices.size - 1, max_points, dtype=int)
        indices = indices[selected]
    return indices


def _interp_columns(
    source: NamedLog,
    names: Sequence[str],
    target_time: np.ndarray,
) -> np.ndarray:
    source_values = source.columns(names)
    result = np.empty((target_time.size, len(names)))
    for column in range(len(names)):
        result[:, column] = np.interp(
            target_time,
            source.time,
            source_values[:, column],
            left=np.nan,
            right=np.nan,
        )
    return result


def _detect_applied_to_crg_delay(run: ComplianceRun) -> float:
    applied = run.logs.get("applied")
    torso = run.logs.get("torso")
    if applied is None or torso is None or torso.time.size < 2:
        return 0.0
    torso_names = tuple(f"l_w{idx}" for idx in range(6)) + tuple(
        f"r_w{idx}" for idx in range(6)
    )
    applied_names = (
        "l_fx", "l_fy", "l_fz", "l_tx", "l_ty", "l_tz",
        "r_fx", "r_fy", "r_fz", "r_tx", "r_ty", "r_tz",
    )
    if not torso.has(*torso_names) or not applied.has(*applied_names):
        return 0.0

    median_dt = float(np.median(np.diff(torso.time)))
    candidates = tuple(dict.fromkeys((0.0, max(0.0, median_dt))))
    source_window = run.nonzero_wrench_window or run.force_window
    best_score = float("inf")
    best_delay = 0.0
    for delay in candidates:
        applied_values = _interp_columns(
            applied, applied_names, torso.time - delay
        )
        difference = torso.columns(torso_names) - applied_values
        valid = np.all(np.isfinite(difference), axis=1)
        if source_window is not None:
            valid &= (torso.time >= source_window[0] + delay) & (
                torso.time <= source_window[1] + delay
            )
        if not np.any(valid):
            continue
        score = float(np.sqrt(np.mean(np.sum(difference[valid] ** 2, axis=1))))
        if score < best_score:
            best_score = score
            best_delay = delay
    return best_delay


def _vector_rmse(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if selected.shape[0] == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(np.square(selected), axis=1))))


def _vector_peak(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if selected.shape[0] == 0:
        return float("nan")
    return float(np.max(np.linalg.norm(selected, axis=1)))


def _axis_rmse(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if selected.shape[0] == 0:
        return np.full(values.shape[1], np.nan)
    return np.sqrt(np.mean(np.square(selected), axis=0))


def _ab_contribution(
    run: ComplianceRun,
    torso_time: np.ndarray,
    side: str,
    torso_angles: np.ndarray,
) -> Optional[np.ndarray]:
    jacobian = run.logs.get("jacobian")
    if jacobian is None:
        return None
    names = [
        f"{side}_Ab_{row}_{column}"
        for row in range(6)
        for column in range(3)
    ]
    if not jacobian.has(*names):
        return None
    flat = _interp_columns(jacobian, names, torso_time)
    matrices = flat.reshape((-1, 6, 3))
    return np.einsum("nij,nj->ni", matrices, torso_angles)


def _relative_torso_offsets(log: NamedLog) -> Tuple[np.ndarray, np.ndarray]:
    nominal = log.columns(("nom_roll", "nom_pitch", "nom_yaw"))

    def rotations(rpy: np.ndarray) -> Rotation:
        return Rotation.from_euler("ZYX", rpy[:, [2, 1, 0]])

    nominal_rotation = rotations(nominal)
    if log.has(
        "offset_roll", "offset_pitch", "offset_yaw",
        "err_rotvec_x", "err_rotvec_y", "err_rotvec_z",
    ):
        # WalkingManager::err_rotation(desired, current) logs
        #   e = angle * R_desired * axis,
        # where Exp(angle * axis) = R_current^T * R_desired. Reconstruct the
        # measured rotation from this small, high-resolution SO(3) error
        # instead of subtracting quantized absolute Euler angles near +/-pi.
        # The latter produces artificial 1e-5-rad stair steps in old logs.
        desired_relative = rotations(
            log.columns(("offset_roll", "offset_pitch", "offset_yaw"))
        )
        desired_rotation = desired_relative * nominal_rotation
        error_world = log.columns(
            ("err_rotvec_x", "err_rotvec_y", "err_rotvec_z")
        )
        error_desired = desired_rotation.inv().apply(error_world)
        desired_from_current = Rotation.from_rotvec(error_desired)
        current_rotation = desired_rotation * desired_from_current.inv()
        current_relative = current_rotation * nominal_rotation.inv()
    else:
        desired = log.columns(("des_roll", "des_pitch", "des_yaw"))
        current = log.columns(("cur_roll", "cur_pitch", "cur_yaw"))
        desired_relative = rotations(desired) * nominal_rotation.inv()
        current_relative = rotations(current) * nominal_rotation.inv()

    desired_ypr = desired_relative.as_euler("ZYX")
    current_ypr = current_relative.as_euler("ZYX")
    return desired_ypr[:, [2, 1, 0]], current_ypr[:, [2, 1, 0]]


def compute_compliance_metrics(run: ComplianceRun) -> Dict[str, object]:
    rho = _metadata_float(run.metadata, "crg_rho")
    for alias in ("rho", "crg_rou", "rou"):
        if math.isfinite(rho):
            break
        rho = _metadata_float(run.metadata, alias)
    admittance_scale = _metadata_float(
        run.metadata, "crg_admittance_scale"
    )
    metrics: Dict[str, object] = {
        "label": run.label,
        "folder": str(run.folder),
        "rho": rho,
        "crg_admittance_scale": admittance_scale,
        "crg_arm_ma_translation_kg": _metadata_float(
            run.metadata, "crg_arm_ma_translation_kg"
        ),
        "crg_arm_da_translation_n_s_per_m": _metadata_float(
            run.metadata, "crg_arm_da_translation_n_s_per_m"
        ),
        "crg_arm_ka_translation_n_per_m": _metadata_float(
            run.metadata, "crg_arm_ka_translation_n_per_m"
        ),
        "crg_admittance_translation_limit_enabled": _metadata_float(
            run.metadata, "crg_admittance_translation_limit_enabled"
        ),
        "crg_admittance_translation_limit_m": _metadata_float(
            run.metadata, "crg_admittance_translation_limit_m"
        ),
        "crg_admittance_translation_limit_effective_m": _metadata_float(
            run.metadata, "crg_admittance_translation_limit_effective_m"
        ),
        "crg_arm_force_at_translation_limit_n": _metadata_float(
            run.metadata, "crg_arm_force_at_translation_limit_n"
        ),
        "run_status": run.metadata.get("status", "unknown"),
        "applied_to_crg_causal_delay_s": run.causal_delay_s,
    }
    if run.force_window is not None:
        metrics["force_window_start_s"] = run.force_window[0]
        metrics["force_window_end_s"] = run.force_window[1]
    if run.nonzero_wrench_window is not None:
        metrics["nonzero_wrench_start_s"] = run.nonzero_wrench_window[0]
        metrics["nonzero_wrench_end_s"] = run.nonzero_wrench_window[1]

    applied = run.logs.get("applied")
    if applied is not None and applied.has(*APPLIED_WRENCH_COLUMNS):
        mask = _window_mask(applied.time, run)
        left_force = applied.columns(("l_fx", "l_fy", "l_fz"))
        right_force = applied.columns(("r_fx", "r_fy", "r_fz"))
        left_torque = applied.columns(("l_tx", "l_ty", "l_tz"))
        right_torque = applied.columns(("r_tx", "r_ty", "r_tz"))
        nonzero_mask = np.linalg.norm(
            applied.columns(APPLIED_WRENCH_COLUMNS), axis=1
        ) > 1.0e-10
        metrics.update(
            {
                "applied_samples": int(applied.values.shape[0]),
                "nonzero_applied_samples": int(np.count_nonzero(nonzero_mask)),
                "left_force_peak_N": _vector_peak(left_force, mask),
                "right_force_peak_N": _vector_peak(right_force, mask),
                "left_torque_peak_Nm": _vector_peak(left_torque, mask),
                "right_torque_peak_Nm": _vector_peak(right_torque, mask),
            }
        )

    torso = run.logs["torso"]
    torso_mask = _window_mask(torso.time, run, causal=True)
    metrics["torso_samples"] = int(torso.values.shape[0])
    metrics["causal_analysis_torso_samples"] = int(np.count_nonzero(torso_mask))
    if run.nonzero_wrench_window is not None:
        metrics["nonzero_causal_torso_samples"] = int(
            np.count_nonzero(torso_mask)
        )
    if torso.values.shape[0] > 1:
        torso_dt = np.diff(torso.time)
        metrics["torso_median_dt_s"] = float(np.median(torso_dt))
        metrics["torso_max_dt_jitter_s"] = float(
            np.max(np.abs(torso_dt - np.median(torso_dt)))
        )

    if torso.has("qp_solved") and np.any(torso_mask):
        metrics["qp_success_percent"] = float(
            100.0 * np.mean(torso.column("qp_solved")[torso_mask] > 0.5)
        )

    angle_names = ("xbfinal3", "xbfinal4", "xbfinal5")
    if torso.has(*angle_names):
        torso_angles = torso.columns(angle_names)
        torso_angles_deg = np.rad2deg(torso_angles)
        if np.any(torso_mask):
            peak = np.max(np.abs(torso_angles_deg[torso_mask]), axis=0)
            for name, value in zip(("roll", "pitch", "yaw"), peak):
                metrics[f"torso_{name}_peak_deg"] = float(value)

        recovery_mask = _recovery_mask(torso.time, run)
        if np.any(recovery_mask):
            metrics["post_force_final_1s_torso_reference_rms_deg"] = _vector_rmse(
                torso_angles_deg, recovery_mask
            )

    for side in ("l", "r"):
        residual_names = tuple(f"{side}_arm{idx}" for idx in range(3))
        target_names = tuple(f"{side}_dxf{idx}" for idx in range(3))
        raw_admittance_names = tuple(f"{side}_dx{idx}" for idx in range(3))
        prefix = "left" if side == "l" else "right"
        if torso.has(*target_names) and np.any(torso_mask):
            target = torso.columns(target_names)
            target_window = target[torso_mask]
            metrics[f"{prefix}_admittance_target_peak_mm"] = (
                1000.0 * _vector_peak(target, torso_mask)
            )
            metrics[f"{prefix}_admittance_target_max_abs_component_mm"] = (
                1000.0 * float(np.max(np.abs(target_window)))
            )
            limit_enabled = _metric_float(
                metrics, "crg_admittance_translation_limit_enabled"
            )
            limit_m = _metric_float(
                metrics, "crg_admittance_translation_limit_m"
            )
            limit_signal = (
                torso.columns(raw_admittance_names)
                if torso.has(*raw_admittance_names)
                else target
            )
            limit_window = limit_signal[torso_mask]
            metrics[f"{prefix}_admittance_raw_max_abs_component_mm"] = (
                1000.0 * float(np.max(np.abs(limit_window)))
            )
            if (
                math.isfinite(limit_enabled)
                and limit_enabled > 0.5
                and math.isfinite(limit_m)
                and limit_m > 0.0
            ):
                metrics[f"{prefix}_admittance_limit_utilization_percent"] = (
                    100.0 * float(np.max(np.abs(limit_window))) / limit_m
                )
                near_limit = np.any(
                    np.abs(limit_window) >= 0.999 * limit_m,
                    axis=1,
                )
                metrics[f"{prefix}_admittance_near_limit_samples_percent"] = (
                    100.0 * float(np.mean(near_limit))
                )
        if torso.has(*residual_names):
            residual = torso.columns(residual_names)
            metrics[f"{prefix}_arm_residual_peak_mm"] = 1000.0 * _vector_peak(
                residual, torso_mask
            )
            metrics[f"{prefix}_arm_residual_rms_mm"] = 1000.0 * _vector_rmse(
                residual, torso_mask
            )
            recovery_mask = _recovery_mask(torso.time, run)
            if np.any(recovery_mask):
                metrics[f"post_force_final_1s_{prefix}_arm_residual_rms_mm"] = (
                    1000.0 * _vector_rmse(residual, recovery_mask)
                )
        if torso.has(*target_names, *residual_names, *angle_names):
            target = torso.columns(target_names)
            residual = torso.columns(residual_names)
            contribution = _ab_contribution(
                run, torso.time, side, torso.columns(angle_names)
            )
            if contribution is not None:
                reconstruction_error = target - (contribution[:, :3] + residual)
                metrics[f"{prefix}_allocation_reconstruction_rmse_mm"] = (
                    1000.0 * _vector_rmse(reconstruction_error, torso_mask)
                )
                metrics[f"{prefix}_allocation_reconstruction_max_mm"] = (
                    1000.0 * _vector_peak(reconstruction_error, torso_mask)
                )
                target_energy = np.sum(target * target, axis=1)
                projection_mask = torso_mask & (target_energy > 1.0e-12)
                if np.any(projection_mask):
                    torso_projection = (
                        np.sum(target * contribution[:, :3], axis=1)
                        / np.where(target_energy > 1.0e-12, target_energy, 1.0)
                    )
                    hand_projection = (
                        np.sum(target * residual, axis=1)
                        / np.where(target_energy > 1.0e-12, target_energy, 1.0)
                    )
                    metrics[
                        f"{prefix}_torso_allocation_projection_percent"
                    ] = 100.0 * float(
                        np.mean(torso_projection[projection_mask])
                    )
                    metrics[
                        f"{prefix}_hand_allocation_projection_percent"
                    ] = 100.0 * float(
                        np.mean(hand_projection[projection_mask])
                    )

    jacobian = run.logs.get("jacobian")
    if jacobian is not None and jacobian.has("valid"):
        jac_mask = _window_mask(jacobian.time, run, causal=True)
        if np.any(jac_mask):
            metrics["Ab_valid_percent"] = float(
                100.0 * np.mean(jacobian.column("valid")[jac_mask] > 0.5)
            )

    if applied is not None and torso.has(
        *(tuple(f"l_w{idx}" for idx in range(6)) +
          tuple(f"r_w{idx}" for idx in range(6)))
    ):
        torso_wrench_names = tuple(f"l_w{idx}" for idx in range(6)) + tuple(
            f"r_w{idx}" for idx in range(6)
        )
        if applied.has(*APPLIED_WRENCH_COLUMNS):
            # The applied wrench is logged before the MuJoCo step and the CRG
            # state after the control update. Compare at the delay selected
            # once for the run, and use the causally shifted force window.
            applied_at_torso = _interp_columns(
                applied,
                APPLIED_WRENCH_COLUMNS,
                torso.time - run.causal_delay_s,
            )
            difference = torso.columns(torso_wrench_names) - applied_at_torso
            finite = np.all(np.isfinite(difference), axis=1)
            comparison_mask = torso_mask & finite
            if np.any(comparison_mask):
                force_difference = difference[:, (0, 1, 2, 6, 7, 8)]
                torque_difference = difference[:, (3, 4, 5, 9, 10, 11)]
                metrics["applied_to_crg_force_rmse_N"] = _vector_rmse(
                    force_difference, comparison_mask
                )
                metrics["applied_to_crg_force_max_N"] = _vector_peak(
                    force_difference, comparison_mask
                )
                metrics["applied_to_crg_torque_rmse_Nm"] = _vector_rmse(
                    torque_difference, comparison_mask
                )
                metrics["applied_to_crg_torque_max_Nm"] = _vector_peak(
                    torque_difference, comparison_mask
                )

    wbc_torso = run.logs.get("wbc_torso")
    torso_error_names = ("err_rotvec_x", "err_rotvec_y", "err_rotvec_z")
    if wbc_torso is not None and wbc_torso.has(*torso_error_names):
        wbc_mask = _window_mask(wbc_torso.time, run, causal=True)
        error_deg = np.rad2deg(wbc_torso.columns(torso_error_names))
        axis_rmse = _axis_rmse(error_deg, wbc_mask)
        for name, value in zip(("x", "y", "z"), axis_rmse):
            metrics[f"wbc_torso_rotvec_{name}_rmse_deg"] = float(value)
        metrics["wbc_torso_3d_rmse_deg"] = _vector_rmse(error_deg, wbc_mask)
        metrics["wbc_torso_3d_max_error_deg"] = _vector_peak(
            error_deg, wbc_mask
        )
        if torso.has(
            "xbfinal3", "xbfinal4", "xbfinal5"
        ) and wbc_torso.has("offset_roll", "offset_pitch", "offset_yaw"):
            crg_reference = _interp_columns(
                torso,
                ("xbfinal3", "xbfinal4", "xbfinal5"),
                wbc_torso.time,
            )
            wbc_reference = wbc_torso.columns(
                ("offset_roll", "offset_pitch", "offset_yaw")
            )
            reference_error_deg = np.rad2deg(wbc_reference - crg_reference)
            finite = np.all(np.isfinite(reference_error_deg), axis=1)
            reference_mask = wbc_mask & finite
            if np.any(reference_mask):
                metrics["crg_to_wbc_torso_reference_rmse_deg"] = _vector_rmse(
                    reference_error_deg, reference_mask
                )
                metrics["crg_to_wbc_torso_reference_max_deg"] = _vector_peak(
                    reference_error_deg, reference_mask
                )

    wbc_hand = run.logs.get("wbc_hand")
    if wbc_hand is not None:
        hand_window_mask = _window_mask(wbc_hand.time, run, causal=True)
        for side, prefix in (("l", "left"), ("r", "right")):
            reference_names = tuple(f"{side}_ref{idx}" for idx in range(3))
            achieved_names = tuple(f"{side}_ach{idx}" for idx in range(3))
            if not wbc_hand.has(*reference_names, *achieved_names):
                continue
            reference = wbc_hand.columns(reference_names)
            achieved = wbc_hand.columns(achieved_names)
            error = achieved - reference
            hand_mask = hand_window_mask & (
                np.linalg.norm(reference, axis=1) > 1.0e-9
            )
            metrics[f"wbc_{prefix}_hand_active_task_samples"] = int(
                np.count_nonzero(hand_mask)
            )
            axis_rmse = _axis_rmse(error, hand_mask)
            for axis, value in zip(("x", "y", "z"), axis_rmse):
                metrics[f"wbc_{prefix}_hand_{axis}_acc_rmse_mps2"] = float(value)
            reference_rms = _vector_rmse(reference, hand_mask)
            error_rmse = _vector_rmse(error, hand_mask)
            metrics[f"wbc_{prefix}_hand_3d_acc_reference_rms_mps2"] = (
                reference_rms
            )
            metrics[f"wbc_{prefix}_hand_3d_acc_rmse_mps2"] = error_rmse
            if math.isfinite(reference_rms) and reference_rms > 1.0e-9:
                metrics[f"wbc_{prefix}_hand_3d_acc_nrmse_percent"] = (
                    100.0 * error_rmse / reference_rms
                )
            metrics[f"wbc_{prefix}_hand_3d_acc_max_error_mps2"] = _vector_peak(
                error, hand_mask
            )

    wbc_hand_position = run.logs.get("wbc_hand_position")
    if wbc_hand_position is not None:
        position_window_mask = _window_mask(
            wbc_hand_position.time, run, causal=True
        )
        for side, prefix in (("l", "left"), ("r", "right")):
            reference_names = tuple(
                f"{side}_ref_{axis}" for axis in ("x", "y", "z")
            )
            achieved_names = tuple(
                f"{side}_ach_{axis}" for axis in ("x", "y", "z")
            )
            if not wbc_hand_position.has(
                *reference_names, *achieved_names
            ):
                continue
            reference = wbc_hand_position.columns(reference_names)
            achieved = wbc_hand_position.columns(achieved_names)
            error_mm = 1000.0 * (achieved - reference)
            reference_mm = 1000.0 * reference
            hand_mask = position_window_mask & (
                np.linalg.norm(reference, axis=1) > 1.0e-9
            )
            metrics[
                f"wbc_{prefix}_hand_position_active_samples"
            ] = int(np.count_nonzero(hand_mask))
            axis_rmse = _axis_rmse(error_mm, hand_mask)
            for axis, value in zip(("x", "y", "z"), axis_rmse):
                metrics[
                    f"wbc_{prefix}_hand_{axis}_position_rmse_mm"
                ] = float(value)
            reference_rms = _vector_rmse(reference_mm, hand_mask)
            error_rmse = _vector_rmse(error_mm, hand_mask)
            metrics[
                f"wbc_{prefix}_hand_3d_position_reference_rms_mm"
            ] = reference_rms
            metrics[
                f"wbc_{prefix}_hand_3d_position_rmse_mm"
            ] = error_rmse
            if math.isfinite(reference_rms) and reference_rms > 1.0e-9:
                metrics[
                    f"wbc_{prefix}_hand_3d_position_nrmse_percent"
                ] = 100.0 * error_rmse / reference_rms
            metrics[
                f"wbc_{prefix}_hand_3d_position_max_error_mm"
            ] = _vector_peak(error_mm, hand_mask)

    wrist_validation = run.logs.get("wrist_validation")
    if wrist_validation is not None:
        estimator_window_mask = _window_mask(wrist_validation.time, run)
        for side, prefix in (("l", "left"), ("r", "right")):
            ground_truth_names = tuple(
                f"{side}_gt_f{axis}" for axis in ("x", "y", "z")
            )
            raw_names = tuple(
                f"{side}_est_f{axis}" for axis in ("x", "y", "z")
            )
            filtered_names = tuple(
                f"{side}_estf_f{axis}" for axis in ("x", "y", "z")
            )
            if not wrist_validation.has(
                *ground_truth_names, *raw_names, *filtered_names
            ):
                continue
            ground_truth = wrist_validation.columns(ground_truth_names)
            raw = wrist_validation.columns(raw_names)
            filtered = wrist_validation.columns(filtered_names)
            active_mask = estimator_window_mask & (
                np.linalg.norm(ground_truth, axis=1) > 1.0e-9
            )
            metrics[f"{prefix}_force_estimator_active_samples"] = int(
                np.count_nonzero(active_mask)
            )
            for estimate_name, estimate in (
                ("raw", raw),
                ("filtered", filtered),
            ):
                error = estimate - ground_truth
                axis_rmse = _axis_rmse(error, active_mask)
                for axis, value in zip(("x", "y", "z"), axis_rmse):
                    metrics[
                        f"{prefix}_force_estimator_{estimate_name}_{axis}_rmse_N"
                    ] = float(value)
                metrics[
                    f"{prefix}_force_estimator_{estimate_name}_3d_rmse_N"
                ] = _vector_rmse(error, active_mask)
                metrics[
                    f"{prefix}_force_estimator_{estimate_name}_3d_max_error_N"
                ] = _vector_peak(error, active_mask)

    return metrics


def _metadata_float(metadata: Mapping[str, str], name: str) -> float:
    try:
        value = float(metadata.get(name, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _recovery_mask(time: np.ndarray, run: ComplianceRun) -> np.ndarray:
    window = run.force_window or run.nonzero_wrench_window
    if window is None:
        return np.zeros(time.shape, dtype=bool)
    causal_end = window[1] + run.causal_delay_s
    if time[-1] <= causal_end:
        return np.zeros(time.shape, dtype=bool)
    start = max(causal_end, float(time[-1]) - 1.0)
    return (time > causal_end) & (time >= start)


def _write_metrics(
    run: ComplianceRun,
    metrics: Mapping[str, object],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "crg_wbc_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        for key, value in metrics.items():
            writer.writerow((key, _json_safe(value)))

    json_path = output_dir / "crg_wbc_metrics.json"
    json_path.write_text(
        json.dumps(
            {key: _json_safe(value) for key, value in metrics.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    qa_lines = _qa_lines(run, metrics)
    (output_dir / "crg_wbc_qa.txt").write_text(
        "\n".join(qa_lines) + "\n", encoding="utf-8"
    )


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _qa_lines(
    run: ComplianceRun, metrics: Mapping[str, object]
) -> List[str]:
    lines = [
        f"CRG/WBC plot QA: {run.label}",
        f"source={run.folder}",
    ]
    missing_columns = _missing_required_columns(run)
    for key in COMPLETE_EVIDENCE_LOGS:
        if key not in run.logs:
            lines.append(f"FAIL required_log_schema {LOG_FILENAMES[key]} missing_log")
        elif key in missing_columns:
            lines.append(
                f"FAIL required_log_schema {LOG_FILENAMES[key]} missing_columns="
                + ",".join(missing_columns[key])
            )
        else:
            lines.append(f"PASS required_log_schema {LOG_FILENAMES[key]}")

    for key, log in run.logs.items():
        if log.time.size <= 1:
            lines.append(
                f"FAIL {key}_finite_monotonic samples={log.time.size} insufficient_timing_data"
            )
            continue
        sample_dt = np.diff(log.time)
        dt = float(np.median(sample_dt))
        max_jitter = float(np.max(np.abs(sample_dt - dt)))
        lines.append(
            f"PASS {key}_finite_monotonic samples={log.time.size} "
            f"median_dt_s={dt:.9g} max_dt_jitter_s={max_jitter:.9g}"
        )
        timing_pass = abs(dt - 0.002) <= 1.0e-5 and max_jitter <= 1.0e-5
        lines.append(
            f"{'PASS' if timing_pass else 'FAIL'} {key}_500Hz_timing "
            f"expected_dt_s=0.002 median_dt_s={dt:.9g} "
            f"max_dt_jitter_s={max_jitter:.9g}"
        )

    applied = run.logs.get("applied")
    if applied is not None and applied.has(
        "l_enabled", "r_enabled", *APPLIED_WRENCH_COLUMNS
    ):
        enabled = (applied.column("l_enabled") > 0.5) | (
            applied.column("r_enabled") > 0.5
        )
        nonzero = np.linalg.norm(
            applied.columns(APPLIED_WRENCH_COLUMNS), axis=1
        ) > 1.0e-10
        violations = int(np.count_nonzero(nonzero & ~enabled))
        lines.append(
            f"{'PASS' if violations == 0 else 'FAIL'} "
            f"nonzero_wrench_implies_enabled violations={violations}"
        )

    qp = _metric_float(metrics, "qp_success_percent")
    lines.append(
        f"{'PASS' if qp >= 99.999 else 'FAIL'} qp_success_percent={qp:.9g}"
    )
    ab_valid = _metric_float(metrics, "Ab_valid_percent")
    lines.append(
        f"{'PASS' if ab_valid >= 99.999 else 'FAIL'} Ab_valid_percent={ab_valid:.9g}"
    )

    for side in ("left", "right"):
        key = f"{side}_allocation_reconstruction_max_mm"
        value = _metric_float(metrics, key)
        status = "PASS" if math.isfinite(value) and value <= 1.0e-3 else "FAIL"
        lines.append(f"{status} {key}={value:.9g}")
        torso_projection = _metric_float(
            metrics, f"{side}_torso_allocation_projection_percent"
        )
        hand_projection = _metric_float(
            metrics, f"{side}_hand_allocation_projection_percent"
        )
        rho = _metric_float(metrics, "rho")
        target_percent = 100.0 * rho
        projection_pass = (
            math.isfinite(target_percent)
            and math.isfinite(torso_projection)
            and math.isfinite(hand_projection)
            and abs(torso_projection - target_percent) <= 5.0
            and abs(hand_projection - (100.0 - target_percent)) <= 5.0
        )
        lines.append(
            f"{'PASS' if projection_pass else 'FAIL'} "
            f"{side}_allocation_projection_percent "
            f"torso={torso_projection:.9g} hand={hand_projection:.9g} "
            f"rho_target={target_percent:.9g}"
        )

    limit_enabled = _metric_float(
        metrics, "crg_admittance_translation_limit_enabled"
    )
    if math.isfinite(limit_enabled) and limit_enabled > 0.5:
        for side in ("left", "right"):
            key = f"{side}_admittance_limit_utilization_percent"
            value = _metric_float(metrics, key)
            if not math.isfinite(value) or value > 100.001:
                status = "FAIL"
            elif value >= 99.9:
                status = "WARN"
            else:
                status = "PASS"
            lines.append(f"{status} {key}={value:.9g}")
            near_limit_key = (
                f"{side}_admittance_near_limit_samples_percent"
            )
            near_limit_value = _metric_float(metrics, near_limit_key)
            near_limit_status = (
                "WARN"
                if math.isfinite(near_limit_value) and near_limit_value > 0.0
                else "PASS"
            )
            lines.append(
                f"{near_limit_status} {near_limit_key}="
                f"{near_limit_value:.9g}"
            )

    force_match = _metric_float(metrics, "applied_to_crg_force_rmse_N")
    torque_match = _metric_float(metrics, "applied_to_crg_torque_rmse_Nm")
    causal_delay = _metric_float(metrics, "applied_to_crg_causal_delay_s")
    match_pass = (
        math.isfinite(force_match)
        and math.isfinite(torque_match)
        and force_match <= 1.0e-9
        and torque_match <= 1.0e-9
    )
    lines.append(
        f"{'PASS' if match_pass else 'FAIL'} applied_to_crg_causal_match "
        f"force_rmse_N={force_match:.9g} torque_rmse_Nm={torque_match:.9g} "
        f"delay_s={causal_delay:.9g}"
    )
    reference_match = _metric_float(
        metrics, "crg_to_wbc_torso_reference_rmse_deg"
    )
    lines.append(
        f"{'PASS' if math.isfinite(reference_match) and reference_match <= 1.0e-8 else 'FAIL'} "
        f"crg_to_wbc_torso_reference_rmse_deg={reference_match:.9g}"
    )

    lines.append(
        "INFO hand_tracking_scope=arm_relative_cartesian_position_and_selected_"
        "xyz_operational_space_acceleration"
    )
    for side in ("left", "right"):
        position_samples = _metric_float(
            metrics, f"wbc_{side}_hand_position_active_samples"
        )
        position_rmse = _metric_float(
            metrics, f"wbc_{side}_hand_3d_position_rmse_mm"
        )
        position_nrmse = _metric_float(
            metrics, f"wbc_{side}_hand_3d_position_nrmse_percent"
        )
        lines.append(
            f"{'PASS' if math.isfinite(position_rmse) and position_rmse <= 2.0 else 'FAIL'} "
            f"wbc_{side}_hand_position_tracking "
            f"active_samples={position_samples:.9g} "
            f"rmse_mm={position_rmse:.9g} "
            f"nrmse_percent={position_nrmse:.9g}"
        )
        active_samples = _metric_float(
            metrics, f"wbc_{side}_hand_active_task_samples"
        )
        rmse = _metric_float(
            metrics, f"wbc_{side}_hand_3d_acc_rmse_mps2"
        )
        nrmse = _metric_float(
            metrics, f"wbc_{side}_hand_3d_acc_nrmse_percent"
        )
        lines.append(
            f"INFO wbc_{side}_hand_acceleration_tracking "
            f"active_samples={active_samples:.9g} "
            f"rmse_mps2={rmse:.9g} nrmse_percent={nrmse:.9g}"
        )

    lines.append(
        "INFO force_estimator_scope=logged_sample_ground_truth_vs_raw_and_"
        "filtered_estimates_physical_estimator_update_is_one_step_later"
    )
    for side in ("left", "right"):
        active_samples = _metric_float(
            metrics, f"{side}_force_estimator_active_samples"
        )
        raw_rmse = _metric_float(
            metrics, f"{side}_force_estimator_raw_3d_rmse_N"
        )
        filtered_rmse = _metric_float(
            metrics, f"{side}_force_estimator_filtered_3d_rmse_N"
        )
        lines.append(
            f"INFO {side}_force_estimator_tracking "
            f"active_samples={active_samples:.9g} "
            f"raw_3d_rmse_N={raw_rmse:.9g} "
            f"filtered_3d_rmse_N={filtered_rmse:.9g}"
        )

    recovery_metrics = (
        ("post_force_final_1s_torso_reference_rms_deg", 1.0e-2),
        ("post_force_final_1s_left_arm_residual_rms_mm", 1.0e-2),
        ("post_force_final_1s_right_arm_residual_rms_mm", 1.0e-2),
    )
    for key, threshold in recovery_metrics:
        value = _metric_float(metrics, key)
        if not math.isfinite(value):
            lines.append(f"WARN {key}=unavailable")
        else:
            lines.append(
                f"{'PASS' if value <= threshold else 'FAIL'} {key}={value:.9g} "
                f"threshold={threshold:.9g}"
            )

    if run.force_window is None:
        lines.append("WARN force_window unavailable")
    else:
        lines.append(
            f"PASS force_window_s={run.force_window[0]:.9g},{run.force_window[1]:.9g}"
        )
        for key in COMPLETE_EVIDENCE_LOGS:
            log = run.logs.get(key)
            if log is None:
                continue
            delay = (
                run.causal_delay_s
                if key in ("torso", "jacobian", "wbc_torso", "wbc_hand")
                else 0.0
            )
            expected_start = run.force_window[0] + delay
            expected_end = run.force_window[1] + delay
            covers = log.time[0] <= expected_start and log.time[-1] >= expected_end
            lines.append(
                f"{'PASS' if covers else 'FAIL'} {key}_covers_force_window="
                f"{log.time[0]:.9g},{log.time[-1]:.9g} "
                f"required={expected_start:.9g},{expected_end:.9g}"
            )
    return lines


def _metric_float(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key, float("nan"))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _admittance_config_text(metrics: Mapping[str, object]) -> str:
    scale = _metric_float(metrics, "crg_admittance_scale")
    stiffness = _metric_float(
        metrics, "crg_arm_ka_translation_n_per_m"
    )
    limit_enabled = _metric_float(
        metrics, "crg_admittance_translation_limit_enabled"
    )
    limit_m = _metric_float(
        metrics, "crg_admittance_translation_limit_m"
    )
    force_at_limit = _metric_float(
        metrics, "crg_arm_force_at_translation_limit_n"
    )
    parts: List[str] = []
    if math.isfinite(scale):
        parts.append(f"admittance scale={scale:g}")
    if math.isfinite(stiffness):
        parts.append(f"Ka,xyz={stiffness:g} N/m")
    if math.isfinite(limit_enabled) and limit_enabled > 0.5:
        if math.isfinite(limit_m):
            parts.append(f"pre-allocation |delta_xc,xyz|≤{1000.0 * limit_m:g} mm")
        else:
            parts.append("pre-allocation delta_xc xyz limit enabled")
    elif math.isfinite(limit_enabled):
        parts.append("pre-allocation delta_xc xyz limit disabled")
    if (
        math.isfinite(limit_enabled)
        and limit_enabled > 0.5
        and math.isfinite(force_at_limit)
    ):
        parts.append(f"steady per-axis F_limit={force_at_limit:g} N")
    return "; ".join(parts)


def _display_label(value: str) -> str:
    match = re.match(
        r"^(.*?)[_-]r(?:ho|ou)(?:[_=])?([-+]?(?:\d+(?:\.\d*)?|\.\d+))$",
        value.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return value
    base = match.group(1).replace("_", " ").strip(" -_")
    return f"{base}, ρ={match.group(2)}" if base else f"ρ={match.group(2)}"


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _shade_force_window(axis: plt.Axes, run: ComplianceRun) -> None:
    if run.force_window is None:
        return
    start, end = run.force_window
    axis.axvspan(start, end, color=FORCE_WINDOW_COLOR, alpha=0.12, zorder=-10)
    axis.axvline(start, color="#777777", linestyle=":", linewidth=0.8)
    axis.axvline(end, color="#777777", linestyle=":", linewidth=0.8)


def _figure_header(
    figure: plt.Figure,
    title: str,
    subtitle: str,
    top: float = 0.91,
) -> None:
    figure.suptitle(title, fontsize=15, fontweight="semibold", y=0.985)
    figure.text(
        0.5,
        0.952,
        subtitle,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#55585E",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, top))


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> List[Path]:
    written: List[Path] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 300
        figure.savefig(path, **kwargs)
        written.append(path)
    plt.close(figure)
    return written


def _plot_mujoco_applied_force(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    applied = run.logs.get("applied")
    required = (
        "l_fx", "l_fy", "l_fz",
        "r_fx", "r_fy", "r_fz",
    )
    if applied is None or not applied.has(*required):
        return []
    indices = _plot_indices(applied.time, max_points, time_range)
    figure, axes = plt.subplots(
        2, 3, figsize=(16, 8), sharex=True, sharey=True
    )
    for row, (side, side_name) in enumerate((("l", "Left"), ("r", "Right"))):
        force = applied.columns(
            tuple(f"{side}_f{axis}" for axis in ("x", "y", "z"))
        )
        for column, (axis_name, color) in enumerate(
            zip(("Fx", "Fy", "Fz"), AXIS_COLORS)
        ):
            axis = axes[row, column]
            axis.plot(
                applied.time[indices],
                force[indices, column],
                color=color,
                linewidth=1.8,
            )
            axis.axhline(0.0, color="#777777", linewidth=0.7, zorder=-5)
            axis.set_title(f"{side_name} wrist — {axis_name}")
            if column == 0:
                axis.set_ylabel("Force [N]")
            if row == 1:
                axis.set_xlabel("Time [s]")
            _shade_force_window(axis, run)
            _style_axis(axis)

    plotted_force = applied.columns(required)[indices]
    force_limit = max(1.0e-6, 1.05 * float(np.max(np.abs(plotted_force))))
    for axis in axes.flat:
        axis.set_ylim(-force_limit, force_limit)

    left_peak = _metric_float(metrics, "left_force_peak_N")
    right_peak = _metric_float(metrics, "right_force_peak_N")
    peak_text = (
        f"force-window 3D peak L/R = {left_peak:.4g}/{right_peak:.4g} N"
        if math.isfinite(left_peak) and math.isfinite(right_peak)
        else "force-window peak unavailable"
    )
    _figure_header(
        figure,
        "MuJoCo applied external force (ground truth) — "
        f"{_display_label(run.label)}",
        "Known force written through xfrc_applied; world/LWA xyz components; "
        f"{peak_text}",
        top=0.89,
    )
    return _save_figure(
        figure, output_dir, "01_mujoco_applied_external_force", formats
    )


def _plot_crg_allocation(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    torso = run.logs["torso"]
    required = tuple(f"xbfinal{idx}" for idx in range(3, 6))
    if not torso.has(*required):
        return []
    torso_angles = torso.columns(required)
    indices = _plot_indices(torso.time, max_points, time_range)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    independent_ab = False
    plotted_sides = 0

    for row, side in enumerate(("l", "r")):
        target_names = tuple(f"{side}_dxf{idx}" for idx in range(3))
        residual_names = tuple(f"{side}_arm{idx}" for idx in range(3))
        if not torso.has(*target_names, *residual_names):
            continue
        plotted_sides += 1
        target = torso.columns(target_names)
        residual = torso.columns(residual_names)
        contribution = _ab_contribution(
            run, torso.time, side, torso_angles
        )
        independent_ab = independent_ab or contribution is not None
        if contribution is None:
            contribution = target - residual
        reconstruction = contribution[:, :3] + residual

        for column, axis_name in enumerate(("x", "y", "z")):
            axis = axes[row, column]
            axis.plot(
                torso.time[indices],
                1000.0 * target[indices, column],
                color=REFERENCE_COLOR,
                linewidth=1.8,
                label="total target δx_c",
            )
            axis.plot(
                torso.time[indices],
                1000.0 * contribution[indices, column],
                color=TORSO_COLOR,
                linestyle="--",
                linewidth=1.4,
                label="torso contribution A_b δx_b",
            )
            axis.plot(
                torso.time[indices],
                1000.0 * residual[indices, column],
                color=ARM_COLOR,
                linestyle="-.",
                linewidth=1.4,
                label="hand contribution δx_a",
            )
            axis.plot(
                torso.time[indices],
                1000.0 * reconstruction[indices, column],
                color=RECONSTRUCTION_COLOR,
                linestyle=":",
                linewidth=1.2,
                label="torso + hand",
            )
            axis.set_title(
                f"{'Left' if side == 'l' else 'Right'} hand — {axis_name}"
            )
            if column == 0:
                axis.set_ylabel("Translation [mm]")
            if row == 1:
                axis.set_xlabel("Time [s]")
            _shade_force_window(axis, run)
            _style_axis(axis)

    if plotted_sides == 0:
        plt.close(figure)
        return []

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=4,
            fontsize=9,
        )
    left_error = _metric_float(
        metrics, "left_allocation_reconstruction_max_mm"
    )
    right_error = _metric_float(
        metrics, "right_allocation_reconstruction_max_mm"
    )
    source_text = "logged kinematic A_b" if independent_ab else "inferred contribution"
    _figure_header(
        figure,
        "CRG total compliance allocation: torso + hand — "
        f"{_display_label(run.label)}",
        "Selected translational allocation: total δx_c = torso A_bδx_b + "
        "hand δx_a; "
        f"{source_text}; max reconstruction error L/R = "
        f"{left_error:.3g}/{right_error:.3g} mm",
        top=0.865,
    )
    return _save_figure(
        figure, output_dir, "02_crg_total_torso_hand_allocation", formats
    )


def _plot_crg_torso_reference(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    torso = run.logs["torso"]
    raw_names = ("xb3", "xb4", "xb5")
    filtered_names = ("xbf3", "xbf4", "xbf5")
    final_names = ("xbfinal3", "xbfinal4", "xbfinal5")
    if not torso.has(*raw_names, *filtered_names, *final_names):
        return []
    raw = np.rad2deg(torso.columns(raw_names))
    filtered = np.rad2deg(torso.columns(filtered_names))
    final = np.rad2deg(torso.columns(final_names))
    indices = _plot_indices(torso.time, max_points, time_range)
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for component, (axis, name, color) in enumerate(
        zip(axes, ("Roll", "Pitch", "Yaw"), AXIS_COLORS)
    ):
        axis.plot(
            torso.time[indices], raw[indices, component],
            color=color, linestyle=":", linewidth=1.0, label="raw QP",
        )
        axis.plot(
            torso.time[indices], filtered[indices, component],
            color=color, linestyle="--", linewidth=1.3, label="filtered",
        )
        axis.plot(
            torso.time[indices], final[indices, component],
            color=color, linewidth=1.8, label="final reference",
        )
        axis.set_ylabel(f"{name} [deg]")
        _shade_force_window(axis, run)
        _style_axis(axis)
    axes[-1].set_xlabel("Time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        fontsize=8,
    )
    qp = _metric_float(metrics, "qp_success_percent")
    rho = _metric_float(metrics, "rho")
    rho_text = f"ρ={rho:g}" if math.isfinite(rho) else "ρ unavailable"
    config_text = _admittance_config_text(metrics)
    subtitle_parts = [rho_text]
    if config_text:
        subtitle_parts.append(config_text)
    subtitle_parts.append(
        f"QP success during causal nonzero-wrench window = {qp:.5g}%"
    )
    _figure_header(
        figure,
        f"CRG torso orientation reference — {_display_label(run.label)}",
        "; ".join(subtitle_parts),
        top=0.865,
    )
    return _save_figure(
        figure, output_dir, "03_crg_torso_reference", formats
    )


def _plot_wbc_torso_tracking(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    log = run.logs.get("wbc_torso")
    required = (
        "nom_roll", "nom_pitch", "nom_yaw",
        "des_roll", "des_pitch", "des_yaw",
        "cur_roll", "cur_pitch", "cur_yaw",
        "err_rotvec_x", "err_rotvec_y", "err_rotvec_z",
    )
    if log is None or not log.has(*required):
        return []
    desired_relative, current_relative = _relative_torso_offsets(log)
    desired_deg = np.rad2deg(desired_relative)
    current_deg = np.rad2deg(current_relative)
    error_deg = np.rad2deg(
        log.columns(("err_rotvec_x", "err_rotvec_y", "err_rotvec_z"))
    )
    error_norm_deg = np.linalg.norm(error_deg, axis=1)
    indices = _plot_indices(log.time, max_points, time_range)

    crg_reference = None
    torso = run.logs.get("torso")
    if torso is not None and torso.has("xbfinal3", "xbfinal4", "xbfinal5"):
        crg_reference = np.rad2deg(
            _interp_columns(
                torso, ("xbfinal3", "xbfinal4", "xbfinal5"), log.time
            )
        )

    figure, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for component, (axis, name, color) in enumerate(
        zip(axes[:3], ("Roll", "Pitch", "Yaw"), AXIS_COLORS)
    ):
        if crg_reference is not None:
            axis.plot(
                log.time[indices], crg_reference[indices, component],
                color=REFERENCE_COLOR, linestyle=":", linewidth=1.2,
                label="CRG reference",
            )
        axis.plot(
            log.time[indices], desired_deg[indices, component],
            color=color, linestyle="--", linewidth=1.5, label="WBC desired",
        )
        axis.plot(
            log.time[indices], current_deg[indices, component],
            color=color, linewidth=1.8, label="measured",
        )
        axis.set_ylabel(f"{name} [deg]")
        _shade_force_window(axis, run)
        _style_axis(axis)

    for component, (name, color) in enumerate(
        zip(("x", "y", "z"), AXIS_COLORS)
    ):
        axes[3].plot(
            log.time[indices], error_deg[indices, component],
            color=color, linewidth=1.0, alpha=0.8, label=f"error {name}",
        )
    axes[3].plot(
        log.time[indices], error_norm_deg[indices],
        color=REFERENCE_COLOR, linewidth=1.8, label="3D error norm",
    )
    axes[3].set_ylabel("Rotvec error [deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper right", ncol=4, fontsize=8)
    _shade_force_window(axes[3], run)
    _style_axis(axes[3])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=3,
            fontsize=8,
        )

    rmse = _metric_float(metrics, "wbc_torso_3d_rmse_deg")
    maximum = _metric_float(metrics, "wbc_torso_3d_max_error_deg")
    reference_rmse = _metric_float(
        metrics, "crg_to_wbc_torso_reference_rmse_deg"
    )
    _figure_header(
        figure,
        f"WBC torso orientation tracking — {_display_label(run.label)}",
        "Orientation is shown relative to the time-varying nominal torso; "
        "measured attitude is reconstructed from the logged SO(3) error; "
        f"causal-window 3D RMSE/max = {rmse:.4g}/{maximum:.4g} deg; "
        f"CRG→WBC reference RMSE = {reference_rmse:.3g} deg",
        top=0.865,
    )
    return _save_figure(
        figure, output_dir, "03_wbc_torso_orientation_tracking", formats
    )


def _plot_wbc_hand_position_tracking(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    log = run.logs.get("wbc_hand_position")
    required = tuple(
        f"{side}_{kind}_{axis}"
        for side in ("l", "r")
        for kind in ("ref", "ach")
        for axis in ("x", "y", "z")
    )
    if log is None or not log.has(*required):
        return []
    indices = _plot_indices(log.time, max_points, time_range)
    figure, axes = plt.subplots(
        3, 2, figsize=(14, 10), sharex=True, sharey="row"
    )
    row_values: List[List[np.ndarray]] = [[], [], []]
    for column, (side, side_name) in enumerate(
        (("l", "Left"), ("r", "Right"))
    ):
        reference = 1000.0 * log.columns(
            tuple(f"{side}_ref_{axis}" for axis in ("x", "y", "z"))
        )
        achieved = 1000.0 * log.columns(
            tuple(f"{side}_ach_{axis}" for axis in ("x", "y", "z"))
        )
        for row, (axis_name, color) in enumerate(
            zip(("x", "y", "z"), AXIS_COLORS)
        ):
            row_values[row].extend(
                (reference[indices, row], achieved[indices, row])
            )
            axis = axes[row, column]
            axis.plot(
                log.time[indices],
                reference[indices, row],
                color=REFERENCE_COLOR,
                linestyle="--",
                linewidth=1.5,
                label="CRG arm residual",
            )
            axis.plot(
                log.time[indices],
                achieved[indices, row],
                color=color,
                linewidth=1.5,
                label="measured arm displacement",
            )
            axis.set_title(f"{side_name} hand — {axis_name}")
            if column == 0:
                axis.set_ylabel("Arm-relative displacement [mm]")
            if row == 2:
                axis.set_xlabel("Time [s]")
            _shade_force_window(axis, run)
            _style_axis(axis)

    for row, values_for_row in enumerate(row_values):
        if not values_for_row:
            continue
        maximum = max(
            float(np.max(np.abs(values))) for values in values_for_row
        )
        limit = max(1.0e-3, 1.05 * maximum)
        for axis in axes[row, :]:
            axis.set_ylim(-limit, limit)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=2,
            fontsize=8,
        )
    left_rmse = _metric_float(
        metrics, "wbc_left_hand_3d_position_rmse_mm"
    )
    right_rmse = _metric_float(
        metrics, "wbc_right_hand_3d_position_rmse_mm"
    )
    left_nrmse = _metric_float(
        metrics, "wbc_left_hand_3d_position_nrmse_percent"
    )
    right_nrmse = _metric_float(
        metrics, "wbc_right_hand_3d_position_nrmse_percent"
    )
    _figure_header(
        figure,
        "WBC residual-hand Cartesian position tracking — "
        f"{_display_label(run.label)}",
        "CRG residual reference vs measured arm-only wrist displacement; "
        f"causal-window 3D RMSE L/R = {left_rmse:.4g}/{right_rmse:.4g} mm; "
        f"normalized = {left_nrmse:.3g}%/{right_nrmse:.3g}%",
        top=0.865,
    )
    return _save_figure(
        figure, output_dir, "04_wbc_hand_position_tracking", formats
    )


def _plot_wbc_hand_tracking(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    log = run.logs.get("wbc_hand")
    required = tuple(
        f"{side}_{kind}{index}"
        for side in ("l", "r")
        for kind in ("ref", "ach")
        for index in range(3)
    )
    if log is None or not log.has(*required):
        return []
    indices = _plot_indices(log.time, max_points, time_range)
    figure, axes = plt.subplots(
        3, 2, figsize=(14, 10), sharex=True, sharey="row"
    )
    row_values: List[List[np.ndarray]] = [[], [], []]
    for column, (side, side_name) in enumerate((("l", "Left"), ("r", "Right"))):
        reference_names = tuple(f"{side}_ref{idx}" for idx in range(3))
        achieved_names = tuple(f"{side}_ach{idx}" for idx in range(3))
        if not log.has(*reference_names, *achieved_names):
            continue
        reference = log.columns(reference_names)
        achieved = log.columns(achieved_names)
        for row, (axis_name, color) in enumerate(zip(("x", "y", "z"), AXIS_COLORS)):
            row_values[row].extend(
                (reference[indices, row], achieved[indices, row])
            )
            axis = axes[row, column]
            axis.plot(
                log.time[indices], reference[indices, row],
                color=REFERENCE_COLOR, linestyle="--", linewidth=1.5,
                label="reference",
            )
            axis.plot(
                log.time[indices], achieved[indices, row],
                color=color, linewidth=1.5, label="QP achieved",
            )
            axis.set_title(f"{side_name} hand — {axis_name}")
            if column == 0:
                axis.set_ylabel("Acceleration [m/s²]")
            if row == 2:
                axis.set_xlabel("Time [s]")
            _shade_force_window(axis, run)
            _style_axis(axis)

    for row, values_for_row in enumerate(row_values):
        if not values_for_row:
            continue
        maximum = max(
            float(np.max(np.abs(values))) for values in values_for_row
        )
        limit = max(1.0e-6, 1.05 * maximum)
        for axis in axes[row, :]:
            axis.set_ylim(-limit, limit)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=2,
            fontsize=8,
        )

    left_rmse = _metric_float(metrics, "wbc_left_hand_3d_acc_rmse_mps2")
    right_rmse = _metric_float(metrics, "wbc_right_hand_3d_acc_rmse_mps2")
    left_nrmse = _metric_float(
        metrics, "wbc_left_hand_3d_acc_nrmse_percent"
    )
    right_nrmse = _metric_float(
        metrics, "wbc_right_hand_3d_acc_nrmse_percent"
    )
    _figure_header(
        figure,
        "WBC hand operational-space acceleration tracking — "
        f"{_display_label(run.label)}",
        "Selected xyz WBC task: reference vs QP-achieved acceleration; "
        "measured Cartesian displacement is shown separately; "
        "causal-window 3D RMSE "
        f"L/R = {left_rmse:.4g}/{right_rmse:.4g} m/s²; "
        f"normalized = {left_nrmse:.3g}%/{right_nrmse:.3g}%",
        top=0.865,
    )
    return _save_figure(
        figure, output_dir, "05_wbc_hand_acceleration_tracking", formats
    )


def _plot_end_to_end_overview(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    torso = run.logs["torso"]
    indices = _plot_indices(torso.time, max_points, time_range)
    figure, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

    applied = run.logs.get("applied")
    if applied is not None and applied.has(
        "l_fx", "l_fy", "l_fz", "r_fx", "r_fy", "r_fz"
    ):
        left_force = _interp_columns(
            applied, ("l_fx", "l_fy", "l_fz"), torso.time
        )
        right_force = _interp_columns(
            applied, ("r_fx", "r_fy", "r_fz"), torso.time
        )
        for force, side_name, linestyle in (
            (left_force, "left", "-"),
            (right_force, "right", "--"),
        ):
            for component, (axis_name, color) in enumerate(
                zip(("Fx", "Fy", "Fz"), AXIS_COLORS)
            ):
                if np.nanmax(np.abs(force[:, component])) <= 1.0e-10:
                    continue
                axes[0].plot(
                    torso.time[indices],
                    force[indices, component],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"{side_name} {axis_name}",
                )
    axes[0].set_ylabel("Signed force [N]")
    if axes[0].lines:
        axes[0].legend(loc="upper left", ncol=4, fontsize=8)

    for side, color, linestyle, side_name in (
        ("l", LEFT_COLOR, "-", "left"),
        ("r", RIGHT_COLOR, "--", "right"),
    ):
        target_names = tuple(f"{side}_dxf{idx}" for idx in range(3))
        residual_names = tuple(f"{side}_arm{idx}" for idx in range(3))
        if torso.has(*target_names):
            target_norm = 1000.0 * np.linalg.norm(
                torso.columns(target_names), axis=1
            )
            axes[1].plot(
                torso.time[indices], target_norm[indices], color=color,
                linewidth=1.2, alpha=0.55, label=f"{side_name} total target",
            )
        if torso.has(*residual_names):
            residual_norm = 1000.0 * np.linalg.norm(
                torso.columns(residual_names), axis=1
            )
            axes[1].plot(
                torso.time[indices], residual_norm[indices], color=color,
                linestyle=linestyle, linewidth=1.8,
                label=f"{side_name} arm residual",
            )
    axes[1].set_ylabel("CRG translation [mm]")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc="upper left", ncol=2, fontsize=8)

    if torso.has("xbfinal3", "xbfinal4", "xbfinal5"):
        torso_reference = np.rad2deg(
            torso.columns(("xbfinal3", "xbfinal4", "xbfinal5"))
        )
        for component, (name, color) in enumerate(
            zip(("roll", "pitch", "yaw"), AXIS_COLORS)
        ):
            axes[2].plot(
                torso.time[indices], torso_reference[indices, component],
                color=color, linestyle="--", linewidth=1.5,
                label=f"CRG {name}",
            )
    wbc_torso = run.logs.get("wbc_torso")
    relative_rpy_names = (
        "nom_roll", "nom_pitch", "nom_yaw",
        "des_roll", "des_pitch", "des_yaw",
        "cur_roll", "cur_pitch", "cur_yaw",
    )
    if wbc_torso is not None and wbc_torso.has(*relative_rpy_names):
        _, actual_relative = _relative_torso_offsets(wbc_torso)
        actual_deg = np.rad2deg(actual_relative)
        actual_at_torso = np.column_stack(
            [
                np.interp(
                    torso.time, wbc_torso.time, actual_deg[:, component],
                    left=np.nan, right=np.nan,
                )
                for component in range(3)
            ]
        )
        for component, (name, color) in enumerate(
            zip(("roll", "pitch", "yaw"), AXIS_COLORS)
        ):
            axes[2].plot(
                torso.time[indices], actual_at_torso[indices, component],
                color=color, linewidth=1.3, alpha=0.8,
                label=f"measured {name}",
            )
    axes[2].set_ylabel("Torso offset [deg]")
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend(loc="upper left", ncol=3, fontsize=8)

    if wbc_torso is not None and wbc_torso.has(
        "err_rotvec_x", "err_rotvec_y", "err_rotvec_z"
    ):
        torso_error = np.rad2deg(
            wbc_torso.columns(("err_rotvec_x", "err_rotvec_y", "err_rotvec_z"))
        )
        torso_error_norm = np.linalg.norm(torso_error, axis=1)
        error_at_torso = np.interp(
            torso.time, wbc_torso.time, torso_error_norm,
            left=np.nan, right=np.nan,
        )
        axes[3].plot(
            torso.time[indices], error_at_torso[indices],
            color=TORSO_COLOR, linewidth=1.7, label="torso rotvec error norm",
        )
    axes[3].set_ylabel("Torso error [deg]")
    if axes[3].get_legend_handles_labels()[0]:
        axes[3].legend(loc="upper left", fontsize=8)

    wbc_hand = run.logs.get("wbc_hand")
    if wbc_hand is not None:
        for side, color, linestyle, side_name in (
            ("l", LEFT_COLOR, "-", "left"),
            ("r", RIGHT_COLOR, "--", "right"),
        ):
            reference_names = tuple(f"{side}_ref{idx}" for idx in range(3))
            achieved_names = tuple(f"{side}_ach{idx}" for idx in range(3))
            if not wbc_hand.has(*reference_names, *achieved_names):
                continue
            error_norm = np.linalg.norm(
                wbc_hand.columns(achieved_names)
                - wbc_hand.columns(reference_names),
                axis=1,
            )
            error_at_torso = np.interp(
                torso.time, wbc_hand.time, error_norm,
                left=np.nan, right=np.nan,
            )
            axes[4].plot(
                torso.time[indices], error_at_torso[indices], color=color,
                linestyle=linestyle, linewidth=1.5,
                label=f"{side_name} hand QP acceleration error",
            )
    axes[4].set_ylabel("Hand task acc. error [m/s²]")
    axes[4].set_xlabel("Time [s]")
    if axes[4].get_legend_handles_labels()[0]:
        axes[4].legend(loc="upper left", ncol=2, fontsize=8)

    for axis in axes:
        _shade_force_window(axis, run)
        _style_axis(axis)

    rho = _metric_float(metrics, "rho")
    qp = _metric_float(metrics, "qp_success_percent")
    ab_valid = _metric_float(metrics, "Ab_valid_percent")
    rho_text = f"ρ={rho:g}" if math.isfinite(rho) else "ρ unavailable"
    config_text = _admittance_config_text(metrics)
    subtitle_parts = [rho_text]
    if config_text:
        subtitle_parts.append(config_text)
    subtitle_parts.append(
        f"nonzero-wrench-window QP/Ab valid = {qp:.5g}%/{ab_valid:.5g}%"
    )
    _figure_header(
        figure,
        "External wrench → CRG allocation → WBC tracking — "
        f"{_display_label(run.label)}",
        "; ".join(subtitle_parts),
        top=0.92,
    )
    return _save_figure(
        figure, output_dir, "06_end_to_end_compliance_summary", formats
    )


def _plot_wrist_force_validation(
    run: ComplianceRun,
    output_dir: Path,
    formats: Sequence[str],
    max_points: int,
    time_range: Optional[Sequence[float]],
    metrics: Mapping[str, object],
) -> List[Path]:
    log = run.logs.get("wrist_validation")
    if log is None:
        return []
    required = tuple(
        f"{side}_{kind}_{axis}"
        for side in ("l", "r")
        for kind in ("gt", "est", "estf")
        for axis in ("fx", "fy", "fz")
    )
    if not log.has(*required):
        return []
    indices = _plot_indices(log.time, max_points, time_range)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    plotted_values: List[np.ndarray] = []
    for row, (side, side_name) in enumerate((("l", "Left"), ("r", "Right"))):
        prefix = "left" if side == "l" else "right"
        for column, (axis_name, color) in enumerate(
            zip(("fx", "fy", "fz"), AXIS_COLORS)
        ):
            axis = axes[row, column]
            ground_truth = log.column(f"{side}_gt_{axis_name}")
            raw_estimate = log.column(f"{side}_est_{axis_name}")
            filtered_estimate = log.column(f"{side}_estf_{axis_name}")
            plotted_values.extend(
                (
                    ground_truth[indices],
                    raw_estimate[indices],
                    filtered_estimate[indices],
                )
            )
            axis.plot(
                log.time[indices], ground_truth[indices],
                color=REFERENCE_COLOR, linewidth=1.7, label="ground truth",
            )
            axis.plot(
                log.time[indices], raw_estimate[indices],
                color=color, linewidth=0.9, alpha=0.5, label="raw estimate",
            )
            axis.plot(
                log.time[indices], filtered_estimate[indices],
                color=color, linestyle="--", linewidth=1.4,
                label="filtered estimate",
            )
            axis.axhline(0.0, color="#777777", linewidth=0.7, zorder=-5)
            axis.set_title(f"{side_name} wrist — {axis_name.upper()}")
            if column == 0:
                axis.set_ylabel("Force [N]")
            if row == 1:
                axis.set_xlabel("Time [s]")
            _shade_force_window(axis, run)
            _style_axis(axis)
            filtered_axis_rmse = _metric_float(
                metrics,
                f"{prefix}_force_estimator_filtered_{axis_name[1]}_rmse_N",
            )
            if math.isfinite(filtered_axis_rmse):
                annotation_at_top = float(np.mean(ground_truth[indices])) < 0.0
                axis.text(
                    0.02,
                    0.95 if annotation_at_top else 0.05,
                    f"filtered RMSE = {filtered_axis_rmse:.3g} N",
                    transform=axis.transAxes,
                    ha="left",
                    va="top" if annotation_at_top else "bottom",
                    fontsize=8,
                    color="#55585E",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                        "pad": 1.5,
                    },
                )

    estimator_limit = max(
        1.0e-6,
        1.05 * max(float(np.max(np.abs(values))) for values in plotted_values),
    )
    for axis in axes.flat:
        axis.set_ylim(-estimator_limit, estimator_limit)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=3,
            fontsize=8,
        )

    left_raw = _metric_float(
        metrics, "left_force_estimator_raw_3d_rmse_N"
    )
    left_filtered = _metric_float(
        metrics, "left_force_estimator_filtered_3d_rmse_N"
    )
    right_raw = _metric_float(
        metrics, "right_force_estimator_raw_3d_rmse_N"
    )
    right_filtered = _metric_float(
        metrics, "right_force_estimator_filtered_3d_rmse_N"
    )
    rmse_values = (left_raw, left_filtered, right_raw, right_filtered)
    rmse_text = (
        "logged-sample 3D RMSE raw→filtered L/R = "
        f"{left_raw:.3g}→{left_filtered:.3g} / "
        f"{right_raw:.3g}→{right_filtered:.3g} N"
        if all(math.isfinite(value) for value in rmse_values)
        else "logged-sample estimator RMSE unavailable"
    )
    _figure_header(
        figure,
        "Estimated external force vs MuJoCo ground truth — "
        f"{_display_label(run.label)}",
        "World/LWA xyz; "
        f"{rmse_text}; estimator update physically follows the applied sample "
        "by one control step",
        top=0.855,
    )
    return _save_figure(
        figure, output_dir, "06_estimated_external_force", formats
    )


def generate_compliance_evidence(
    folder: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    *,
    label: Optional[str] = None,
    formats: Sequence[str] = ("png", "pdf"),
    max_points: int = 0,
    time_range: Optional[Sequence[float]] = None,
    force_window: Optional[Sequence[float]] = None,
    strict: bool = False,
) -> Tuple[ComplianceRun, Dict[str, object], List[Path]]:
    run = load_compliance_run(
        folder, label=label, force_window_override=force_window
    )
    output_path = Path(output_dir).expanduser().resolve()
    missing = [key for key in COMPLETE_EVIDENCE_LOGS if key not in run.logs]
    missing_columns = _missing_required_columns(run)
    problems = [LOG_FILENAMES[key] for key in missing]
    problems.extend(
        f"{LOG_FILENAMES[key]}({','.join(names)})"
        for key, names in missing_columns.items()
    )
    if strict and problems:
        raise ValueError(
            "Incomplete evidence log/schema: " + "; ".join(problems)
        )

    metrics = compute_compliance_metrics(run)

    if missing or missing_columns:
        message = "Incomplete evidence log/schema: " + "; ".join(problems)
        print(f"[WARN] {message}; available figures will still be generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.staging-",
        dir=str(output_path.parent),
    ) as staging_name:
        staging_path = Path(staging_name)
        _write_metrics(run, metrics, staging_path)
        staged_written: List[Path] = []
        with plt.rc_context(PLOT_STYLE):
            staged_written.extend(
                _plot_mujoco_applied_force(
                    run, staging_path, formats, max_points, time_range, metrics
                )
            )
            staged_written.extend(
                _plot_crg_allocation(
                    run, staging_path, formats, max_points, time_range, metrics
                )
            )
            staged_written.extend(
                _plot_wbc_torso_tracking(
                    run, staging_path, formats, max_points, time_range, metrics
                )
            )
            staged_written.extend(
                _plot_wbc_hand_position_tracking(
                    run, staging_path, formats, max_points, time_range, metrics
                )
            )
            staged_written.extend(
                _plot_wbc_hand_tracking(
                    run, staging_path, formats, max_points, time_range, metrics
                )
            )
            staged_written.extend(
                _plot_wrist_force_validation(
                    run, staging_path, formats, max_points, time_range, metrics
                )
            )
        _commit_known_artifacts(
            staging_path,
            output_path,
            SINGLE_CLEANUP_FIGURE_STEMS,
            ("crg_wbc_metrics.csv", "crg_wbc_metrics.json", "crg_wbc_qa.txt"),
        )
        written = [output_path / path.name for path in staged_written]

    print(
        f"[INFO] Generated {len(written)} CRG/WBC figure files in {output_path}"
    )
    return run, metrics, written


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return "run" if cleaned in ("", ".", "..") else cleaned


def _comparison_bar_groups(
    axis: plt.Axes,
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    ylabel: str,
    title: str,
    annotate: bool = True,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(1, len(series))
    finite_values = [
        float(value)
        for _, values, _ in series
        for value in values
        if math.isfinite(float(value))
    ]
    value_scale = max(finite_values, default=0.0)
    for index, (series_name, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2.0) * width
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            label=series_name,
            color=color,
            edgecolor="#404348",
            linewidth=0.5,
        )
        if annotate and len(labels) <= 8:
            # Matplotlib 3.3 (used by the robot environment) predates
            # Axes.bar_label, so keep an explicit compatible annotation.
            for bar, value in zip(bars, values):
                if not math.isfinite(float(value)):
                    continue
                if value_scale > 0.0 and abs(float(value)) <= value_scale * 1.0e-4:
                    continue
                axis.annotate(
                    f"{float(value):.3g}",
                    xy=(bar.get_x() + 0.5 * bar.get_width(), bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if value_scale > 0.0:
        axis.set_ylim(0.0, 1.14 * value_scale)
    else:
        axis.set_ylim(bottom=0.0)
    axis.legend(fontsize=8, ncol=max(1, min(3, len(series))))
    _style_axis(axis)


def _write_comparison_metrics(
    metrics_rows: Sequence[Mapping[str, object]], output_dir: Path
) -> None:
    keys: List[str] = []
    for row in metrics_rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with (output_dir / "crg_wbc_comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in keys})
    (output_dir / "crg_wbc_comparison_metrics.json").write_text(
        json.dumps(
            [
                {key: _json_safe(value) for key, value in row.items()}
                for row in metrics_rows
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _rho_pair_identity(run: ComplianceRun) -> Optional[Tuple[str, int]]:
    for candidate in (run.label, run.folder.name):
        match = re.match(
            r"^(.*?)[_-]r(?:ho|ou)(?:[_=])?([01](?:\.0+)?)$",
            candidate.strip(),
            flags=re.IGNORECASE,
        )
        if match is not None:
            return match.group(1).strip("_- ").lower(), int(float(match.group(2)))
    return None


def _command_without_rho(run: ComplianceRun) -> Optional[Tuple[str, ...]]:
    command = run.metadata.get("command")
    if not command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    normalized: List[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in ("--crg-rho", "--crg-rou", "--rho"):
            skip_next = True
            continue
        if token.startswith(("--crg-rho=", "--crg-rou=", "--rho=")):
            continue
        normalized.append(token)
    return tuple(normalized)


def _check_paired_inputs(
    runs: Sequence[ComplianceRun],
    metrics_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    grouped: Dict[str, Dict[int, int]] = {}
    qa_lines = [
        "CRG/WBC rho-pair input QA",
        "rho effects are paired by enable timing and signed applied force; recorded commands must also match except for rho",
        "if command metadata is unavailable, force/timing pairing remains available and command status is WARN",
        "wrist-origin equivalent torque is reported separately because its lever arm changes with robot state",
    ]
    for index, run in enumerate(runs):
        identity = _rho_pair_identity(run)
        if identity is None:
            continue
        group, rho_index = identity
        if rho_index in grouped.setdefault(group, {}):
            qa_lines.append(
                f"FAIL {group} duplicate_rho={rho_index} labels="
                f"{runs[grouped[group][rho_index]].label},{run.label}"
            )
            continue
        grouped[group][rho_index] = index

    pair_count = 0
    for group, members in sorted(grouped.items()):
        if 0 not in members or 1 not in members:
            qa_lines.append(
                f"WARN {group} incomplete_pair available_rho="
                + ",".join(str(value) for value in sorted(members))
            )
            continue
        pair_count += 1
        first_index, second_index = members[0], members[1]
        first = runs[first_index].logs.get("applied")
        second = runs[second_index].logs.get("applied")
        time_difference = float("nan")
        enable_difference = float("nan")
        force_difference = float("nan")
        torque_difference = float("nan")
        position_difference = float("nan")
        first_command = _command_without_rho(runs[first_index])
        second_command = _command_without_rho(runs[second_index])
        command_available = first_command is not None and second_command is not None
        command_match = command_available and first_command == second_command
        matched = False
        if (
            first is not None
            and second is not None
            and first.has(*APPLIED_WRENCH_COLUMNS)
            and second.has(*APPLIED_WRENCH_COLUMNS)
            and first.has("l_enabled", "r_enabled")
            and second.has("l_enabled", "r_enabled")
            and first.time.shape == second.time.shape
        ):
            time_difference = float(np.max(np.abs(first.time - second.time)))
            enable_difference = float(
                np.max(
                    np.abs(
                        first.columns(("l_enabled", "r_enabled"))
                        - second.columns(("l_enabled", "r_enabled"))
                    )
                )
            )
            first_wrench = first.columns(APPLIED_WRENCH_COLUMNS)
            second_wrench = second.columns(APPLIED_WRENCH_COLUMNS)
            wrench_difference = np.abs(first_wrench - second_wrench)
            force_difference = float(
                np.max(wrench_difference[:, (0, 1, 2, 6, 7, 8)])
            )
            torque_difference = float(
                np.max(wrench_difference[:, (3, 4, 5, 9, 10, 11)])
            )
            position_names = (
                "l_px", "l_py", "l_pz", "r_px", "r_py", "r_pz"
            )
            if first.has(*position_names) and second.has(*position_names):
                position_difference = float(
                    np.max(
                        np.abs(
                            first.columns(position_names)
                            - second.columns(position_names)
                        )
                    )
                )
            matched = (
                time_difference <= 1.0e-12
                and enable_difference <= 1.0e-12
                and force_difference <= 1.0e-12
                and (command_match or not command_available)
            )

        for index in (first_index, second_index):
            metrics_rows[index]["paired_input_group"] = group
            metrics_rows[index]["paired_input_match"] = matched
            metrics_rows[index]["paired_command_except_rho_match"] = command_match
            metrics_rows[index]["paired_input_time_max_diff_s"] = time_difference
            metrics_rows[index]["paired_enable_max_diff"] = enable_difference
            metrics_rows[index]["paired_input_force_max_diff_N"] = force_difference
            metrics_rows[index][
                "paired_wrist_origin_torque_max_diff_Nm"
            ] = torque_difference
            metrics_rows[index][
                "paired_application_position_max_diff_m"
            ] = position_difference
        command_status = "PASS" if command_match else "WARN"
        qa_lines.append(
            f"{command_status} {group} command_except_rho_match={command_match}"
        )
        qa_lines.append(
            f"{'PASS' if matched else 'FAIL'} {group} rho0_vs_rho1_applied_force "
            f"time_max_diff_s={time_difference:.9g} "
            f"enable_max_diff={enable_difference:.9g} "
            f"force_max_diff_N={force_difference:.9g}"
        )
        qa_lines.append(
            f"INFO {group} state_dependent_wrist_origin_geometry "
            f"application_position_max_diff_m={position_difference:.9g} "
            f"equivalent_torque_max_diff_Nm={torque_difference:.9g}"
        )

    if pair_count == 0:
        qa_lines.append(
            "WARN no rho0/rho1 label pairs found; use names such as roll_rho0 and roll_rho1"
        )
    (output_dir / "crg_wbc_comparison_qa.txt").write_text(
        "\n".join(qa_lines) + "\n", encoding="utf-8"
    )


def _plot_comparison(
    runs: Sequence[ComplianceRun],
    metrics_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    labels = [_display_label(run.label) for run in runs]

    def values(key: str) -> List[float]:
        return [_metric_float(row, key) for row in metrics_rows]

    figure, axes = plt.subplots(2, 3, figsize=(19, 11))
    _comparison_bar_groups(
        axes[0, 0], labels,
        (
            ("roll", values("torso_roll_peak_deg"), AXIS_COLORS[0]),
            ("pitch", values("torso_pitch_peak_deg"), AXIS_COLORS[1]),
            ("yaw", values("torso_yaw_peak_deg"), AXIS_COLORS[2]),
        ),
        "Peak angle [deg]", "CRG torso reference peak",
    )
    _comparison_bar_groups(
        axes[0, 1], labels,
        (
            ("left", values("left_arm_residual_peak_mm"), LEFT_COLOR),
            ("right", values("right_arm_residual_peak_mm"), RIGHT_COLOR),
        ),
        "Peak residual [mm]", "CRG arm residual translation",
    )
    _comparison_bar_groups(
        axes[0, 2], labels,
        (("torso", values("wbc_torso_3d_rmse_deg"), TORSO_COLOR),),
        "3D RMSE [deg]", "WBC torso tracking",
    )
    _comparison_bar_groups(
        axes[1, 0], labels,
        (
            ("left", values("wbc_left_hand_3d_acc_rmse_mps2"), LEFT_COLOR),
            ("right", values("wbc_right_hand_3d_acc_rmse_mps2"), RIGHT_COLOR),
        ),
        "3D RMSE [m/s²]", "WBC hand task acceleration error",
    )
    _comparison_bar_groups(
        axes[1, 1], labels,
        (
            ("QP solved", values("qp_success_percent"), "#0072B2"),
            ("A_b valid", values("Ab_valid_percent"), "#E69F00"),
        ),
        "Success [%]", "CRG numerical quality",
        annotate=False,
    )
    axes[1, 1].set_ylim(0.0, 105.0)
    _comparison_bar_groups(
        axes[1, 2], labels,
        (
            (
                "left",
                values("left_allocation_reconstruction_max_mm"),
                LEFT_COLOR,
            ),
            (
                "right",
                values("right_allocation_reconstruction_max_mm"),
                RIGHT_COLOR,
            ),
        ),
        "Max error [mm]", "Selected translational Eq. (13) closure",
        annotate=False,
    )
    _figure_header(
        figure,
        "CRG/WBC compliance experiment comparison",
        "Metrics use causally aligned nonzero-wrench windows; compare ρ only within the same load (see QA)",
        top=0.91,
    )
    return _save_figure(
        figure, output_dir, "crg_wbc_run_comparison", formats
    )


def _plot_hand_task_comparison(
    runs: Sequence[ComplianceRun],
    metrics_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    labels = [_display_label(run.label) for run in runs]

    def values(key: str) -> List[float]:
        return [_metric_float(row, key) for row in metrics_rows]

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    _comparison_bar_groups(
        axes[0],
        labels,
        (
            ("left", values("wbc_left_hand_3d_acc_rmse_mps2"), LEFT_COLOR),
            ("right", values("wbc_right_hand_3d_acc_rmse_mps2"), RIGHT_COLOR),
        ),
        "3D RMSE [m/s²]",
        "Absolute acceleration-task error",
    )
    _comparison_bar_groups(
        axes[1],
        labels,
        (
            (
                "left",
                values("wbc_left_hand_3d_acc_nrmse_percent"),
                LEFT_COLOR,
            ),
            (
                "right",
                values("wbc_right_hand_3d_acc_nrmse_percent"),
                RIGHT_COLOR,
            ),
        ),
        "NRMSE [% of reference RMS]",
        "Normalized acceleration-task error",
    )
    _figure_header(
        figure,
        "WBC hand-task realization diagnostic",
        "Selected xyz reference versus QP-achieved acceleration; this is not measured hand-pose tracking",
        top=0.86,
    )
    return _save_figure(
        figure, output_dir, "crg_wbc_hand_task_comparison", formats
    )


def generate_compliance_batch(
    folders: Sequence[os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
    *,
    labels: Optional[Sequence[str]] = None,
    formats: Sequence[str] = ("png", "pdf"),
    max_points: int = 0,
    time_range: Optional[Sequence[float]] = None,
    force_window: Optional[Sequence[float]] = None,
    strict: bool = False,
) -> Tuple[List[ComplianceRun], List[Dict[str, object]], List[Path]]:
    if len(folders) < 2:
        raise ValueError("At least two comparison folders are required")
    if labels is not None and len(labels) != len(folders):
        raise ValueError("--labels count must match --compare folder count")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    runs: List[ComplianceRun] = []
    metrics_rows: List[Dict[str, object]] = []
    written: List[Path] = []
    resolved_labels = [
        labels[index] if labels is not None else Path(folder).name
        for index, folder in enumerate(folders)
    ]
    safe_labels = [_safe_label(label) for label in resolved_labels]
    if len(set(safe_labels)) != len(safe_labels):
        raise ValueError(
            "Comparison labels must be unique after filename sanitization"
        )

    run_outputs = [
        (output_path / safe_label).resolve() for safe_label in safe_labels
    ]
    for label, run_output in zip(resolved_labels, run_outputs):
        if run_output.parent != output_path:
            raise ValueError(f"Unsafe comparison label path: {label}")

    for index, folder in enumerate(folders):
        label = resolved_labels[index]
        run_output = run_outputs[index]
        run, metrics, run_written = generate_compliance_evidence(
            folder,
            run_output,
            label=label,
            formats=formats,
            max_points=max_points,
            time_range=time_range,
            force_window=force_window,
            strict=strict,
        )
        runs.append(run)
        metrics_rows.append(metrics)
        written.extend(run_written)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.comparison-staging-",
        dir=str(output_path.parent),
    ) as staging_name:
        staging_path = Path(staging_name)
        _check_paired_inputs(runs, metrics_rows, staging_path)
        _write_comparison_metrics(metrics_rows, staging_path)
        staged_written: List[Path] = []
        with plt.rc_context(PLOT_STYLE):
            staged_written.extend(
                _plot_comparison(runs, metrics_rows, staging_path, formats)
            )
            staged_written.extend(
                _plot_hand_task_comparison(
                    runs, metrics_rows, staging_path, formats
                )
            )
        _commit_known_artifacts(
            staging_path,
            output_path,
            BATCH_FIGURE_STEMS,
            (
                "crg_wbc_comparison_metrics.csv",
                "crg_wbc_comparison_metrics.json",
                "crg_wbc_comparison_qa.txt",
            ),
        )
        written.extend(
            output_path / path.name for path in staged_written
        )
    print(
        f"[INFO] Generated comparison for {len(runs)} runs in {output_path}"
    )
    return runs, metrics_rows, written
