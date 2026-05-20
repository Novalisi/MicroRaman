import re
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import rampy as rp
from pybaselines import Baseline
from scipy.signal import find_peaks
import csv
from collections import Counter
from typing import Iterable
import pickle
from scipy.optimize import minimize


DB_ROOT = Path(r"/work/SLoPP_and_SLoPP-E")

def extract_laser_id(filename: str) -> str:
    # Searching in the name of files the type of laser in order to apply the correction
    match = re.search(r'(532|633|785)nm', filename)
    if match:
        return match.group(1)
    return "default"


def estimate_si_shift(files: list[Path]) -> float:
    # Find the strongest peak between 500-540 in Si files and align to 520
    # Checking the type of laser in order to better correct the shift
    
    peaks_by_laser: dict[str, list[float]] = {}
    
    for path in files:
        if not is_si_file(path):
            continue
            
        laser = extract_laser_id(path.name)
        x, y = read_spectrum(path)
        
        mask = (x >= 500) & (x <= 540)
        if not np.any(mask):
            continue
            
        x_roi = x[mask]
        y_roi = y[mask]
        peak_idx = int(np.argmax(y_roi))
        
        if laser not in peaks_by_laser:
            peaks_by_laser[laser] = []
        peaks_by_laser[laser].append(float(x_roi[peak_idx]))
        
    shifts: dict[str, float] = {}
    for laser, peaks in peaks_by_laser.items():
        peak_mean = float(np.mean(peaks))
        shifts[laser] = 520.0 - peak_mean
        
    return shifts

def baseline_correct(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Compute baseline with arPLS/arPLSe and return both baseline and corrected signal
    # I'm using pybaselines that is a traduction in python of a matlab code
    # To better understand arPLS: https://doi.org/10.1039/C4AN01061B 
    model = Baseline(x)
    if hasattr(model, "arplse"):
        baseline, _ = model.arplse(y)
    else:
        baseline, _ = model.arpls(y)
    return baseline, y - baseline


def smooth_whittaker(
    x: np.ndarray,
    y: np.ndarray,
    lam: float = 1e3,
    order: int = 3,
) -> np.ndarray:
    # Whittaker smoothing using rampy, a package for Raman spectroscopy
    # In order to better understand: https://pubs.acs.org/doi/10.1021/ac034173t
    try:
        y_smooth = rp.smooth(x, y, method="whittaker", Lambda=lam, order=order)
        return np.asarray(y_smooth)
    except Exception as exc:
        raise RuntimeError(f"Whittaker smoothing failed: {exc}")

def read_spectrum(path: Path, x_shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    # Load two-column spectrum (x, y) from text file, with optional x shift
    data = np.genfromtxt(path, comments="#", delimiter=None)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    if data.shape[1] < 2:
        raise ValueError(f"File {path.name} does not have at least 2 columns")
    x = data[:, 0] + x_shift
    y = data[:, 1]
    return x, y


def base_name(path: Path) -> str:
    # Strip trailing _NN suffix so parts of the same spectrum share one key
    match = re.match(r"(.+)_\d+\.txt$", path.name)
    return match.group(1) if match else path.stem


def is_si_file(path: Path) -> bool:
    # Identify Si calibration spectra
    return base_name(path).lower().startswith("si")


def group_files_by_base(files: list[Path]) -> dict[str, list[Path]]:
    # Group files by base name (same spectrum, different parts)
    # The spectra of a measure has the same base name and a numerical suffix (_01/_02/_03)
    grouped: dict[str, list[Path]] = {}
    for path in files:
        key = base_name(path)
        grouped.setdefault(key, []).append(path)
    return grouped

def compute_overlap_offset(
	merged_x: np.ndarray,
	merged_y: np.ndarray,
	x: np.ndarray,
	y: np.ndarray,
	tolerance: float,
) -> float:
	# Estimate vertical offset using overlapping x values within tolerance, in order to prevent double count or incongruence
	if merged_x.size == 0 or x.size == 0:
		return 0.0

	indices = np.searchsorted(merged_x, x)
	diffs = []
	for i, idx in enumerate(indices):
		candidates = []
		if idx < merged_x.size:
			candidates.append(idx)
		if idx > 0:
			candidates.append(idx - 1)
		if not candidates:
			continue
		best = min(candidates, key=lambda j: abs(merged_x[j] - x[i]))
		if abs(merged_x[best] - x[i]) <= tolerance:
			diffs.append(merged_y[best] - y[i])
	if not diffs:
		return 0.0
	return float(np.mean(diffs))


def merge_and_dedup(
    merged_x: np.ndarray,
    merged_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Merge spectra and remove duplicate x by binning within tolerance
    x_all = np.concatenate([merged_x, x])
    y_all = np.concatenate([merged_y, y])
    order = np.argsort(x_all)
    x_sorted = x_all[order]
    y_sorted = y_all[order]

    x_binned = np.round(x_sorted / tolerance) * tolerance
    acc_y: dict[float, float] = {}
    acc_x: dict[float, float] = {}
    counts: dict[float, int] = {}
    for xb, xi, yi in zip(x_binned, x_sorted, y_sorted):
        acc_y[xb] = acc_y.get(xb, 0.0) + yi
        acc_x[xb] = acc_x.get(xb, 0.0) + xi
        counts[xb] = counts.get(xb, 0) + 1
    
    keys = sorted(acc_y.keys())
    x_avg = np.array([acc_x[k] / counts[k] for k in keys], dtype=float)
    y_avg = np.array([acc_y[k] / counts[k] for k in keys], dtype=float)
    return x_avg, y_avg


def concatenate_spectra(
    paths: list[Path],
    tolerance: float = 0.5,
    global_shift: dict | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")

    spectra = []
    for path in paths:
        if isinstance(global_shift, dict):
            laser = extract_laser_id(path.name)
            current_shift = global_shift.get(laser, 0.0)
        else:
            current_shift = global_shift
            
        shift = 0.0 if is_si_file(path) else current_shift

        x, y = read_spectrum(path, x_shift=shift)
        order = np.argsort(x)
        spectra.append((x[order], y[order]))

    # Process from low to high x to keep a consistent reference
    spectra.sort(key=lambda pair: pair[0][0])
    merged_x, merged_y = spectra[0]

    for x, y in spectra[1:]:
        offset = compute_overlap_offset(merged_x, merged_y, x, y, tolerance)
        y_aligned = y + offset
        merged_x, merged_y = merge_and_dedup(
            merged_x, merged_y, x, y_aligned, tolerance
        )

    return merged_x, merged_y

def preprocess_xy_custom(x: np.ndarray, y: np.ndarray, apply_smooth: bool) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 3:
        raise ValueError("Not enough points")

    mask = x >= 200
    x = x[mask]
    y = y[mask]

    baseline, y_corr = baseline_correct(x, y)
    
    if apply_smooth:
        y_final = smooth_whittaker(x, y_corr)
    else:
        y_final = y_corr
    
    return x, y_final

def load_query_spectra(QUERY_DIR: Path, apply_smooth:bool) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    # Load spectra with Si correction and concatenation
    files = sorted(QUERY_DIR.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {QUERY_DIR}")

    global_shift = estimate_si_shift(files)
    grouped = group_files_by_base(files)

    for key, paths in grouped.items():
        if is_si_file(Path(key)):
            continue

        x, y = concatenate_spectra(paths, tolerance=0.5, global_shift=global_shift)

        try:
            x, y = preprocess_xy_custom(x, y, apply_smooth)
        except Exception as exc:
            print(f"Skip query {key}: {exc}")
            continue

        yield key, x, y

def load_db_spectra(apply_smooth:bool) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    # Loading of the databases txt spectra
	paths = sorted(DB_ROOT.rglob("*.txt"))
	if not paths:
		raise FileNotFoundError(f"No .txt files found under {DB_ROOT}")

	items: list[tuple[Path, np.ndarray, np.ndarray]] = []
	for path in paths:
		try:
			x, y = read_spectrum(path)
			order = np.argsort(x)
			x = x[order]
			y = y[order]
			x, y = preprocess_xy_custom(x, y, apply_smooth)
		except Exception as exc:
			print(f"Skip db {path.name}: {exc}")
			continue
		items.append((path, x, y))
	return items


def resample_overlap(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:

    start = max(float(x1[0]), float(x2[0]))
    end = min(float(x1[-1]), float(x2[-1]))

    if end <= start:
        return None

    dx1 = np.mean(np.diff(x1))
    dx2 = np.mean(np.diff(x2))

    dx = min(dx1, dx2)
    grid = np.arange(start, end, 2.0)

    y1i = np.interp(grid, x1, y1)
    y2i = np.interp(grid, x2, y2)

    return y1i, y2i, grid


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Cosine similarity, for reference: https://doi.org/10.1016/j.saa.2025.126702
    # I used also the "Correlation distance" that is also called "Pearson similarity" 
	denom = float(np.linalg.norm(a) * np.linalg.norm(b))
	if denom == 0.0:
		return 0.0
	return float(np.dot(a, b) / denom)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    # Pearson similarity, the difference of before is the sottraction of every point of the array with respect to
        # the mean
	a = a - float(np.mean(a))
	b = b - float(np.mean(b))
	denom = float(np.linalg.norm(a) * np.linalg.norm(b))
	if denom == 0.0:
		return 0.0
	return float(np.dot(a, b) / denom)


def plot_rank_histograms(rows: list[list[str]]) -> None:
    # Counting the number of material of the same type, in the same sample and plot it in a histogram
    # Not used at the moment
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)
	for rank in range(1, TOP_N + 1):
		rank_rows = [row for row in rows if row[1] == str(rank)]
		if not rank_rows:
			continue
		labels = [Path(row[3]).name for row in rank_rows]
		counts = Counter(labels)
		sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
		names = [item[0] for item in sorted_items]
		values = [item[1] for item in sorted_items]

		plt.figure(figsize=(10, 5))
		plt.bar(names, values, color="#1f77b4")
		plt.title(f"Rank {rank} histogram")
		plt.xlabel("Directory")
		plt.ylabel("Count")
		plt.xticks(rotation=60, ha="right")
		plt.grid(axis="y", linestyle="--", alpha=0.7)
		plt.tight_layout()
		out_path = PLOTS_DIR / f"rank_{rank}.png"
		plt.savefig(out_path, dpi=150)
		plt.close()


def get_db_spectra_cache(apply_smooth:bool):
    # Saving the database in a cache files in order to make the script faster
    if Path("/work/db_cache.pkl").exists():
        print("Cache charging")
        with open("/work/db_cache.pkl", "rb") as f:
            return pickle.load(f)
    
    items = load_db_spectra(True)

    with open("/work/db_cache.pkl", "wb") as f:
        pickle.dump(items, f)
    
    return items


def square_root_transform(yd: np.ndarray, yq: np.ndarray):
    # Square Root Transformation
    yq_proc = np.sqrt(np.clip(yq, 0, None))
    yd_proc = np.sqrt(np.clip(yd, 0, None))


    #yq_proc = np.where(yq_proc < 0.1 * np.max(yq_proc), 0, yq_proc)
    #yd_proc = np.where(yd_proc < 0.1 * np.max(yd_proc), 0, yd_proc)

    # L2 normalization, also used in the paper "Raman spectra comparison"

    return yd_proc, yq_proc

def power_transformation(yd: np.ndarray, yq: np.ndarray):
    # Power transformation
    yq_p = np.power(np.clip(yq, 0, None), 1.2)
    yd_p = np.power(np.clip(yd, 0, None), 1.2)
    
    #yq_p = np.where(yq_p < 0.02 * np.max(yq_p), 0, yq_p)
    #yd_p = np.where(yd_p < 0.02 * np.max(yd_p), 0, yd_p)

    # L2 normalization, also used in the paper "Raman spectra comparison"


    return yd_p, yq_p


def nn_en_mixture_analysis(y, X, component_names=None, lam=0.01, alpha=0.96):
# Mixture analysis NN-EN:  https://doi.org/10.1002/cem.3293
    y = np.array(y, dtype=float)
    X = np.array(X, dtype=float)
    N, M = X.shape
    
    if component_names is None:
        component_names = [f"Polimero_{i+1}" for i in range(M)]
        
    def objective(r):
        residual = np.linalg.norm(y - np.dot(X, r), ord=2)
        l1_penalty = np.sum(np.abs(r))
        l2_penalty = np.linalg.norm(r, ord=2)
        
        return residual + lam * (alpha * l1_penalty + (1 - alpha) * l2_penalty)

    #  r >= 0
    bounds = [(0, None) for _ in range(M)]
    
    r0 = np.ones(M) * 0.01
    
    res = minimize(objective, r0, bounds=bounds, method='L-BFGS-B')
    
    if not res.success:
        print("[Sorgente] Attenzione: L'ottimizzazione non ha raggiunto una convergenza perfetta.")
        
    estimated_coefficients = res.x
    
    total_signal = np.sum(estimated_coefficients)
    if total_signal > 0:
        percentages = (estimated_coefficients / total_signal) * 100
    else:
        percentages = np.zeros(M)
        
    output_results = []
    for i in range(M):
        output_results.append({
            "name": component_names[i],
            "coefficient": estimated_coefficients[i],
            "percentage": percentages[i]
        })
        
    # best result first
    output_results.sort(key=lambda x: x['coefficient'], reverse=True)
    
    print("\n" + "="*65)
    print("      RISULTATI DECONVOLUZIONE MISCELA POLIMERICA (NN-EN)      ")
    print("="*65)
    print(f"{'Polimero / Componente':<30} | {'Concentrazione (Coeff)':<15} | {'Contributo %':<12}")
    print("-"*65)
    
    for item in output_results:
        if item['coefficient'] > 1e-4:
            print(f"{item['name']:<30} | {item['coefficient']:<22.4f} | {item['percentage']:<10.2f}%")
            
    print("="*65)
    
    return output_results
