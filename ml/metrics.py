"""Shared regression metrics for age prediction."""

import numpy as np

AGE_BANDS = [
    (1, 12), (13, 19), (20, 29), (30, 39), (40, 49),
    (50, 59), (60, 69), (70, 79), (80, 100),
]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    cs2 = np.mean(abs_errors <= 2) * 100  # % of predictions within 2 years
    cs3 = np.mean(abs_errors <= 3) * 100  # % of predictions within 3 years
    cs5 = np.mean(abs_errors <= 5) * 100  # % of predictions within 5 years

    mean_signed_error = errors.mean()  # >0 means model over-predicts age on average, <0 under-predicts

    return {"mae": mae, "rmse": rmse, "r2": r2, "cs2": cs2, "cs3": cs3, "cs5": cs5,
            "mean_signed_error": mean_signed_error}


def mae_by_age(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_errors = np.abs(y_pred - y_true)

    result = {}
    for age in sorted(np.unique(y_true)):
        mask = y_true == age
        result[int(age)] = {"mae": float(abs_errors[mask].mean()), "n": int(mask.sum())}
    return result


def mae_by_band(y_true, y_pred, bands=AGE_BANDS):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    result = {}
    for lo, hi in bands:
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() == 0:
            continue
        result[f"{lo}-{hi}"] = {
            "mae": float(abs_errors[mask].mean()),
            "mean_signed_error": float(errors[mask].mean()),  # bias direction for this band
            "n": int(mask.sum()),
        }
    return result


def age_range_label(predicted_age, half_width=4):
    lo = max(0, round(predicted_age - half_width))
    hi = round(predicted_age + half_width)
    return f"{lo}-{hi}"
