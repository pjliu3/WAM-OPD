"""PWTR exclusive prefixes over selected temporal ranks and their supplied q."""
from collections import defaultdict
from collections.abc import Mapping
import math
from numbers import Real


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _identifier(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise ValueError(f"{name} must be a nonempty string or integer")
    return value


def _log_ratio(prefix, eta):
    current, initial, teacher = prefix
    # Subtract a common prefix anchor BEFORE adding log mixture coefficients:
    # adding a small log(eta) to a huge negative prefix first loses that offset.
    anchor = max(initial, teacher)
    first = math.log(eta) + (initial - anchor)
    second = math.log1p(-eta) + (teacher - anchor)
    peak = max(first, second)
    offset = peak + math.log(math.exp(first - peak) + math.exp(second - peak))
    return (current - anchor) - offset


def reweight_records(records, probabilities, *, strength=.1, eta=2 / 3, cap=2.):
    """Add the fixed PWTR weights to a complete selected-point inventory.

    Each record must provide record_id, trajectory_id, decision_index, task_name,
    condition, behavior_version, phi_current, phi_initial, and phi_teacher.
    phi_* are per-decision scores from all three models on the same paths.
    The input may be in any order; output retains that order. For each model,
    Phi[k] = sum(phi[j] for j < k) over selected temporal ranks. The public
    wrapper maps selected original decision indices to consecutive ranks only
    for this aggregation; omitted environment decisions do not contribute.

    log_weight = strength * (Phi_current - logsumexp(log(eta)+Phi_initial,
                              log(1-eta)+Phi_teacher)).
    The release fixes strength=.1, eta=2/3 and cap=2. The behavior_version never
    chooses which reference scores enter this formula.

    probabilities is the explicit q=1/(N*K_tau) from the selected-point view.
    Its support must exactly match that view's scored record IDs. Each (task_name,
    condition) group conditionalizes this supplied q, normalizes exp(log_weight),
    clips to cap, then normalizes again. Final weights can exceed cap.
    """
    strength, eta, cap = (_finite(strength, "strength"), _finite(eta, "eta"), _finite(cap, "cap"))
    if (strength, eta, cap) != (.1, 2 / 3, 2.):
        raise ValueError("PWTR fixes strength=.1, eta=2/3 and cap=2")
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("probabilities must be a nonempty sampled-view mapping")
    q = {rid: _finite(value, "probability") for rid, value in probabilities.items()}
    if any(value <= 0 for value in q.values()) or not math.isclose(math.fsum(q.values()), 1., rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("sampled-view probabilities must be positive and sum to one")
    required = ("record_id", "trajectory_id", "decision_index", "task_name", "condition", "behavior_version",
                "phi_current", "phi_initial", "phi_teacher")
    result, episodes, groups, ids = [], defaultdict(list), defaultdict(list), set()
    for original in records:
        if not isinstance(original, Mapping) or any(key not in original for key in required):
            raise ValueError("each scored record requires full metadata and all three model scores")
        row = dict(original)
        for key in ("record_id", "trajectory_id", "task_name"):
            _identifier(row[key], key)
        rid, tid, step = row["record_id"], row["trajectory_id"], row["decision_index"]
        if rid in ids:
            raise ValueError("record_id must be unique across the complete selected-point view")
        ids.add(rid)
        if type(step) is not int or step < 0:
            raise ValueError("decision_index must be a nonnegative integer")
        if row["condition"] not in ("clean", "rand") or row["behavior_version"] not in ("current_base", "teacher"):
            raise ValueError("unknown condition or behavior_version")
        for key in ("phi_current", "phi_initial", "phi_teacher"):
            row[key] = _finite(row[key], key)
        result.append(row)
        episodes[tid].append(row)
        groups[row["task_name"], row["condition"]].append(row)
    if not result or ids != set(q):
        raise ValueError("scored records must exactly cover the supplied selected-point inventory")
    for episode in episodes.values():
        episode.sort(key=lambda row: row["decision_index"])
        if [row["decision_index"] for row in episode] != list(range(len(episode))):
            raise ValueError("trajectory decisions must be unique and contiguous from zero")
        if len({(row["task_name"], row["condition"], row["behavior_version"]) for row in episode}) != 1:
            raise ValueError("trajectory task, condition, and behavior metadata must remain constant")
        if any("trajectory_length" in row and row["trajectory_length"] != len(episode) for row in episode):
            raise ValueError("scored trajectory is missing records declared by trajectory_length")
        prefix = [0., 0., 0.]
        for index, row in enumerate(episode):
            row.update(zip(("Phi_current", "Phi_initial", "Phi_teacher"), prefix))
            row["log_weight"] = _finite(strength * _log_ratio(prefix, eta), "log_weight")
            if index < len(episode) - 1:
                prefix = [_finite(total + row[key], "prefix score")
                          for total, key in zip(prefix, ("phi_current", "phi_initial", "phi_teacher"))]
    for rows in groups.values():
        mass = math.fsum(q[row["record_id"]] for row in rows)
        for row in rows:
            row["probability"] = q[row["record_id"]]
            row["conditional_probability"] = row["probability"] / mass
        maximum = max(row["log_weight"] for row in rows)
        scaled = [math.exp(row["log_weight"] - maximum) for row in rows]
        normalizer = math.fsum(row["conditional_probability"] * value for row, value in zip(rows, scaled))
        for row, value in zip(rows, scaled):
            row["preclip_weight"] = _finite(value / normalizer, "preclip_weight")
            row["weight"] = min(row["preclip_weight"], cap)
        clipped_normalizer = math.fsum(row["conditional_probability"] * row["weight"] for row in rows)
        for row in rows:
            row["weight"] = _finite(row["weight"] / clipped_normalizer, "weight")
    return result
