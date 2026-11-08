from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.evaluation.core.ranking import chance_metrics, normalize, ranking_metrics
from src.evaluation.observability.samples import PositionSample, surrogate_matrix
from src.evaluation.observability.surrogates import (
    SURROGATE_TASKS,
    SurrogateTask,
    matching_matrix,
    surrogate_class_labels,
)


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.where(std > 0, std, 1.0)


def choose_matched_candidate(
    anchor: np.ndarray,
    candidates: np.ndarray,
    rng: np.random.Generator,
    *,
    top_k: int = 20,
) -> int:
    if len(candidates) == 0:
        raise ValueError("candidates cannot be empty")
    distances = np.linalg.norm(
        np.asarray(candidates, dtype=np.float64) - np.asarray(anchor), axis=1
    )
    count = max(1, min(top_k, len(distances)))
    pool = np.argsort(distances)[:count]
    return int(pool[int(rng.integers(len(pool)))])


@dataclass(frozen=True)
class RankingCase:
    query: int
    candidates: tuple[int, ...]
    positive_position: int


def evaluate_ranking(
    embeddings: np.ndarray, cases: Sequence[RankingCase]
) -> dict[str, float]:
    """Top-1, MRR, and mean rank for fixed-size candidate sets."""
    if not cases:
        return {
            "top1": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "n_queries": 0,
            **chance_metrics(5),
        }
    values = normalize(embeddings)
    scores = []
    relevance = []
    for case in cases:
        candidates = np.asarray(case.candidates, dtype=np.int64)
        scores.append(values[candidates] @ values[case.query])
        relevance.append({case.positive_position: 0})
    metrics = ranking_metrics(np.vstack(scores), relevance, ks=())
    return {
        "top1": metrics["top1"],
        "mrr": metrics["mrr"],
        "mean_rank": metrics["mean_rank"],
        "n_queries": metrics["n_queries"],
        **chance_metrics(len(cases[0].candidates)),
    }


def _choose_negatives(
    anchor: int,
    candidates: Sequence[int],
    surrogate_z: np.ndarray,
    rng: np.random.Generator,
    count: int,
    *,
    matched: bool,
    top_k: int = 20,
) -> list[int]:
    candidates = list(candidates)
    if not matched:
        return [int(value) for value in rng.choice(candidates, count, replace=False)]
    chosen = []
    available = list(candidates)
    for _ in range(count):
        local = choose_matched_candidate(
            surrogate_z[anchor],
            surrogate_z[available],
            rng,
            top_k=top_k,
        )
        chosen.append(int(available.pop(local)))
    return chosen


def _shuffle_case(
    query: int,
    candidates: list[int],
    positive: int,
    rng: np.random.Generator,
) -> RankingCase:
    order = rng.permutation(len(candidates))
    shuffled = [candidates[int(index)] for index in order]
    positive_position = int(np.flatnonzero(order == 0)[0])
    return RankingCase(query, tuple(shuffled), positive_position)


def build_theme_ranking_cases(
    samples: Sequence[PositionSample],
    theme: str,
    *,
    k: int = 5,
    max_queries: int = 200,
    matched: bool,
    surrogate_z: np.ndarray,
    seed: int = 0,
) -> list[RankingCase]:
    positives = [
        index for index, sample in enumerate(samples) if theme in sample.themes
    ]
    negatives = [
        index for index, sample in enumerate(samples) if theme not in sample.themes
    ]
    if len(positives) < 2 or len(negatives) < k - 1:
        return []
    rng = np.random.default_rng([seed, zlib.crc32(theme.encode("utf-8")), int(matched)])
    queries = rng.permutation(positives)[:max_queries]
    cases = []
    for query in queries:
        query = int(query)
        positive_pool = [
            index
            for index in positives
            if index != query and samples[index].puzzle_id != samples[query].puzzle_id
        ]
        negative_pool = [
            index
            for index in negatives
            if samples[index].puzzle_id != samples[query].puzzle_id
        ]
        if not positive_pool or len(negative_pool) < k - 1:
            continue
        positive = int(rng.choice(positive_pool))
        chosen_negatives = _choose_negatives(
            query,
            negative_pool,
            surrogate_z,
            rng,
            k - 1,
            matched=matched,
        )
        cases.append(_shuffle_case(query, [positive, *chosen_negatives], positive, rng))
    return cases


def build_temporal_ranking_cases(
    samples: Sequence[PositionSample],
    *,
    k: int = 5,
    max_queries: int = 200,
    matched: bool,
    surrogate_z: np.ndarray,
    seed: int = 0,
    max_ply_distance: int = 5,
) -> list[RankingCase]:
    if max_ply_distance < 1:
        raise ValueError("max_ply_distance must be at least 1")
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        if sample.source == "test" and sample.game_id is not None:
            groups.setdefault(sample.game_id, []).append(index)
    query_pool = [
        index for group in groups.values() if len(group) >= 2 for index in group
    ]
    if not query_pool:
        return []
    rng = np.random.default_rng([seed, int(matched)])
    queries = rng.permutation(query_pool)[:max_queries]
    cases = []
    for query in queries:
        query = int(query)
        query_sample = samples[query]
        same_game = [index for index in groups[query_sample.game_id] if index != query]
        if not same_game:
            continue
        if query_sample.ply is not None:
            positive_pool = [
                index
                for index in same_game
                if abs((samples[index].ply or 0) - (query_sample.ply or 0))
                <= max_ply_distance
            ]
        else:
            positive_pool = same_game
        if not positive_pool:
            continue
        positive = int(rng.choice(positive_pool))
        negative_pool = [
            index
            for index, sample in enumerate(samples)
            if sample.source == "test"
            and sample.game_id is not None
            and sample.game_id != query_sample.game_id
        ]
        if len(negative_pool) < k - 1:
            continue
        chosen_negatives = _choose_negatives(
            query,
            negative_pool,
            surrogate_z,
            rng,
            k - 1,
            matched=matched,
        )
        cases.append(_shuffle_case(query, [positive, *chosen_negatives], positive, rng))
    return cases


def _mean_ranking(rows: Sequence[dict], k: int = 5) -> dict[str, float]:
    if not rows:
        return {
            "top1": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "n_queries": 0,
            **chance_metrics(k),
        }
    keys = (
        "top1",
        "mrr",
        "mean_rank",
        "chance_top1",
        "chance_mrr",
        "chance_mean_rank",
    )
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    result["n_queries"] = int(sum(row["n_queries"] for row in rows))
    return result


def theme_ranking_metrics(
    embeddings: np.ndarray,
    samples: Sequence[PositionSample],
    themes: Sequence[str],
    *,
    k: int = 5,
    max_queries: int = 200,
    seed: int = 0,
) -> dict:
    surrogate_z = standardize_columns(surrogate_matrix(samples))
    per_theme: dict[str, dict] = {}
    for theme in themes:
        per_theme[theme] = {}
        for mode in ("random", "matched"):
            cases = build_theme_ranking_cases(
                samples,
                theme,
                k=k,
                max_queries=max_queries,
                matched=mode == "matched",
                surrogate_z=surrogate_z,
                seed=seed,
            )
            per_theme[theme][mode] = evaluate_ranking(embeddings, cases)
    random_rows = [
        per_theme[theme]["random"]
        for theme in themes
        if per_theme[theme]["random"]["n_queries"] > 0
    ]
    matched_rows = [
        per_theme[theme]["matched"]
        for theme in themes
        if per_theme[theme]["matched"]["n_queries"] > 0
    ]
    return {
        "per_theme": per_theme,
        "mean_random": _mean_ranking(random_rows, k),
        "mean_matched": _mean_ranking(matched_rows, k),
    }


def temporal_ranking_metrics(
    embeddings: np.ndarray,
    samples: Sequence[PositionSample],
    *,
    k: int = 5,
    max_queries: int = 200,
    seed: int = 0,
    max_ply_distance: int = 5,
) -> dict:
    surrogate_z = standardize_columns(surrogate_matrix(samples))
    result = {}
    for mode in ("random", "matched"):
        cases = build_temporal_ranking_cases(
            samples,
            k=k,
            max_queries=max_queries,
            matched=mode == "matched",
            surrogate_z=surrogate_z,
            seed=seed,
            max_ply_distance=max_ply_distance,
        )
        result[mode] = evaluate_ranking(embeddings, cases)
    return result


def _candidate_ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    for index, score in enumerate(scores):
        greater = float(np.sum(scores > score))
        ties = max(0.0, float(np.sum(scores == score)) - 1.0)
        ranks[index] = 1.0 + greater + 0.5 * ties
    return ranks


def _case_performance(values: np.ndarray, case: RankingCase) -> tuple[float, float]:
    scores = values[list(case.candidates)] @ values[case.query]
    ranks = _candidate_ranks(scores)
    positive_score = scores[case.positive_position]
    others = [
        score for index, score in enumerate(scores) if index != case.positive_position
    ]
    margin = positive_score - (max(others) if others else positive_score)
    return float(ranks[case.positive_position]), float(margin)


def _select_case_buckets(
    values: np.ndarray,
    cases: Sequence[RankingCase],
    *,
    max_per_bucket: int,
    rng: np.random.Generator,
) -> list[tuple[str, RankingCase]]:
    if not cases or max_per_bucket <= 0:
        return []
    performance = [(_case_performance(values, case), case) for case in cases]
    performance.sort(key=lambda item: (item[0][0], -item[0][1]))
    ordered = [case for _, case in performance]
    count = min(max_per_bucket, len(ordered))
    best = ordered[:count]
    remaining = ordered[count:]
    worst_count = min(max_per_bucket, len(remaining))
    worst = list(reversed(remaining[-worst_count:])) if worst_count else []
    middle = remaining[: len(remaining) - worst_count]
    random_count = min(max_per_bucket, len(middle))
    if random_count:
        indices = rng.choice(len(middle), size=random_count, replace=False)
        random_cases = [middle[int(index)] for index in indices]
    else:
        random_cases = []
    selected = []
    selected.extend(("best", case) for case in best)
    selected.extend(("random", case) for case in random_cases)
    selected.extend(("worst", case) for case in worst)
    return selected


def _case_payload(
    values: np.ndarray,
    samples: Sequence[PositionSample],
    case: RankingCase,
    *,
    group: str,
    matched: bool,
    theme: str | None,
    task: str | None = None,
    class_label: str | None = None,
    bucket: str | None = None,
) -> dict:
    candidates = list(case.candidates)
    scores = values[candidates] @ values[case.query]
    ranks = _candidate_ranks(scores)
    query_sample = samples[case.query]
    payload_candidates = []
    for index, candidate in enumerate(candidates):
        sample = samples[candidate]
        payload_candidates.append(
            {
                "sample_id": int(sample.sample_id),
                "fen": sample.fen,
                "score": float(scores[index]),
                "rank": float(ranks[index]),
                "is_positive": index == case.positive_position,
                "themes": list(sample.themes),
                "puzzle_id": sample.puzzle_id,
                "game_id": sample.game_id,
                "ply": sample.ply,
            }
        )
    return {
        "group": group,
        "task": task,
        "class_label": class_label,
        "theme": theme,
        "bucket": bucket,
        "matched": bool(matched),
        "query_sample_id": int(query_sample.sample_id),
        "query_fen": query_sample.fen,
        "query_themes": list(query_sample.themes),
        "query_game_id": query_sample.game_id,
        "query_ply": query_sample.ply,
        "positive_rank": float(ranks[case.positive_position]),
        "top1": bool(ranks[case.positive_position] <= 1.0),
        "candidates": payload_candidates,
    }


def theme_ranking_examples(
    embeddings: np.ndarray,
    samples: Sequence[PositionSample],
    themes: Sequence[str],
    *,
    k: int = 5,
    max_queries: int = 200,
    seed: int = 0,
    matched: bool = False,
    max_per_bucket: int = 3,
) -> list[dict]:
    surrogate_z = standardize_columns(surrogate_matrix(samples))
    values = normalize(embeddings)
    examples = []
    for theme in themes:
        cases = build_theme_ranking_cases(
            samples,
            theme,
            k=k,
            max_queries=max_queries,
            matched=matched,
            surrogate_z=surrogate_z,
            seed=seed,
        )
        rng = np.random.default_rng(
            [seed, zlib.crc32(theme.encode("utf-8")), int(matched), 17]
        )
        for bucket, case in _select_case_buckets(
            values, cases, max_per_bucket=max_per_bucket, rng=rng
        ):
            examples.append(
                _case_payload(
                    values,
                    samples,
                    case,
                    group=theme,
                    matched=matched,
                    theme=theme,
                    task="motif",
                    class_label=theme,
                    bucket=bucket,
                )
            )
    return examples


def temporal_ranking_examples(
    embeddings: np.ndarray,
    samples: Sequence[PositionSample],
    *,
    k: int = 5,
    max_queries: int = 200,
    seed: int = 0,
    matched: bool = False,
    max_per_bucket: int = 3,
    max_ply_distance: int = 5,
) -> list[dict]:
    surrogate_z = standardize_columns(surrogate_matrix(samples))
    cases = build_temporal_ranking_cases(
        samples,
        k=k,
        max_queries=max_queries,
        matched=matched,
        surrogate_z=surrogate_z,
        seed=seed,
        max_ply_distance=max_ply_distance,
    )
    values = normalize(embeddings)
    rng = np.random.default_rng([seed, int(matched), 23])
    return [
        _case_payload(
            values,
            samples,
            case,
            group="temporal",
            matched=matched,
            theme=None,
            task="temporal",
            class_label="temporal",
            bucket=bucket,
        )
        for bucket, case in _select_case_buckets(
            values, cases, max_per_bucket=max_per_bucket, rng=rng
        )
    ]


def _group_key(sample: PositionSample) -> str:
    if sample.puzzle_id is not None:
        return f"puzzle:{sample.puzzle_id}"
    if sample.game_id is not None:
        return f"game:{sample.game_id}"
    return f"sample:{sample.sample_id}"


def build_class_ranking_cases(
    samples: Sequence[PositionSample],
    labels: np.ndarray,
    class_label: str,
    *,
    k: int = 5,
    max_queries: int = 200,
    matched: bool,
    surrogate_z: np.ndarray,
    seed: int = 0,
) -> list[RankingCase]:
    positives = [index for index, value in enumerate(labels) if value == class_label]
    negatives = [index for index, value in enumerate(labels) if value != class_label]
    if len(positives) < 2 or len(negatives) < k - 1:
        return []
    rng = np.random.default_rng(
        [seed, zlib.crc32(str(class_label).encode("utf-8")), int(matched)]
    )
    queries = rng.permutation(positives)[:max_queries]
    groups = [_group_key(sample) for sample in samples]
    cases = []
    for query in queries:
        query = int(query)
        positive_pool = [
            index
            for index in positives
            if index != query and groups[index] != groups[query]
        ]
        negative_pool = [index for index in negatives if groups[index] != groups[query]]
        if not positive_pool or len(negative_pool) < k - 1:
            continue
        positive = int(rng.choice(positive_pool))
        chosen_negatives = _choose_negatives(
            query,
            negative_pool,
            surrogate_z,
            rng,
            k - 1,
            matched=matched,
        )
        cases.append(_shuffle_case(query, [positive, *chosen_negatives], positive, rng))
    return cases


def _mean_class_ranking(rows: Sequence[dict], k: int) -> dict[str, float]:
    return _mean_ranking(rows, k)


def surrogate_ranking_metrics(
    embeddings: np.ndarray,
    samples: Sequence[PositionSample],
    tasks: Sequence[SurrogateTask] = SURROGATE_TASKS,
    *,
    k: int = 5,
    max_queries_per_class: int = 200,
    seed: int = 0,
) -> dict:
    full_z = standardize_columns(surrogate_matrix(samples))
    per_task: dict[str, dict] = {}
    for task in tasks:
        labels = surrogate_class_labels(samples, task)
        loo_z = standardize_columns(matching_matrix(samples, exclude_key=task.key))
        per_class: dict[str, dict] = {}
        for class_label in task.classes:
            if not np.any(labels == class_label):
                continue
            row = {}
            for mode in ("random", "matched"):
                cases = build_class_ranking_cases(
                    samples,
                    labels,
                    class_label,
                    k=k,
                    max_queries=max_queries_per_class,
                    matched=mode == "matched",
                    surrogate_z=loo_z if mode == "matched" else full_z,
                    seed=seed + zlib.crc32(task.name.encode("utf-8")),
                )
                row[mode] = evaluate_ranking(embeddings, cases)
            per_class[class_label] = row
        random_rows = [
            row["random"]
            for row in per_class.values()
            if row["random"]["n_queries"] > 0
        ]
        matched_rows = [
            row["matched"]
            for row in per_class.values()
            if row["matched"]["n_queries"] > 0
        ]
        per_task[task.name] = {
            "label": task.label,
            "classes": list(task.classes),
            "per_class": per_class,
            "mean_random": _mean_class_ranking(random_rows, k),
            "mean_matched": _mean_class_ranking(matched_rows, k),
        }
    all_random = [task["mean_random"] for task in per_task.values()]
    all_matched = [task["mean_matched"] for task in per_task.values()]
    return {
        "per_task": per_task,
        "mean_random": _mean_class_ranking(
            [row for row in all_random if row["n_queries"] > 0], k
        ),
        "mean_matched": _mean_class_ranking(
            [row for row in all_matched if row["n_queries"] > 0], k
        ),
    }


def surrogate_ranking_examples(
    embeddings: np.ndarray,
    samples: Sequence[PositionSample],
    tasks: Sequence[SurrogateTask] = SURROGATE_TASKS,
    *,
    k: int = 5,
    max_queries_per_class: int = 200,
    seed: int = 0,
    modes: Sequence[str] = ("matched",),
    max_per_bucket: int = 3,
) -> dict[str, dict[str, dict[str, dict[str, list[dict]]]]]:
    full_z = standardize_columns(surrogate_matrix(samples))
    values = normalize(embeddings)
    result: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for task in tasks:
        labels = surrogate_class_labels(samples, task)
        loo_z = standardize_columns(matching_matrix(samples, exclude_key=task.key))
        task_result: dict[str, dict[str, list[dict]]] = {}
        for class_label in task.classes:
            if not np.any(labels == class_label):
                continue
            class_result: dict[str, list[dict]] = {}
            for mode in modes:
                cases = build_class_ranking_cases(
                    samples,
                    labels,
                    class_label,
                    k=k,
                    max_queries=max_queries_per_class,
                    matched=mode == "matched",
                    surrogate_z=loo_z if mode == "matched" else full_z,
                    seed=seed + zlib.crc32(task.name.encode("utf-8")),
                )
                rng = np.random.default_rng(
                    [
                        seed,
                        zlib.crc32(task.name.encode("utf-8")),
                        zlib.crc32(str(class_label).encode("utf-8")),
                        int(mode == "matched"),
                        31,
                    ]
                )
                selected = _select_case_buckets(
                    values,
                    cases,
                    max_per_bucket=max_per_bucket,
                    rng=rng,
                )
                buckets = {"best": [], "random": [], "worst": []}
                for bucket, case in selected:
                    buckets[bucket].append(
                        _case_payload(
                            values,
                            samples,
                            case,
                            group=class_label,
                            matched=mode == "matched",
                            theme=None,
                            task=task.name,
                            class_label=class_label,
                            bucket=bucket,
                        )
                    )
                class_result[mode] = buckets
            task_result[class_label] = class_result
        result[task.name] = task_result
    return result
