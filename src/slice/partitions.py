# src/slice/partitions.py
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from cloud.codelengths import log_C_CAT



@lru_cache(maxsize=None)
def stirling2(n: int, k: int) -> int:
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    if k == 1 or k == n:
        return 1
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)


@lru_cache(maxsize=None)
def log_stirling2(n: int, k: int) -> float:
    value = stirling2(n, k)
    if value <= 0:
        raise ValueError(f"Stirling number S({n}, {k}) is zero.")
    return math.log2(value)


@lru_cache(maxsize=None)
def bell_number(n: int) -> int:
    return sum(stirling2(n, k) for k in range(n + 1))


def cat_nml_codelength_from_counts(counts: np.ndarray, domain_size: int | None = None) -> float:
    arr = np.asarray(counts, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    if np.any(arr < 0):
        raise ValueError("counts must be non-negative")
    n = int(arr.sum())
    K = int(domain_size if domain_size is not None else arr.size)
    if n == 0:
        return 0.0
    nz = arr[arr > 0]
    nll = float(np.sum(nz * (math.log2(n) - np.log2(nz))))
    return nll + log_C_CAT(n, K)


@dataclass(frozen=True)
class SLICEPartition:
    num_blocks: int
    groups: tuple[tuple[int, ...], ...]
    score_without_cause: float

    def mapping(self, domain_size: int) -> np.ndarray:
        mapping = np.empty(domain_size, dtype=np.int64)
        for label, group in enumerate(self.groups):
            for item in group:
                mapping[item] = label
        return mapping


@dataclass(frozen=True)
class MergeStep:
    num_blocks: int
    merged_pair: tuple[int, int]
    groups: tuple[tuple[int, ...], ...]
    score_without_cause: float


@dataclass(frozen=True)
class GreedySearchResult:
    best: SLICEPartition
    history: tuple[MergeStep, ...]


@dataclass(frozen=True)
class ExactSearchResult:
    best: SLICEPartition
    num_partitions_evaluated: int
    bell_number: int


@dataclass(frozen=True)
class PartitionCandidate:
    groups: tuple[tuple[int, ...], ...]
    score_without_cause: float


def contingency_matrix(cause: np.ndarray, effect: np.ndarray, m_cause: int, m_effect: int) -> np.ndarray:
    cause = np.asarray(cause, dtype=np.int64)
    effect = np.asarray(effect, dtype=np.int64)
    if cause.shape != effect.shape:
        raise ValueError("cause and effect must have the same shape")
    mat = np.zeros((m_cause, m_effect), dtype=np.int64)
    np.add.at(mat, (cause, effect), 1)
    return mat


class GreedyAgglomerativePartitionSearch:
    def __init__(self, *, min_blocks: int = 2):
        if min_blocks < 1:
            raise ValueError("min_blocks must be positive")
        self.min_blocks = int(min_blocks)

    def search(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        m_cause: int,
        m_effect: int,
    ) -> GreedySearchResult:
        if m_cause <= self.min_blocks:
            raise ValueError("m_cause must be strictly larger than min_blocks for greedy merging")

        counts = contingency_matrix(cause, effect, m_cause, m_effect)
        groups: list[tuple[int, ...]] = [tuple([i]) for i in range(m_cause)]
        group_counts: list[np.ndarray] = [counts[i].copy() for i in range(m_cause)]
        group_scores: list[float] = [cat_nml_codelength_from_counts(c, m_effect) for c in group_counts]

        best_partition: SLICEPartition | None = None
        history: list[MergeStep] = []
        current_blocks = m_cause

        while current_blocks > self.min_blocks:
            best_delta: float | None = None
            best_pair: tuple[int, int] | None = None
            best_merged_counts: np.ndarray | None = None
            best_merged_score: float | None = None

            for a in range(current_blocks):
                for b in range(a + 1, current_blocks):
                    merged_counts = group_counts[a] + group_counts[b]
                    merged_score = cat_nml_codelength_from_counts(merged_counts, m_effect)
                    delta = merged_score - group_scores[a] - group_scores[b]
                    if (
                        best_delta is None
                        or delta < best_delta
                        or (math.isclose(delta, best_delta) and (a, b) < best_pair)
                    ):
                        best_delta = delta
                        best_pair = (a, b)
                        best_merged_counts = merged_counts
                        best_merged_score = merged_score

            assert best_pair is not None
            assert best_merged_counts is not None
            assert best_merged_score is not None

            a, b = best_pair
            merged_group = tuple(sorted(groups[a] + groups[b]))

            new_groups = list(groups)
            new_group_counts = list(group_counts)
            new_group_scores = list(group_scores)

            new_groups[a] = merged_group
            new_group_counts[a] = best_merged_counts
            new_group_scores[a] = best_merged_score
            del new_groups[b]
            del new_group_counts[b]
            del new_group_scores[b]

            groups = new_groups
            group_counts = new_group_counts
            group_scores = new_group_scores
            current_blocks -= 1

            stage_score = (
                np.log2(m_cause - 2)
                + log_stirling2(m_cause, current_blocks)
                + float(sum(group_scores))
            )

            snapshot = tuple(tuple(g) for g in groups)
            history.append(
                MergeStep(
                    num_blocks=current_blocks,
                    merged_pair=best_pair,
                    groups=snapshot,
                    score_without_cause=stage_score,
                )
            )

            partition = SLICEPartition(
                num_blocks=current_blocks,
                groups=snapshot,
                score_without_cause=stage_score,
            )
            if best_partition is None or partition.score_without_cause < best_partition.score_without_cause:
                best_partition = partition

        assert best_partition is not None
        return GreedySearchResult(best=best_partition, history=tuple(history))


class ExactPartitionSearch:
    def __init__(self, *, min_blocks: int = 2, max_blocks: int | None = None):
        if min_blocks < 1:
            raise ValueError("min_blocks must be positive")
        self.min_blocks = int(min_blocks)
        self.max_blocks = None if max_blocks is None else int(max_blocks)
        if self.max_blocks is not None and self.max_blocks < self.min_blocks:
            raise ValueError("max_blocks must be greater than or equal to min_blocks")

    def search(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        m_cause: int,
        m_effect: int,
    ) -> ExactSearchResult:
        max_blocks = m_cause if self.max_blocks is None else min(self.max_blocks, m_cause)
        if self.min_blocks > max_blocks:
            raise ValueError("No admissible number of blocks for exact search")

        counts = contingency_matrix(cause, effect, m_cause, m_effect)
        subset_scores, subset_groups = self._precompute_subset_terms(counts, m_cause, m_effect)

        best: PartitionCandidate | None = None
        num_evaluated = 0
        initial_mask = 1
        blocks: list[int] = [initial_mask]

        def block_score(mask: int) -> float:
            return subset_scores[mask]

        def recurse(next_item: int, current_block_score_sum: float) -> None:
            nonlocal best, num_evaluated

            current_k = len(blocks)
            remaining_items = m_cause - next_item
            if current_k > max_blocks:
                return
            if current_k + remaining_items < self.min_blocks:
                return

            if next_item == m_cause:
                if self.min_blocks <= current_k <= max_blocks:
                    total_score = (
                        np.log2(m_cause - 2)
                        + log_stirling2(m_cause, current_k)
                        + current_block_score_sum
                    )
                    groups = tuple(subset_groups[mask] for mask in blocks)
                    num_evaluated += 1
                    candidate = PartitionCandidate(groups=groups, score_without_cause=total_score)
                    if (
                        best is None
                        or candidate.score_without_cause < best.score_without_cause
                        or (
                            math.isclose(candidate.score_without_cause, best.score_without_cause)
                            and candidate.groups < best.groups
                        )
                    ):
                        best = candidate
                return

            item_mask = 1 << next_item

            # Option 1: add the item to an existing block.
            for idx in range(current_k):
                old_mask = blocks[idx]
                new_mask = old_mask | item_mask
                if new_mask == old_mask:
                    continue
                old_score = block_score(old_mask)
                new_score = block_score(new_mask)
                blocks[idx] = new_mask
                recurse(next_item + 1, current_block_score_sum - old_score + new_score)
                blocks[idx] = old_mask

            # Option 2: start a new block at the end.
            if current_k < max_blocks:
                blocks.append(item_mask)
                recurse(next_item + 1, current_block_score_sum + block_score(item_mask))
                blocks.pop()

        recurse(next_item=1, current_block_score_sum=block_score(initial_mask))

        if best is None:
            raise RuntimeError("Exact search did not find any admissible partition")

        partition = SLICEPartition(
            num_blocks=len(best.groups),
            groups=best.groups,
            score_without_cause=best.score_without_cause,
        )
        return ExactSearchResult(
            best=partition,
            num_partitions_evaluated=num_evaluated,
            bell_number=bell_number(m_cause),
        )

    @staticmethod
    def _precompute_subset_terms(
        counts: np.ndarray,
        m_cause: int,
        m_effect: int,
    ) -> tuple[dict[int, float], dict[int, tuple[int, ...]]]:
        subset_counts: dict[int, np.ndarray] = {}
        subset_scores: dict[int, float] = {}
        subset_groups: dict[int, tuple[int, ...]] = {}
        for mask in range(1, 1 << m_cause):
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            if prev == 0:
                agg = counts[idx].copy()
            else:
                agg = subset_counts[prev] + counts[idx]
            subset_counts[mask] = agg
            subset_scores[mask] = cat_nml_codelength_from_counts(agg, m_effect)
            subset_groups[mask] = tuple(i for i in range(m_cause) if (mask >> i) & 1)
        return subset_scores, subset_groups