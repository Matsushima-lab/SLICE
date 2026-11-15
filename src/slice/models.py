# src/slice/models.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cloud.enums import CausalGraph
from cloud.models import BaseCLOUD, encode_categories

from .partitions import (
    ExactPartitionSearch,
    ExactSearchResult,
    GreedyAgglomerativePartitionSearch,
    GreedySearchResult,
    SLICEPartition,
    cat_nml_codelength_from_counts,
)


@dataclass(frozen=True)
class DirectedFitDetails:
    base_cause_codelength: float
    partition: SLICEPartition
    search_method: str
    search: GreedySearchResult | ExactSearchResult


class DiscreteSLICE(BaseCLOUD):
    def __init__(
        self,
        X,
        Y,
        model_candidates: set[CausalGraph] | None = None,
        *,
        already_encoded: bool = False,
        domain_sizes: tuple[int, int] | None = None,
        min_hidden_cardinality: int = 2,
        exact_search_max_cause_cardinality: int | None = 8,
    ):
        super().__init__(X, Y, model_candidates)
        self.min_hidden_cardinality = int(min_hidden_cardinality)
        if self.min_hidden_cardinality < 1:
            raise ValueError("min_hidden_cardinality must be positive")
        self.exact_search_max_cause_cardinality = (
            None if exact_search_max_cause_cardinality is None else int(exact_search_max_cause_cardinality)
        )
        if (
            self.exact_search_max_cause_cardinality is not None
            and self.exact_search_max_cause_cardinality < self.min_hidden_cardinality
        ):
            raise ValueError(
                "exact_search_max_cause_cardinality must be >= min_hidden_cardinality, or None to disable exact search"
            )

        if already_encoded:
            self.Xd = np.asarray(X, dtype=np.int64)
            self.Yd = np.asarray(Y, dtype=np.int64)
            if domain_sizes is not None:
                self.mX, self.mY = map(int, domain_sizes)
            else:
                self.mX = int(self.Xd.max()) + 1 if self.Xd.size else 0
                self.mY = int(self.Yd.max()) + 1 if self.Yd.size else 0
        else:
            self.Xd, self.mX = encode_categories(self.X)
            self.Yd, self.mY = encode_categories(self.Y)

        self.x_to_y_details: DirectedFitDetails | None = None
        self.y_to_x_details: DirectedFitDetails | None = None

    @staticmethod
    def _joint_labels(x: np.ndarray, y: np.ndarray, m_y: int) -> np.ndarray:
        return np.asarray(x, dtype=np.int64) * int(m_y) + np.asarray(y, dtype=np.int64)

    def _run_partition_search(
        self,
        *,
        cause: np.ndarray,
        effect: np.ndarray,
        m_cause: int,
        m_effect: int,
        max_admissible_hidden: int,
    ) -> tuple[str, GreedySearchResult | ExactSearchResult]:
        use_exact = (
            self.exact_search_max_cause_cardinality is not None
            and m_cause <= self.exact_search_max_cause_cardinality
        )
        if use_exact:
            result = ExactPartitionSearch(
                min_blocks=self.min_hidden_cardinality,
                max_blocks=max_admissible_hidden,
            ).search(
                cause=cause,
                effect=effect,
                m_cause=m_cause,
                m_effect=m_effect,
            )
            return "exact", result

        result = GreedyAgglomerativePartitionSearch(
            min_blocks=self.min_hidden_cardinality
        ).search(
            cause=cause,
            effect=effect,
            m_cause=m_cause,
            m_effect=m_effect,
        )
        return "greedy", result

    def _directed_codelength(
        self,
        *,
        cause: np.ndarray,
        effect: np.ndarray,
        m_cause: int,
        m_effect: int,
    ) -> tuple[float, DirectedFitDetails | None]:
        cause_counts = np.bincount(cause, minlength=m_cause)
        base_cause = cat_nml_codelength_from_counts(cause_counts, m_cause)

        max_admissible_hidden = m_cause - 1
        if max_admissible_hidden < self.min_hidden_cardinality:
            return math.inf, None

        search_method, search = self._run_partition_search(
            cause=cause,
            effect=effect,
            m_cause=m_cause,
            m_effect=m_effect,
            max_admissible_hidden=max_admissible_hidden,
        )
        details = DirectedFitDetails(
            base_cause_codelength=base_cause,
            partition=search.best,
            search_method=search_method,
            search=search,
        )
        return base_cause + search.best.score_without_cause, details

    def _compute_codelengths(self) -> dict[CausalGraph, float]:
        Ls: dict[CausalGraph, float] = {}

        if CausalGraph.X_causes_Y in self.model_candidates:
            score, details = self._directed_codelength(
                cause=self.Xd,
                effect=self.Yd,
                m_cause=self.mX,
                m_effect=self.mY,
            )
            self.x_to_y_details = details
            Ls[CausalGraph.X_causes_Y] = float(score)

        if CausalGraph.Y_causes_X in self.model_candidates:
            score, details = self._directed_codelength(
                cause=self.Yd,
                effect=self.Xd,
                m_cause=self.mY,
                m_effect=self.mX,
            )
            self.y_to_x_details = details
            Ls[CausalGraph.Y_causes_X] = float(score)

        if CausalGraph.INDEPENDENT in self.model_candidates:
            x_counts = np.bincount(self.Xd, minlength=self.mX)
            y_counts = np.bincount(self.Yd, minlength=self.mY)
            Ls[CausalGraph.INDEPENDENT] = float(
                cat_nml_codelength_from_counts(x_counts, self.mX)
                + cat_nml_codelength_from_counts(y_counts, self.mY)
            )

        if CausalGraph.CONFOUNDER in self.model_candidates:
            joint = self._joint_labels(self.Xd, self.Yd, self.mY)
            joint_counts = np.bincount(joint, minlength=self.mX * self.mY)
            Ls[CausalGraph.CONFOUNDER] = float(
                cat_nml_codelength_from_counts(joint_counts, self.mX * self.mY)
            )

        return Ls

    def directed_partition(self, graph: CausalGraph) -> SLICEPartition:
        if graph is CausalGraph.X_causes_Y:
            if self.x_to_y_details is None:
                raise RuntimeError("Call fit() first.")
            return self.x_to_y_details.partition
        if graph is CausalGraph.Y_causes_X:
            if self.y_to_x_details is None:
                raise RuntimeError("Call fit() first.")
            return self.y_to_x_details.partition
        raise ValueError("graph must be X_causes_Y or Y_causes_X")

    def directed_structure(self, graph: CausalGraph) -> SLICEPartition:
        return self.directed_partition(graph)

    def directed_details(self, graph: CausalGraph) -> DirectedFitDetails:
        if graph is CausalGraph.X_causes_Y:
            if self.x_to_y_details is None:
                raise RuntimeError("Call fit() first.")
            return self.x_to_y_details
        if graph is CausalGraph.Y_causes_X:
            if self.y_to_x_details is None:
                raise RuntimeError("Call fit() first.")
            return self.y_to_x_details
        raise ValueError("graph must be X_causes_Y or Y_causes_X")

    def summary(self) -> str:
        base = super().summary()
        extras: list[str] = []
        if self._fit is not None:
            if self.x_to_y_details is not None:
                groups = self.x_to_y_details.partition.groups
                extras.append(
                    f"Best X->Y SLICE partition ({self.x_to_y_details.search_method}): "
                    f"m_Z={len(groups)}, groups={groups}"
                )
            if self.y_to_x_details is not None:
                groups = self.y_to_x_details.partition.groups
                extras.append(
                    f"Best Y->X SLICE partition ({self.y_to_x_details.search_method}): "
                    f"m_Z={len(groups)}, groups={groups}"
                )
        return base if not extras else base + "\n" + "\n".join(extras)