"""Shared input aliases and fixed output schemas for public annotations."""

from os import PathLike
from typing import TypeAlias, TypedDict

from ete4 import PhyloTree

PathInput: TypeAlias = str | PathLike[str]
TreeSource: TypeAlias = PathInput | PhyloTree


class NotungRootStats(TypedDict, total=False):
    ntg_num_opt_root: int
    ntg_best_root_score: float
    ntg_worst_root_score: float


class RootStats(TypedDict, total=False):
    num_rho_peak: int
    rooting_method: str
