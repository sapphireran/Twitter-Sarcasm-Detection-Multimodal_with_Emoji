"""Evaluate predictions on cue-defined slices of a split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ccs2lab.cues import CueProfile, profile_text
from ccs2lab.metrics import BinaryScores, binary_scores


SlicePred = Callable[[CueProfile], bool]


def _explicit(profile: CueProfile) -> bool:
    return profile.has_explicit


def _no_explicit(profile: CueProfile) -> bool:
    return not profile.has_explicit


def _emoji(profile: CueProfile) -> bool:
    return profile.has_emoji


def _no_emoji(profile: CueProfile) -> bool:
    return not profile.has_emoji


def _not_tag(profile: CueProfile) -> bool:
    return profile.has_not_tag


SLICE_PREDICATES: dict[str, SlicePred] = {
    "all": lambda _profile: True,
    "explicit_cue": _explicit,
    "no_explicit_cue": _no_explicit,
    "has_emoji": _emoji,
    "no_emoji": _no_emoji,
    "not_tag": _not_tag,
}


@dataclass(frozen=True)
class SliceScores:
    name: str
    n: int
    scores: BinaryScores | None

    @property
    def empty(self) -> bool:
        return self.n == 0


def slice_scores(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    *,
    names: tuple[str, ...] | None = None,
) -> list[SliceScores]:
    if not (len(texts) == len(y_true) == len(y_pred)):
        raise ValueError("texts, y_true, and y_pred must align")
    profiles = [profile_text(text) for text in texts]
    wanted = names or tuple(SLICE_PREDICATES)
    out: list[SliceScores] = []
    for name in wanted:
        pred = SLICE_PREDICATES[name]
        idx = [i for i, profile in enumerate(profiles) if pred(profile)]
        if not idx:
            out.append(SliceScores(name=name, n=0, scores=None))
            continue
        scores = binary_scores(
            [y_true[i] for i in idx],
            [y_pred[i] for i in idx],
        )
        out.append(SliceScores(name=name, n=len(idx), scores=scores))
    return out
