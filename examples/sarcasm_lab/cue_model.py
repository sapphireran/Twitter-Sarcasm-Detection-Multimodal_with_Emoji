"""Tiny classifiers that only see the cue feature vector."""

from __future__ import annotations

from dataclasses import dataclass

from .cues import CueFeatures


@dataclass
class RuleCueClassifier:
    """Hand-written rule: sarcasm hashtag, or positive stem + negative emoji.

    This is a teaching baseline, not a tuned model. It shows how much of the
    subtest is explained by the distant-supervision tags themselves.
    """

    require_hashtag_or_combo: bool = True

    def predict_one(self, cues: CueFeatures) -> int:
        if cues.has_sarcasm_hashtag:
            return 1
        if cues.has_positive_stem and (cues.has_negative_emoji or cues.n_ellipsis > 0):
            return 1
        return 0

    def predict(self, cue_list: list[CueFeatures]) -> list[int]:
        return [self.predict_one(cues) for cues in cue_list]


@dataclass
class CountCueClassifier:
    """Score = weighted sum of cue flags; predict 1 if score >= threshold."""

    hashtag_weight: float = 2.5
    combo_weight: float = 1.5
    emoji_weight: float = 0.35
    hashtag_count_weight: float = 0.15
    threshold: float = 1.0

    def score(self, cues: CueFeatures) -> float:
        score = 0.0
        if cues.has_sarcasm_hashtag:
            score += self.hashtag_weight
        if cues.has_positive_stem and cues.has_negative_emoji:
            score += self.combo_weight
        score += self.emoji_weight * min(cues.n_emoji, 4)
        score += self.hashtag_count_weight * min(cues.n_hashtags, 4)
        return score

    def predict_one(self, cues: CueFeatures) -> int:
        return 1 if self.score(cues) >= self.threshold else 0

    def predict(self, cue_list: list[CueFeatures]) -> list[int]:
        return [self.predict_one(cues) for cues in cue_list]
