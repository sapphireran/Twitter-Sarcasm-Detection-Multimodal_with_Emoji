"""Lightweight tweet tokenization and lexical sarcasm cues.

The original course notebooks depend on NLTK's ``TweetTokenizer`` plus GloVe
and emoji2vec. These helpers keep the same *ideas* — keep hashtags, mentions,
and emoji as tokens, then look at contrast cues — but they run with only the
standard library so the examples stay reproducible.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

HASHTAG_RE = re.compile(r"(?i)#\w+")
MENTION_RE = re.compile(r"(?i)@\w+|<user>")
URL_RE = re.compile(r"(?i)https?://\S+|www\.\S+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
PLAYFUL_LOVE_RE = re.compile(r"^l+o+v+e+$")
TOKEN_RE = re.compile(
    r"(?i)"
    r"https?://\S+|www\.\S+"
    r"|#\w+"
    r"|@\w+|<user>"
    r"|[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]"
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"
    r"|\d+"
    r"|[\.\!\?]+"
    r"|[^\s]"
)
EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U0001F1E6-\U0001F1FF]"
)

SARCASM_HASHTAGS = frozenset(
    {
        "#not",
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#irony",
        "#ironic",
    }
)
POSITIVE_WORDS = frozenset(
    {
        "love",
        "loved",
        "loves",
        "great",
        "awesome",
        "amazing",
        "perfect",
        "wonderful",
        "best",
        "happy",
        "glad",
        "yay",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "hate",
        "hated",
        "awful",
        "terrible",
        "worst",
        "stupid",
        "dumb",
        "pissed",
        "angry",
        "sad",
        "pain",
        "ugly",
    }
)
# Crying faces are left out on purpose: they are common in sincere tweets.
NEGATIVE_EMOJI = frozenset("😒😑😩🔫👎💔😅🙄")
POSITIVE_EMOJI = frozenset("😂😊😍❤😘😄😃😁✨")


def normalize_commas(text: str) -> str:
    """Mirror ``data_utils.ReadOpen``: commas are treated as spaces."""
    return " ".join(text.strip().split(","))


def is_emoji(char: str) -> bool:
    if not char:
        return False
    if EMOJI_RE.fullmatch(char):
        return True
    return unicodedata.category(char) in {"So", "Sk"} and ord(char) > 127


def tokenize_tweet(text: str, *, lowercase: bool = True) -> list[str]:
    """Approximate NLTK TweetTokenizer enough for examples and tests."""
    cleaned = normalize_commas(text)
    tokens = TOKEN_RE.findall(cleaned)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def extract_emojis(text: str) -> list[str]:
    return EMOJI_RE.findall(text)


@dataclass(frozen=True)
class TweetFeatures:
    """Sparse lexical description of one tweet."""

    token_count: int
    char_count: int
    hashtag_count: int
    mention_count: int
    url_count: int
    emoji_count: int
    exclamation_count: int
    question_count: int
    ellipsis: int
    elongated_count: int
    allcaps_count: int
    sarcasm_hashtag: int
    not_hashtag: int
    positive_word_count: int
    negative_word_count: int
    positive_emoji_count: int
    negative_emoji_count: int
    contrast_cue: int
    love_word: int

    def as_vector(self) -> list[float]:
        """Dense vector used by the tiny logistic baseline."""
        scale = max(self.token_count, 1)
        return [
            self.token_count / 40.0,
            self.char_count / 200.0,
            self.hashtag_count / 5.0,
            self.mention_count / 3.0,
            self.url_count,
            self.emoji_count / 5.0,
            self.exclamation_count / 4.0,
            self.question_count / 3.0,
            float(self.ellipsis),
            self.elongated_count / 3.0,
            self.allcaps_count / 4.0,
            float(self.sarcasm_hashtag),
            float(self.not_hashtag),
            self.positive_word_count / scale,
            self.negative_word_count / scale,
            self.positive_emoji_count / 3.0,
            self.negative_emoji_count / 3.0,
            float(self.contrast_cue),
            float(self.love_word),
        ]


FEATURE_NAMES = [
    "token_count",
    "char_count",
    "hashtag_count",
    "mention_count",
    "url_count",
    "emoji_count",
    "exclamation_count",
    "question_count",
    "ellipsis",
    "elongated_count",
    "allcaps_count",
    "sarcasm_hashtag",
    "not_hashtag",
    "positive_word_rate",
    "negative_word_rate",
    "positive_emoji",
    "negative_emoji",
    "contrast_cue",
    "love_word",
]


def extract_features(text: str) -> TweetFeatures:
    """Turn a tweet into counts that often mark sarcastic contrast."""
    tokens = tokenize_tweet(text, lowercase=False)
    lower_tokens = [token.lower() for token in tokens]
    hashtags = {match.group(0).lower() for match in HASHTAG_RE.finditer(text)}
    emojis = extract_emojis(text)

    sarcasm_hashtag = int(bool(hashtags & SARCASM_HASHTAGS))
    not_hashtag = int("#not" in hashtags)
    positive_words = sum(token in POSITIVE_WORDS for token in lower_tokens)
    negative_words = sum(token in NEGATIVE_WORDS for token in lower_tokens)
    positive_emoji = sum(char in POSITIVE_EMOJI for char in emojis)
    negative_emoji = sum(char in NEGATIVE_EMOJI for char in emojis)
    contrast = int(
        (positive_words > 0 and (negative_emoji > 0 or not_hashtag or sarcasm_hashtag))
        or (positive_emoji > 0 and (negative_words > 0 or not_hashtag))
    )

    return TweetFeatures(
        token_count=len(tokens),
        char_count=len(text),
        hashtag_count=len(HASHTAG_RE.findall(text)),
        mention_count=len(MENTION_RE.findall(text)),
        url_count=len(URL_RE.findall(text)),
        emoji_count=len(emojis),
        exclamation_count=text.count("!"),
        question_count=text.count("?"),
        ellipsis=int("..." in text or "…" in text),
        elongated_count=sum(
            1
            for token in lower_tokens
            if token.isalpha() and ELONGATION_RE.search(token)
        ),
        allcaps_count=sum(
            1
            for token in tokens
            if token.isalpha() and token.isupper() and len(token) > 1
        ),
        sarcasm_hashtag=sarcasm_hashtag,
        not_hashtag=not_hashtag,
        positive_word_count=positive_words,
        negative_word_count=negative_words,
        positive_emoji_count=positive_emoji,
        negative_emoji_count=negative_emoji,
        contrast_cue=contrast,
        love_word=int(
            any(
                token in {"love", "loved", "loves"} or PLAYFUL_LOVE_RE.fullmatch(token)
                for token in lower_tokens
            )
        ),
    )


def features_to_dict(text: str) -> dict[str, int]:
    return asdict(extract_features(text))
