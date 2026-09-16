from examples.lib.tokenize import (
    SARCASM_TAGS,
    find_emoji,
    find_hashtags,
    has_sarcasm_tag,
    tokenize,
)


def test_keeps_hashtags_mentions_and_user_placeholder() -> None:
    tokens = tokenize("Hey <user> I love this #Not #SarcasticTweet @friend")
    assert "<user>" in tokens
    assert "#not" in tokens
    assert "#sarcastictweet" in tokens
    assert "@friend" in tokens
    assert "hey" in tokens


def test_keeps_emoji_as_their_own_tokens() -> None:
    tokens = tokenize("great job 😒 😂")
    assert "😒" in tokens
    assert "😂" in tokens
    assert find_emoji("great job 😒 😂") == ["😒", "😂"]


def test_hashtag_finder_is_casefolding() -> None:
    assert find_hashtags("Wow #NOT #YeahRight") == ["#not", "#yeahright"]
    assert has_sarcasm_tag("this is fine #Sarcasm")
    assert not has_sarcasm_tag("this is fine #monday")
    assert set(SARCASM_TAGS) >= {"#not", "#sarcasm"}
