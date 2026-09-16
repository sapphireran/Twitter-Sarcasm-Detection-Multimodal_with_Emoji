# Sample predictions (lexical floor)

These rows are the featured tweets from `tokenize_tweets.py`. The
**hashtag rule** column is `#not` / `#sarcasm*` / `#yeahright` → 1,
else 0. It is the dumb baseline `lexical_sarcasm_baseline.py`
compares against. Gold labels are from `dataset/test_sentence.csv`
except the comma-quote row, which is from train.

| Gold | Hashtag rule | Tweet | Why the rule is right or wrong |
| ---: | ---: | --- | --- |
| 1 | 1 | I loovee when people text back ... 😒 #sarcastictweet | Explicit tag. Easy for everyone, including SVM. |
| 1 | 1 | Oh how I love getting home from work at 3am and my house being dirty #not | Dominant test pattern. `#not` is the label. |
| 1 | 1 | I love walking to school 😄 #SarcasticTweet | Tag wins; the smile emoji is a distractor for emoji2vec. |
| 0 | 0 | Want to have someone to speak to I'm so bored 😭 | Emoji + complaint, no sarcasm tag. Rule correctly stays 0. |
| 0 | 0 | i just imagined you dancing like this | No cues at all. |
| 1 | 0 | Good thing I spent four years getting MLA formatting shoved down my throat . #itsAPA 😑 👎 | No `#not`. Needs polarity flip / LSTM. |
| 0 | 0 | "I feel so honored 2 organizations , one from NCCU & one from UNCG , want me to come speak at their school 😌" | Positive + emoji, sincere. |

Run `python3 examples/tokenize_tweets.py` to see the token lists for
the first five rows, and `python3 examples/lexical_sarcasm_baseline.py`
for accuracy of a fitted logreg on the full splits.
