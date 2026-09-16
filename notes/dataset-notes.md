# Dataset notes

Personal notes on the three CSV pairs in `dataset/`. This is what the files contain, not a datasheet for a public release.

## Files

Each split is two parallel files with no header:

- `*_sentence.csv` — one tweet per line. Some lines are quoted because the tweet itself has commas.
- `*_label.csv` — `0` or `1` per line, same order.

`ReadOpen` does **not** use a CSV parser for sentences. It reads raw lines, strips, and turns commas into spaces (`' '.join(line.strip().split(','))`) before `TweetTokenizer`. That is clumsy but matches how I trained. If I ever rewrite the reader with `pandas.read_csv`, quote-wrapped tweets will tokenize differently and the old numbers will not come back.

Labels *are* read with pandas (`header=None`). I have not found a length mismatch in this checkout.

## Counts I trust

| Split | lines | sarcastic (`1`) | non-sarcastic (`0`) |
| --- | ---: | ---: | ---: |
| train | 39,780 | 18,488 | 21,292 |
| test | 2,000 | 1,000 | 1,000 |
| subtest | 278 | 172 | 106 |

Train is close to balanced, slightly more negative. Test is exactly balanced. Subtest is not: 61.9% sarcastic. I should have printed class priors next to every subtest score. I did not.

## How the tweets look

Typical social-media leftovers:

- `<user>` instead of @handles on train (9,433 rows). Test and subtest do not use that token at all. That is a silent domain shift I never modeled.
- Hashtags left in (`#Not`, `#MorningTweeps`, `#whitepeopleproblems`). 8,486 train rows have at least one `#word`.
- Emoji as separate characters, sometimes repeated (`😂 😂 😂`).
- Quoted lines when the original tweet had commas or internal quotes. The comma-to-space trick above smashes those.

Length (whitespace-split, before tweet-tokenizer quirks):

| Split | min | max | mean |
| --- | ---: | ---: | ---: |
| train | 1 | 51 | 16.46 |
| test | 1 | 36 | 16.51 |
| subtest | 5 | 36 | 17.70 |

The deep model padded to 78 after `TweetTokenizer` + Keras `Tokenizer`. Plenty of headroom; almost nothing is being truncated.

## Emoji is not evenly spread

I counted Unicode emoji ranges plus a few dingbat blocks. Not a perfect emoji library, close enough to see the shape.

| Split | tweets with ≥1 emoji | rate | sarcastic ∩ emoji | non-sarcastic ∩ emoji |
| --- | ---: | ---: | ---: | ---: |
| train | 5,470 | 13.8% | 2,277 / 18,488 | 3,193 / 21,292 |
| test | 277 | 13.9% | 171 / 1,000 | 106 / 1,000 |
| subtest | 277 | 99.6% | 171 / 172 | 106 / 106 |

Two facts I keep repeating to myself:

1. **Subtest is the emoji slice**, not a second random test. 277/278 rows have emoji. The 171 / 106 sarcastic–emoji vs non-sarcastic–emoji counts on subtest match the test-set emoji subset exactly (171 and 106). I now think subtest *is* "the test rows that contain emoji," plus one leftover sarcastic row without a match in my regex. That would also explain why test and subtest share the same 171/106 emoji-class split.
2. **On train, emoji is a bit more common in the negative class.** A model that only sees "this tweet has a face" should not drift toward sarcasm. If WE helps, it should be because the *identity* of the face interacts with the words.

If subtest really is "test ∩ emoji," then the fair paired comparison I wanted is already sitting there. I still wish I had written that sentence in 2023.

## What sarcasm looks like in this dump

Positive-class examples from train, kept short:

- `Being sore is the best and the worst feeling in the world`
- `Expecting is my favorite crime and disappointment is always my punishment.. 😎😎 😥 😊`
- `Happy birthday to me. Yay.`
- `Late nights early mornings hmmm & i say THANK YOOOOOOO 😚 😍`

The late-night / thank-you cluster repeats with different faces. Some of those labels I would not bet my own money on. Distant supervision from hashtags (`#Not`, `#sarcasm`) is the usual way these corpora get built; I did not keep the original annotation protocol in this repo.

Emoji-bearing negatives are often sincere intensity, not inversion:

- `<user> i hope youre lurking rn. … pretty please?! 😭 😭 😭`
- `100 days until Christmas! 🌲 #too soon #not ready yet`
- `<user> thirteen ghosts is the fucking shit 😍 😍 😍`

That is exactly the confusion I hoped emoji2vec would help with: repeated hearts after a compliment vs hearts after a complaint.

## Tokenization choices that leak into every number

`ReadOpen`:

1. Replace commas with spaces (see above).
2. `TweetTokenizer()` from NLTK — keeps hashtags and emoji better than `str.split`.
3. Lowercase every token.

Then two different futures:

- **Baselines:** stay as a list of tokens. GloVe / emoji2vec lookup is exact string match on those lowercased tokens.
- **Deep model:** Keras `Tokenizer` (default: split on whitespace / punctuation, lower already done). Sequences are integer ids. Pad with `padding='post'`. The embedding matrix is `zeros((count, 200))` where `count` is the number of *training lines*, not `len(word_index)+1`. That is an odd allocation. It worked because the Keras tokenizer's index is 1-based and the number of unique tokens was below 39,780. I would still size the table as `len(tokenizer.word_index) + 1` if I rewrote it.

`preprocess_test` reuses the train tokenizer and the train pad length. Test tokens that never appeared in train become zeros in the sequence (Keras default). Their emoji never gets a chance to land in the table unless that exact token was also in train.

## Two emoji2vec binaries

| File | Size | Used in notebooks? |
| --- | ---: | --- |
| `emoji2vec_twitter.bin` | ~1.3 MB | yes |
| `emoji2vec.bin` | ~2.0 MB | not in the 2023 notebooks I still have |

I do not remember why both are here. All recorded WE numbers used the Twitter one. Do not mix them in a rerun and compare to the table.

GloVe is not in git. Notebooks disagree on the filename:

- `baseline_models.ipynb` / `evaluate_loaded_dl_models.ipynb`: `glove.twitter.27B.200d.bin`
- `get_metrics_of_models.ipynb`: `glove_tt.txt` (word2vec text)

Same 200-d Twitter GloVe, different serialization. The metrics notebook printed `Loaded 1193515 word vectors` during preprocess — that is the 27B / 1.2M vocab.

## Things I would clean if this were more than a course dump

- Parse sentences as CSV, not comma-to-space.
- Document whether subtest is exactly `test` filtered to emoji.
- Keep the original hashtag / distant-supervision rule.
- Put class priors in every results table.
- Drop or explain `<user>` only appearing in train.
- Stop carrying two emoji2vec files with no README line.

Until then, the honest description is: ~40k weakly labeled tweets, a balanced 2k test, and a 278-row emoji-heavy slice that is probably the emoji subset of that test.
