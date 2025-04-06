import pandas as pd
import csv
import nltk
from collections import Counter, defaultdict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('punkt')


class Model:

    def __init__(self, vocab_size, UNK_token='<UNK>'):
        self.n = 3
        self.vocab_size = vocab_size
        self.UNK = UNK_token
        self.vocab = set()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.candidates = defaultdict(set)

    def train(self, corpus: list) -> None:
        freq = Counter(corpus)
        self.vocab = {word for word, _ in freq.most_common(self.vocab_size - 1)}
        self.vocab.add(self.UNK)

        tokens = [w if w in self.vocab else self.UNK for w in corpus]
        padding = ["<s>"] * (self.n // 2) + tokens + ["</s>"] * (self.n // 2)

        for gram in ngrams(padding, self.n):
            center = gram[self.n // 2]
            context = gram[:self.n // 2] + gram[self.n // 2 + 1:]
            self.ngram_counts[gram] += 1
            self.context_counts[context] += 1
            self.candidates[context].add(center)

    def predict(self, context_pair: list, strategy: str = "topk") -> str:
        left, right = context_pair
        left = left[-(self.n // 2):]
        right = right[:(self.n // 2)]

        left = [w if w in self.vocab else self.UNK for w in left]
        right = [w if w in self.vocab else self.UNK for w in right]

        context = tuple(left + right)
        candidates = self.candidates.get(context, [])
        predictions = []

        for word in candidates:
            ngram = tuple(left + [word] + right)
            prob = self.ngram_counts.get(ngram, 0) / self.context_counts.get(context, 1)
            predictions.append((word, prob))

        predictions.sort(key=lambda x: x[1], reverse=True)
        topk = predictions[:20]

        total = sum(p for _, p in topk)
        output = []
        used = 0

        for word, prob in topk:
            norm_prob = round(prob / total, 2) if total > 0 else 0
            used += norm_prob
            output.append(f"{word}:{norm_prob}")

        remainder = round(1.0 - used, 2)
        output.append(f":{remainder if remainder > 0 else 0.01}")
        return " ".join(output)


def tokenize(text: str) -> list:
    return word_tokenize(str(text).lower())


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', header=None,
                       names=['FileId', 'Year', 'LeftContext', 'RightContext'],
                       quoting=csv.QUOTE_NONE).replace(r'\\r|\\n|\n|\\t', ' ', regex=True)


def load_expected(path: str) -> list:
    return pd.read_csv(path, sep='\t', header=None, names=["Word"],
                       quoting=csv.QUOTE_NONE)["Word"].astype(str).tolist()


def build_training_tokens(data_path: str, expected_path: str) -> list:
    df = load_data(data_path)
    expected_words = load_expected(expected_path)

    tokens = []
    for row, label in zip(df.itertuples(index=False), expected_words):
        tokens.extend(tokenize(row.LeftContext))
        tokens.append(label)
        tokens.extend(tokenize(row.RightContext))
    return tokens


def predict_on_dataset(model: Model, input_path: str, output_path: str) -> None:
    df = load_data(input_path)
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Predicting {input_path}"):
        left = tokenize(row.LeftContext)
        right = tokenize(row.RightContext)
        predictions.append(model.predict([left, right]))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))


if __name__ == "__main__":
    train_tokens = build_training_tokens("train/in.tsv", "train/expected.tsv")
    model = Model(vocab_size=30000)
    model.train(train_tokens)

    predict_on_dataset(model, "dev-0/in.tsv", "dev-0/out.tsv")
    predict_on_dataset(model, "test-A/in.tsv", "test-A/out.tsv")
