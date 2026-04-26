import argparse
import pickle
import re
import string
from pathlib import Path

import pandas as pd
from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

LABELS = ["B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-MISC", "I-MISC", "O"]
DEFAULT_DATA_DIR = Path(r"c:\Users\13599\Desktop\Project3\Starter Code\Starter Code\data")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
ORG_KEYWORDS = {
    "agency", "association", "bank", "bureau", "club", "co", "college", "committee", "commission",
    "company", "corp", "corporation", "council", "department", "federation", "fund", "government",
    "group", "inc", "institute", "international", "ltd", "ministry", "organization", "party",
    "school", "service", "society", "team", "union", "university",
}


def word_shape(token):
    chars = []
    for ch in token:
        if ch.isupper():
            chars.append("X")
        elif ch.islower():
            chars.append("x")
        elif ch.isdigit():
            chars.append("d")
        else:
            chars.append(ch)

    compact = []
    for ch in chars:
        if not compact or compact[-1] != ch:
            compact.append(ch)
    return "".join(compact)


def tokenize_sentence(text):
    return TOKEN_PATTERN.findall(text)


def df_to_sentences(df, has_label=True):
    sentences = []
    for _, group in df.groupby("sentence_id", sort=True):
        ordered = group.sort_values("token_idx")
        tokens = ordered["token"].fillna("").astype(str).tolist()
        if has_label:
            labels = ordered["label"].astype(str).tolist()
            sentences.append((tokens, labels))
        else:
            sentences.append(tokens)
    return sentences


def flatten_labels(sentences):
    return [label for _, labels in sentences for label in labels]


def flatten_predictions(predictions):
    return [label for sentence in predictions for label in sentence]


def extract_entities(tokens, labels):
    entities = []
    current_tokens = []
    current_type = None

    for token, label in zip(tokens, labels):
        if label == "O":
            if current_tokens:
                entities.append((current_type, " ".join(current_tokens)))
                current_tokens = []
                current_type = None
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            if current_tokens:
                entities.append((current_type, " ".join(current_tokens)))
            current_tokens = [token]
            current_type = entity_type
        else:
            current_tokens.append(token)

    if current_tokens:
        entities.append((current_type, " ".join(current_tokens)))

    return entities


class MEMM:
    def __init__(self, max_iter=20):
        self.max_iter = max_iter
        self.classifier = None

    def features(self, words, prev_label, pos):
        word = "" if pd.isna(words[pos]) else str(words[pos])
        word_lower = word.lower()
        prev_word = "" if pos == 0 else str(words[pos - 1])
        next_word = "" if pos == len(words) - 1 else str(words[pos + 1])
        prev_lower = prev_word.lower()
        next_lower = next_word.lower()
        word_is_title = word.istitle()
        prev_is_title = prev_word.istitle() if prev_word else False
        next_is_title = next_word.istitle() if next_word else False
        word_is_upper = word.isupper() if word else False
        prev_is_upper = prev_word.isupper() if prev_word else False
        next_is_upper = next_word.isupper() if next_word else False
        prev_is_org_keyword = prev_lower in ORG_KEYWORDS if prev_word else False
        next_is_org_keyword = next_lower in ORG_KEYWORDS if next_word else False

        feats = {
            "bias": 1,
            f"word={word}": 1,
            f"word.lower={word_lower}": 1,
            f"shape={word_shape(word)}": 1,
            f"prev_label={prev_label}": 1,
            f"prev_word.lower={prev_lower}": 1,
            f"next_word.lower={next_lower}": 1,
            f"prev_label+word.lower={prev_label}|{word_lower}": 1,
            f"prev_word+word={prev_lower}|{word_lower}": 1,
            f"word+next_word={word_lower}|{next_lower}": 1,
            f"len={min(len(word), 10)}": 1,
        }

        if pos == 0:
            feats["BOS"] = 1
        if pos == len(words) - 1:
            feats["EOS"] = 1

        if word:
            feats[f"prefix1={word_lower[:1]}"] = 1
            feats[f"prefix2={word_lower[:2]}"] = 1
            feats[f"prefix3={word_lower[:3]}"] = 1
            feats[f"prefix4={word_lower[:4]}"] = 1
            feats[f"suffix1={word_lower[-1:]}"] = 1
            feats[f"suffix2={word_lower[-2:]}"] = 1
            feats[f"suffix3={word_lower[-3:]}"] = 1
            feats[f"suffix4={word_lower[-4:]}"] = 1

            if word[0].isupper():
                feats["is_title_init"] = 1
            if word_is_title:
                feats["is_title"] = 1
            if word_is_upper:
                feats["is_upper"] = 1
            if word.islower():
                feats["is_lower"] = 1
            if word.isdigit():
                feats["is_digit"] = 1
            if any(ch.isdigit() for ch in word):
                feats["has_digit"] = 1
            if "-" in word:
                feats["has_hyphen"] = 1
            if "'" in word:
                feats["has_apostrophe"] = 1
            if "." in word:
                feats["has_period"] = 1
            if all(ch in string.punctuation for ch in word):
                feats["is_punct"] = 1

            if word_lower in ORG_KEYWORDS:
                feats["word_is_org_keyword"] = 1
            if word_lower.endswith(("inc", "corp", "ltd", "co")):
                feats["word_has_org_suffix"] = 1

        if prev_word:
            feats[f"prev_shape={word_shape(prev_word)}"] = 1
            if prev_is_title:
                feats["prev_is_title"] = 1
            if prev_is_org_keyword:
                feats["prev_is_org_keyword"] = 1
            if prev_is_upper:
                feats["prev_is_upper"] = 1

        if next_word:
            feats[f"next_shape={word_shape(next_word)}"] = 1
            if next_is_title:
                feats["next_is_title"] = 1
            if next_is_org_keyword:
                feats["next_is_org_keyword"] = 1
            if next_is_upper:
                feats["next_is_upper"] = 1
            if word_is_title and next_is_title:
                feats["titlecase_bigram"] = 1
            if word_is_upper and next_is_upper:
                feats["uppercase_bigram"] = 1

        if prev_word and word_is_title and prev_is_title:
            feats["prev_curr_titlecase"] = 1
        if next_word and word_is_title and next_is_title:
            feats["curr_next_titlecase"] = 1
        if prev_word and word_is_upper and prev_is_upper:
            feats["prev_curr_uppercase"] = 1
        if next_word and word_is_upper and next_is_upper:
            feats["curr_next_uppercase"] = 1
        if prev_word and prev_is_org_keyword:
            if word_is_title:
                feats["org_keyword_then_title"] = 1
            if word_is_upper:
                feats["org_keyword_then_upper"] = 1

        return feats

    def train(self, train_sents):
        samples = []
        for tokens, labels in train_sents:
            prev = "O"
            for i, label in enumerate(labels):
                samples.append((self.features(tokens, prev, i), label))
                prev = label

        self.classifier = MaxentClassifier.train(samples, max_iter=self.max_iter)

    def predict_sentence(self, tokens):
        if self.classifier is None:
            raise ValueError("Model has not been trained or loaded.")

        preds = []
        prev = "O"
        for i in range(len(tokens)):
            label = self.classifier.classify(self.features(tokens, prev, i))
            preds.append(label)
            prev = label
        return preds

    def predict_corpus(self, sentences):
        return [self.predict_sentence(tokens) for tokens in sentences]

    def save(self, model_path):
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with Path(model_path).open("rb") as f:
            return pickle.load(f)


def load_labeled_data(data_dir):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, dev_df, test_df


def evaluate(model, dev_sents):
    gold = flatten_labels(dev_sents)
    preds = flatten_predictions(model.predict_corpus([tokens for tokens, _ in dev_sents]))

    accuracy = accuracy_score(gold, preds)
    precision = precision_score(gold, preds, average="macro", labels=LABELS, zero_division=0)
    recall = recall_score(gold, preds, average="macro", labels=LABELS, zero_division=0)
    f1 = f1_score(gold, preds, average="macro", labels=LABELS, zero_division=0)

    print("\n" + "=" * 40)
    print("Dev set performance")
    print("=" * 40)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Macro F1  : {f1:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(gold, preds, labels=LABELS, zero_division=0))

    sample_tokens, sample_gold = dev_sents[0]
    sample_pred = model.predict_sentence(sample_tokens)
    print("\nSample prediction comparison:")
    print(f"{'Token':<20} | {'True':<10} | {'Pred':<10}")
    print("-" * 50)
    for token, true_label, pred_label in zip(sample_tokens, sample_gold, sample_pred):
        marker = "*" if true_label != pred_label else " "
        print(f"{token:<20} | {true_label:<10} | {pred_label:<10} {marker}")

    return f1


def write_submission(model, test_df, submission_path):
    submission_path = Path(submission_path)
    test_sents = df_to_sentences(test_df, has_label=False)
    flat_preds = flatten_predictions(model.predict_corpus(test_sents))
    if len(flat_preds) != len(test_df):
        raise ValueError("Prediction length mismatch with test.csv.")

    submission = pd.DataFrame({"id": test_df["id"], "label": flat_preds})
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False, encoding="utf-8")
    print(f"Saved submission to: {submission_path}")


def cmd_train_dev(args):
    train_df, dev_df, _ = load_labeled_data(args.data_dir)
    train_sents = df_to_sentences(train_df, has_label=True)
    dev_sents = df_to_sentences(dev_df, has_label=True)

    model = MEMM(max_iter=args.max_iter)
    model.train(train_sents)
    evaluate(model, dev_sents)
    model.save(args.model_path)
    print(f"Saved dev-trained model to: {args.model_path}")


def cmd_train_full(args):
    train_df, dev_df, test_df = load_labeled_data(args.data_dir)
    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    full_sents = df_to_sentences(full_df, has_label=True)

    model = MEMM(max_iter=args.max_iter)
    model.train(full_sents)
    model.save(args.model_path)
    print(f"Saved full model to: {args.model_path}")

    if args.submission_path:
        write_submission(model, test_df, args.submission_path)


def cmd_predict_test(args):
    _, _, test_df = load_labeled_data(args.data_dir)
    model = MEMM.load(args.model_path)
    write_submission(model, test_df, args.submission_path)


def cmd_demo(args):
    model = MEMM.load(args.model_path)
    tokens = tokenize_sentence(args.sentence)
    labels = model.predict_sentence(tokens)
    entities = extract_entities(tokens, labels)

    print("Tokens and labels:")
    for token, label in zip(tokens, labels):
        print(f"{token:<20} {label}")

    print("\nEntities:")
    if not entities:
        print("No named entities found.")
    else:
        for entity_type, entity_text in entities:
            print(f"{entity_type:<8} {entity_text}")


def build_parser():
    parser = argparse.ArgumentParser(description="Local MEMM NER runner for Project #3.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing train/dev/test CSV files.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_dev = subparsers.add_parser("train-dev", help="Train on train.csv and evaluate on dev.csv.")
    train_dev.add_argument("--max-iter", type=int, default=20)
    train_dev.add_argument("--model-path", default="outputs/model_dev.pkl")
    train_dev.set_defaults(func=cmd_train_dev)

    train_full = subparsers.add_parser("train-full", help="Train on train.csv + dev.csv and optionally export submission.")
    train_full.add_argument("--max-iter", type=int, default=20)
    train_full.add_argument("--model-path", default="outputs/model.pkl")
    train_full.add_argument("--submission-path", default="outputs/submission.csv")
    train_full.set_defaults(func=cmd_train_full)

    predict_test = subparsers.add_parser("predict-test", help="Load a saved model and export Kaggle submission.")
    predict_test.add_argument("--model-path", default="outputs/model.pkl")
    predict_test.add_argument("--submission-path", default="outputs/submission.csv")
    predict_test.set_defaults(func=cmd_predict_test)

    demo = subparsers.add_parser("demo", help="Run one sentence through a saved model.")
    demo.add_argument("--model-path", default="outputs/model.pkl")
    demo.add_argument("--sentence", required=True)
    demo.set_defaults(func=cmd_demo)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
