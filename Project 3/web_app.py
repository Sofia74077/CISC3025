from pathlib import Path
import math
import pickle
import re
import sys
import types
from collections import Counter

from flask import Flask, render_template, request
import nltk

# The trained model inference does not require sklearn metrics.
# This shim avoids import failure when sklearn is unavailable locally.
if "sklearn.metrics" not in sys.modules:
    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")

    def _unused_metric(*args, **kwargs):
        raise RuntimeError("sklearn is required for training/evaluation, not web inference.")

    metrics_module.accuracy_score = _unused_metric
    metrics_module.classification_report = _unused_metric
    metrics_module.f1_score = _unused_metric
    metrics_module.precision_score = _unused_metric
    metrics_module.recall_score = _unused_metric
    sklearn_module.metrics = metrics_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module

from local_memm_ner import MEMM, extract_entities, tokenize_sentence


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "model.pkl"
NB_PROBABILITY_PATH = BASE_DIR / "Project 2" / "word_probability.txt"
NB_CATEGORIES = ["crude", "grain", "money-fx", "acq", "earn"]
NB_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "MEMM":
                return MEMM
            return super().find_class(module, name)

    with MODEL_PATH.open("rb") as model_file:
        return _CompatUnpickler(model_file).load()


class NaiveBayesTextClassifier:
    def __init__(self, probability_path):
        self.probability_path = Path(probability_path)
        self.stemmer = nltk.PorterStemmer()
        self.priors = None
        self.word_probs = {}
        self._load()

    def _load(self):
        if not self.probability_path.exists():
            return

        with self.probability_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return
            self.priors = [float(x) for x in first_line.split()]
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                self.word_probs[parts[0]] = [float(x) for x in parts[1:]]

    @property
    def available(self):
        return bool(self.priors) and bool(self.word_probs)

    def _preprocess(self, text):
        text = text.lower()
        tokens = NB_TOKEN_PATTERN.findall(text)
        return [self.stemmer.stem(tok) for tok in tokens]

    def predict(self, text):
        if not self.available:
            return None

        tokens = self._preprocess(text)
        valid_counts = Counter(tok for tok in tokens if tok in self.word_probs)

        scores = []
        for class_idx, prior in enumerate(self.priors):
            score = math.log(prior)
            for tok, freq in valid_counts.items():
                score += freq * math.log(self.word_probs[tok][class_idx])
            scores.append(score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        # Softmax on log-scores for a simple confidence display.
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        z = sum(exp_scores)
        probs = [x / z for x in exp_scores]

        ranked = sorted(
            [{"label": NB_CATEGORIES[i], "prob": probs[i]} for i in range(len(NB_CATEGORIES))],
            key=lambda x: x["prob"],
            reverse=True,
        )

        return {
            "label": NB_CATEGORIES[best_idx],
            "confidence": probs[best_idx],
            "ranked": ranked,
        }


model = load_model()
nb_classifier = NaiveBayesTextClassifier(NB_PROBABILITY_PATH)


@app.get("/")
def index():
    return render_template("index.html", textcls_enabled=nb_classifier.available)


@app.post("/predict")
def predict():
    sentence = (request.form.get("sentence") or "").strip()
    if not sentence:
        return render_template(
            "index.html",
            sentence=sentence,
            error="Please input a sentence.",
            token_rows=[],
            entities=[],
            textcls_enabled=nb_classifier.available,
            textcls_result=None,
        )

    tokens = tokenize_sentence(sentence)
    labels = model.predict_sentence(tokens)
    entities = extract_entities(tokens, labels)
    token_rows = list(zip(tokens, labels))
    textcls_result = nb_classifier.predict(sentence) if nb_classifier.available else None

    return render_template(
        "index.html",
        sentence=sentence,
        token_rows=token_rows,
        entities=entities,
        textcls_enabled=nb_classifier.available,
        textcls_result=textcls_result,
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
