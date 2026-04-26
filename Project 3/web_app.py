from pathlib import Path
import pickle
import sys
import types

from flask import Flask, render_template, request

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


model = load_model()


@app.get("/")
def index():
    return render_template("index.html")


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
        )

    tokens = tokenize_sentence(sentence)
    labels = model.predict_sentence(tokens)
    entities = extract_entities(tokens, labels)
    token_rows = list(zip(tokens, labels))

    return render_template(
        "index.html",
        sentence=sentence,
        token_rows=token_rows,
        entities=entities,
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
