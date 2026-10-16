"""Minimal inference for the exported model. Usage: python predict.py "dholuo text" """
import json, sys
from pathlib import Path
import numpy as np
from preprocess import TEXT_VARIANTS

HERE = Path(__file__).parent
META = json.load(open(HERE / "label_map.json"))
NORM = TEXT_VARIANTS[META["text_variant"]]

def load():
    if META["kind"] == "transformer":
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(HERE); model = AutoModelForSequenceClassification.from_pretrained(HERE).eval()
        def f(texts):
            enc = tok([NORM(t) for t in texts], truncation=True, max_length=META["max_len"], padding=True, return_tensors="pt")
            with torch.no_grad():
                return torch.softmax(model(**enc).logits, -1).numpy()
        return f
    if META["kind"] == "sklearn":
        import joblib
        pipe = joblib.load(HERE / "model.joblib"); clf = pipe.named_steps["clf"]
        def f(texts):
            x = [NORM(t) for t in texts]
            if hasattr(clf, "predict_proba"):
                raw = pipe.predict_proba(x)
            else:
                z = pipe.decision_function(x) / META.get("temperature", 1.0)
                z = z - z.max(1, keepdims=True); raw = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            out = np.zeros((len(x), len(META["labels"]))); out[:, clf.classes_] = raw
            return out
        return f
    raise NotImplementedError("Load this model kind with the notebook code (bilstm/frozen).")

if __name__ == "__main__":
    f = load(); p = f([" ".join(sys.argv[1:])])[0]
    for i in np.argsort(p)[::-1][:3]:
        print(f"{META['labels'][i]}: {p[i]:.3f}")
