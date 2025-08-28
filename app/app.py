# app/app.py
import os, sys
from threading import Thread
from flask import Flask, request, render_template, jsonify
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(ROOT, "src")
sys.path.append(SRC_DIR)

from data import ensure_har_dataset, load_har_features, build_sequences  # noqa

app = Flask(__name__)

# -------- Globals ----------
MODELS_READY = False
LABEL_MAP = None
SVM_MODEL = None
RF_MODEL = None
SCALER = None
HMM_MODELS = {}

# -------- Train ------------
def train_models():
    global MODELS_READY, LABEL_MAP, SVM_MODEL, RF_MODEL, SCALER, HMM_MODELS

    ds_dir = ensure_har_dataset(os.path.join(ROOT, "data"))
    Xtr, ytr, Xte, yte, subjects, label_map = load_har_features(ds_dir)

    # آموزش سریع‌تر روی Render (اختیاری با env)
    if os.getenv("SMALL_TRAIN", "0") == "1":
        n = min(len(Xtr), 4000)
        Xtr, ytr = Xtr.iloc[:n], ytr[:n]

    LABEL_MAP = label_map

    SCALER = StandardScaler().fit(Xtr.values)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=0)
    svm.fit(SCALER.transform(Xtr.values), ytr)
    SVM_MODEL = svm

    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(Xtr.values, ytr)
    RF_MODEL = rf

    HMM_MODELS = {}
    train_seqs = build_sequences(Xtr, ytr, subjects["train"], max_len=25)
    for act_id in sorted(set(ytr)):
        seqs = [X for (X, y) in train_seqs if y == act_id]
        if not seqs:
            continue
        use = seqs[:50]
        lengths = [len(s) for s in use]
        data = np.vstack(use)
        m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=40, random_state=0)
        m.fit(data, lengths)
        HMM_MODELS[act_id] = m

    MODELS_READY = True


def warmup_async():
    try:
        train_models()
    except Exception as e:
        print("Warmup failed:", repr(e))


# -------- Routes ----------
@app.route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return ("ok", 200)

@app.route("/", methods=["GET", "HEAD"])
def index():
    # health-check سریع
    if request.method == "HEAD":
        return ("", 200)

    # اگر مدل‌ها آماده نیستند، آموزش را در پس‌زمینه استارت کن و صفحه را بالا بیاور
    if not MODELS_READY:
        Thread(target=warmup_async, daemon=True).start()
        return render_template("index.html", labels=None, warming=True)

    return render_template("index.html", labels=LABEL_MAP, warming=False)

@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_READY:
        return jsonify({"status": "warming",
                        "message": "Models are warming up. Please try again shortly."}), 503

    method = request.form.get("method", "svm")
    mode   = request.form.get("mode", "sample")

    ds_dir = ensure_har_dataset(os.path.join(ROOT, "data"))
    Xtr, ytr, Xte, yte, subjects, _ = load_har_features(ds_dir)

    if method == "svm":
        if mode == "upload":
            file = request.files.get("file")
            if not file:
                return "Upload CSV (one row, 561 features).", 400
            df = pd.read_csv(file)
            X = SCALER.transform(df.values)
        else:
            X = SCALER.transform(Xte.iloc[[0]].values)
        yhat = int(SVM_MODEL.predict(X)[0])
        result = {"method": "svm", "pred": yhat, "label": LABEL_MAP.get(yhat, str(yhat))}

    elif method == "rf":
        if mode == "upload":
            file = request.files.get("file")
            if not file:
                return "Upload a CSV with one row of 561 features.", 400
            df = pd.read_csv(file)
            X = df.values
        else:
            X = Xte.iloc[[0]].values
        yhat = int(RF_MODEL.predict(X)[0])
        result = {"method": "rf", "pred": yhat, "label": LABEL_MAP.get(yhat, str(yhat))}

    elif method == "hmm":
        test_seqs = build_sequences(Xte, yte, subjects["test"], max_len=25)
        if not test_seqs:
            return "No sequences available.", 500
        X_seq, true_lab = test_seqs[0]
        best_cls, best_score = None, -1e18
        for act_id, model in HMM_MODELS.items():
            try:
                score = model.score(X_seq)
            except Exception:
                score = -1e18
            if score > best_score:
                best_score, best_cls = score, act_id
        result = {
            "method": "hmm",
            "true": int(true_lab),
            "true_label": LABEL_MAP.get(int(true_lab), str(true_lab)),
            "pred": int(best_cls),
            "pred_label": LABEL_MAP.get(int(best_cls), str(best_cls)),
            "loglik": float(best_score),
        }
    else:
        return "Unknown method.", 400

    return render_template("index.html", labels=LABEL_MAP, result=result, warming=False)

if __name__ == "__main__":
    Thread(target=warmup_async, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
