import os
import sys
from threading import Thread

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # app/
ROOT     = os.path.dirname(BASE_DIR)                    # project root
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
SRC_DIR   = os.path.join(ROOT, "src")
os.makedirs(MODEL_DIR, exist_ok=True)

# برای import از فولدر src
sys.path.append(SRC_DIR)
from data import ensure_har_dataset, load_har_features, build_sequences  # noqa

app = Flask(__name__)

# ---------- Globals ----------
BASE_READY = False     # SVM / RF آماده
HMM_READY  = False     # HMMها آماده
LABEL_MAP  = None

SVM_MODEL  = None
RF_MODEL   = None
SCALER     = None

HMM_MODELS = {}        # dict: class_id -> GaussianHMM

# داده‌های تست (برای دموی سریع)
Xte = None
yte = None
SUBJECTS = None

# داده‌های train برای warmup HMM
Xtr = None
ytr = None


def train_base_models():
    """
    دانلود/بارگذاری دیتا، آماده‌سازی، و آموزش SVM و RandomForest
    (غیربلاکینگ برای UI).
    """
    global BASE_READY, LABEL_MAP, SVM_MODEL, RF_MODEL, SCALER
    global Xte, yte, SUBJECTS, Xtr, ytr

    # اطمینان از وجود دیتاست
    ds_dir = ensure_har_dataset(DATA_DIR)

    # بارگذاری ویژگی‌ها
    X_train, y_train, X_test, y_test, subjects, label_map = load_har_features(ds_dir)
    LABEL_MAP = label_map

    # ذخیره‌ی مجموعه‌های لازم برای UI و HMM-warmup
    Xtr, ytr = X_train, y_train
    Xte, yte = X_test, y_test
    SUBJECTS = subjects

    # --- SVM (با اسکیل) ---
    SCALER = StandardScaler().fit(X_train.values)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=0)
    svm.fit(SCALER.transform(X_train.values), y_train)
    SVM_MODEL = svm

    # --- RandomForest (بدون اسکیل) ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=0, n_jobs=-1
    )
    rf.fit(X_train.values, y_train)
    RF_MODEL = rf

    BASE_READY = True


def warmup_async():
    """
    آموزش HMMها در پس‌زمینه تا UI سریع بالا بیاید و
    شاخه‌ی HMM بعد از کمی صبر آماده شود.
    """
    global HMM_READY, HMM_MODELS

    # اگر مدل‌های پایه هنوز آماده نیستند، اول آن‌ها را بساز
    if not BASE_READY:
        train_base_models()

    # آموزش HMMهای کلاس‌به‌کلاس (sequence likelihood)
    hmm_dict = {}
    train_seqs = build_sequences(Xtr, ytr, SUBJECTS["train"], max_len=25)
    for act_id in sorted(set(ytr)):
        seqs = [X for (X, yy) in train_seqs if yy == act_id]
        if not seqs:
            continue
        # برای سرعت، فقط بخشی از توالی‌ها را می‌گیریم
        use = seqs[:50]
        lengths = [len(s) for s in use]
        data = np.vstack(use)
        m = GaussianHMM(
            n_components=3, covariance_type="diag", n_iter=40, random_state=0
        )
        m.fit(data, lengths)
        hmm_dict[act_id] = m

    HMM_MODELS = hmm_dict
    HMM_READY = True


def ensure_base_ready():
    """اگر مدل‌های پایه آماده نیستند، بساز (برای اجرای لوکال/پس‌زمینه)."""
    if not BASE_READY:
        train_base_models()


# ---------- Health ----------
@app.route("/healthz", methods=["GET", "HEAD"])
def healthz():
    # پاسخ سریع برای health-check
    return ("ok", 200)


# ---------- UI ----------
@app.route("/", methods=["GET", "HEAD"])
def index():
    if request.method == "HEAD":
        return ("", 200)

    # اگر مدل‌های پایه آماده نیستند، اینجا همزمان بساز تا فرم فعال شود
    ensure_base_ready()

    # اگر BASE_READY True شد، فرم باید فعال باشد
    disable_controls = not BASE_READY
    # حتی اگر HMM آماده نباشد فقط بنر نمایش داده شود
    warming_banner = not HMM_READY

    return render_template(
        "index.html",
        labels=LABEL_MAP,
        disable_controls=disable_controls,
        warming_banner=warming_banner,
    )


@app.route("/predict", methods=["POST"])
def predict():
    ensure_base_ready()

    method = request.form.get("method", "svm")
    mode   = request.form.get("mode", "sample")

    # --------- SVM ----------
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

    # --------- RF ----------
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

    # --------- HMM ----------
    elif method == "hmm":
        if not HMM_READY:
            # HMM هنوز در حال آماده‌سازی است؛ فرم فعال بماند اما پیام نمایش بده
            return render_template(
                "index.html",
                labels=LABEL_MAP,
                disable_controls=False,
                warming_banner=True,
                result={
                    "method": "hmm",
                    "status": "warming_up",
                    "msg": "مدل HMM هنوز در حال آماده‌سازی است. بعداً دوباره امتحان کنید.",
                },
            )

        test_seqs = build_sequences(Xte, yte, SUBJECTS["test"], max_len=25)
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

    # بعد از predict، فرم فعال بماند؛ فقط اگر HMM آماده نیست بنر را نشان بده
    return render_template(
        "index.html",
        labels=LABEL_MAP,
        result=result,
        disable_controls=False,
        warming_banner=not HMM_READY,
    )

@app.route("/status")
def status():
    return {
        "BASE_READY": BASE_READY,
        "HMM_READY": HMM_READY
    }, 200


# --- start warmup thread on import (non-blocking, idempotent) ---
try:
    _WARMUP_STARTED  # اگر قبلاً ست شده، خطا نمی‌دهیم
except NameError:
    _WARMUP_STARTED = True
    # warmup خودش اگر لازم باشد train_base_models را صدا می‌زند
    Thread(target=warmup_async, daemon=True).start()


# ---------- Local run ----------
if __name__ == "__main__":
    # برای اجرای لوکال: پایه‌ها را بساز و HMM را در پس‌زمینه گرم کن
    ensure_base_ready()
    Thread(target=warmup_async, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import jsonify

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"base_ready": BASE_READY, "hmm_ready": HMM_READY})
