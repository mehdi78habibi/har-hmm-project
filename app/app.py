import os
import sys
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM

# --- Paths -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # app/
ROOT     = os.path.dirname(BASE_DIR)                    # project root
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
SRC_DIR   = os.path.join(ROOT, "src")
os.makedirs(MODEL_DIR, exist_ok=True)

# برای import ماژول‌های src
sys.path.append(SRC_DIR)
from data import ensure_har_dataset, load_har_features, build_sequences  # noqa

# --- Flask app ---------------------------------------------
app = Flask(__name__)

# --- Globals (مدل‌ها و داده‌ها) ----------------------------
MODELS_READY = False

LABEL_MAP   = None
SVM_MODEL   = None
RF_MODEL    = None
SCALER      = None
HMM_MODELS  = {}

# برای حالت «نمونه‌ی آماده»
TEST_X = None
TEST_y = None
SUBJECTS = None


def train_models():
    """دانلود/آماده‌سازی داده و آموزش مدل‌ها (فقط یک بار)."""
    global LABEL_MAP, SVM_MODEL, RF_MODEL, SCALER, HMM_MODELS
    global TEST_X, TEST_y, SUBJECTS, MODELS_READY

    # دیتاست را اگر نبود دانلود کن
    ds_dir = ensure_har_dataset(DATA_DIR)

    # بارگذاری ویژگی‌ها
    X_train, y_train, X_test, y_test, subjects, label_map = load_har_features(ds_dir)
    LABEL_MAP = label_map
    TEST_X, TEST_y, SUBJECTS = X_test, y_test, subjects

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

    # --- HMM های کلاس‌-محور (sequence likelihood) ---
    HMM_MODELS = {}
    train_seqs = build_sequences(X_train, y_train, subjects["train"], max_len=25)
    for act_id in sorted(set(y_train)):
        seqs = [X for (X, yy) in train_seqs if yy == act_id]
        if not seqs:
            continue
        # محدود به چند توالی برای سرعت
        use = seqs[:50]
        lengths = [len(s) for s in use]
        data = np.vstack(use)
        m = GaussianHMM(
            n_components=3, covariance_type="diag", n_iter=40, random_state=0
        )
        m.fit(data, lengths)
        HMM_MODELS[act_id] = m

    MODELS_READY = True


def ensure_models_ready():
    global MODELS_READY
    if not MODELS_READY:
        train_models()


# --- Health check برای Render -------------------------------
@app.route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return ("ok", 200)


# --- UI ----------------------------------------------------
@app.route("/", methods=["GET", "HEAD"])
def index():
    # Render روی / هم HEAD می‌زند؛ پاسخ سریع بده
    if request.method == "HEAD":
        return ("", 200)
    ensure_models_ready()
    return render_template("index.html", labels=LABEL_MAP)


# --- Prediction endpoint -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    ensure_models_ready()
    method = request.form.get("method", "svm")
    mode   = request.form.get("mode", "sample")

    # برای ساده‌گی از داده‌های تستِ از پیش بارگذاری شده استفاده می‌کنیم
    X_test = TEST_X
    y_test = TEST_y
    subjects = SUBJECTS

    if method == "svm":
        # برداری با ۵۶۱ ویژگی
        if mode == "upload":
            file = request.files.get("file")
            if not file:
                return "Upload CSV (one row, 561 features).", 400
            df = pd.read_csv(file)
            X = SCALER.transform(df.values)
        else:
            X = SCALER.transform(X_test.iloc[[0]].values)

        yhat = int(SVM_MODEL.predict(X)[0])
        result = {"method": "svm", "pred": yhat, "label": LABEL_MAP.get(yhat, str(yhat))}

    elif method == "rf":
        if mode == "upload":
            file = request.files.get("file")
            if not file:
                return "Upload a CSV with one row of 561 features.", 400
            df = pd.read_csv(file)
            X = df.values  # بدون اسکیل
        else:
            X = X_test.iloc[[0]].values

        yhat = int(RF_MODEL.predict(X)[0])
        result = {"method": "rf", "pred": yhat, "label": LABEL_MAP.get(yhat, str(yhat))}

    elif method == "hmm":
        # HMM روی توالی کار می‌کند؛ از یک توالی آماده از داده‌ی تست استفاده می‌کنیم
        test_seqs = build_sequences(X_test, y_test, subjects["test"], max_len=25)
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
            "pred": int(best_cls) if best_cls is not None else -1,
            "pred_label": LABEL_MAP.get(int(best_cls), str(best_cls))
            if best_cls is not None
            else "N/A",
            "loglik": float(best_score),
        }

    else:
        return "Unknown method.", 400

    return render_template("index.html", labels=LABEL_MAP, result=result)


# --- Local run ---------------------------------------------
if __name__ == "__main__":
    train_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
