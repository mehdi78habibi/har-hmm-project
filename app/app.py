import os
from flask import Flask, request, render_template
import numpy as np, pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT, 'data')
MODEL_DIR = os.path.join(ROOT, 'models')
SRC_DIR = os.path.join(ROOT, 'src')
os.makedirs(MODEL_DIR, exist_ok=True)

import sys
sys.path.append(SRC_DIR)
from data import download_har, load_har_features, build_sequences
from hmmlearn.hmm import GaussianHMM

app = Flask(__name__)

MODELS_READY = False

def ensure_models_ready():
    global MODELS_READY
    if not MODELS_READY:
        train_models()
        MODELS_READY = True

LABEL_MAP = None
SVM_MODEL = None
RF_MODEL = None
SCALER = None
HMM_MODELS = {}

def train_models():
    global LABEL_MAP, SVM_MODEL, RF_MODEL, SCALER, HMM_MODELS
    try:
        ds_dir = download_har(os.path.join(ROOT, 'data'))
    except Exception:
        ds_dir = os.path.join(ROOT, 'data', 'UCI HAR Dataset')
    X_train, y_train, X_test, y_test, subjects, label_map = load_har_features(ds_dir)
    LABEL_MAP = label_map
    SCALER = StandardScaler().fit(X_train.values)
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(SCALER.transform(X_train.values), y_train)
    SVM_MODEL = svm
    train_seqs = build_sequences(X_train, y_train, subjects['train'], max_len=25)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0, n_jobs=-1)
    rf.fit(X_train.values, y_train)  # توجه: بدون scaler
    RF_MODEL = rf
    HMM_MODELS = {}
    for act_id in sorted(set(y_train)):
        seqs = [X for (X,y) in train_seqs if y == act_id]
        if not seqs:
            continue
        lengths = [len(s) for s in seqs[:50]]
        data = np.vstack(seqs[:50])
        m = GaussianHMM(n_components=3, covariance_type='diag', n_iter=40, random_state=0)
        m.fit(data, lengths)
        HMM_MODELS[act_id] = m

@app.route('/', methods=['GET'])
def index():
    ensure_models_ready()
    return render_template('index.html', labels=LABEL_MAP)

@app.route('/predict', methods=['POST'])
def predict():
    ensure_models_ready()
    method = request.form.get('method', 'svm')
    mode   = request.form.get('mode', 'sample')

    try:
        ds_dir = download_har(os.path.join(ROOT, 'data'))
    except Exception:
        ds_dir = os.path.join(ROOT, 'data', 'UCI HAR Dataset')

    X_train, y_train, X_test, y_test, subjects, label_map = load_har_features(ds_dir)

    if method == 'svm':
        # --- SVM ---
        if mode == 'upload':
            file = request.files.get('file')
            if not file:
                return "Upload CSV (one row, 561 features).", 400
            df = pd.read_csv(file)
            X = SCALER.transform(df.values)
        else:
            X = SCALER.transform(X_test.iloc[[0]].values)

        yhat = SVM_MODEL.predict(X)[0]
        result = {
            'method': 'svm',
            'pred': int(yhat),
            'label': LABEL_MAP.get(int(yhat), str(yhat))
        }

    elif method == 'rf':
        # --- Random Forest ---
        if mode == 'upload':
            file = request.files.get('file')
            if not file:
                return "Upload a CSV with one row of 561 features.", 400
            df = pd.read_csv(file)
            X = df.values  # بدون اسکیلر
        else:
            X = X_test.iloc[[0]].values

        yhat = RF_MODEL.predict(X)[0]
        result = {
            'method': 'rf',
            'pred': int(yhat),
            'label': LABEL_MAP.get(int(yhat), str(yhat))
        }

    elif method == 'hmm':
        # --- HMM (likelihood سکانس) ---
        test_seqs = build_sequences(X_test, y_test, subjects['test'], max_len=25)
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
            'method': 'hmm',
            'true': int(true_lab),
            'true_label': LABEL_MAP.get(int(true_lab), str(true_lab)),
            'pred': int(best_cls),
            'pred_label': LABEL_MAP.get(int(best_cls), str(best_cls)),
            'loglik': float(best_score)
        }

    else:
        return "Unknown method.", 400

    return render_template('index.html', labels=LABEL_MAP, result=result)

if __name__ == '__main__':
    train_models()
    app.run(host='0.0.0.0', port=5000, debug=True)



