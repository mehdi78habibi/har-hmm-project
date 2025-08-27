import os, zipfile, io, requests
import numpy as np, pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

def download_har(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    out_dir = os.path.join(root, "UCI HAR Dataset")
    if os.path.isdir(out_dir) and os.path.isfile(os.path.join(out_dir, "README.txt")):
        return out_dir
    r = requests.get(UCI_URL, stream=True, timeout=120)
    r.raise_for_status()
    zip_path = os.path.join(root, "UCI_HAR_Dataset.zip")
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return out_dir

def load_har_features(base_dir):
    import pandas as pd, os
    # لیبل‌ها
    act_path = os.path.join(base_dir, "activity_labels.txt")
    label_map = pd.read_csv(act_path, sep=r"\s+", header=None, index_col=0)[1].to_dict()

    # نام ویژگی‌ها + یکتاسازی (name, name.2, name.3, ...)
    feat_path = os.path.join(base_dir, "features.txt")
    feat_names = pd.read_csv(feat_path, sep=r"\s+", header=None, usecols=[1]) \
                   .squeeze("columns").tolist()
    from collections import defaultdict
    seen = defaultdict(int)
    uniq_names = []
    for nm in feat_names:
        seen[nm] += 1
        uniq_names.append(nm if seen[nm] == 1 else f"{nm}.{seen[nm]}")

    train_dir = os.path.join(base_dir, "train")
    test_dir  = os.path.join(base_dir, "test")

    # X ها را بدون names بخوان و بعداً ستون‌ها را ست کن (سازگار با pandas جدید)
    X_train = pd.read_csv(os.path.join(train_dir, "X_train.txt"),
                          sep=r"\s+", header=None, engine="python")
    X_test  = pd.read_csv(os.path.join(test_dir,  "X_test.txt"),
                          sep=r"\s+", header=None, engine="python")
    X_train.columns = uniq_names
    X_test.columns  = uniq_names

    # برچسب‌ها و سابجکت‌ها
    y_train = pd.read_csv(os.path.join(train_dir, "y_train.txt"),
                          sep=r"\s+", header=None).squeeze("columns")
    y_test  = pd.read_csv(os.path.join(test_dir,  "y_test.txt"),
                          sep=r"\s+", header=None).squeeze("columns")
    subj_train = pd.read_csv(os.path.join(train_dir, "subject_train.txt"),
                             sep=r"\s+", header=None).squeeze("columns")
    subj_test  = pd.read_csv(os.path.join(test_dir,  "subject_test.txt"),
                             sep=r"\s+", header=None).squeeze("columns")
    subjects = {"train": subj_train.values, "test": subj_test.values}

    return X_train, y_train.values, X_test, y_test.values, subjects, label_map 


def build_sequences(X: pd.DataFrame, y: pd.Series, subjects: pd.Series, max_len: int = 25):
    df = X.copy()
    df["y"] = y.values
    df["subject"] = subjects.values
    df["idx"] = np.arange(len(df))
    df.sort_values(["subject","idx"], inplace=True)
    seqs = []

    for (subj, act), g in df.groupby(["subject","y"], sort=False):
        arr = g.drop(columns=["y","subject","idx"]).values
        for i in range(0, len(arr), max_len):
            chunk = arr[i:i+max_len]
            if len(chunk) >= 5:
                seqs.append((chunk, int(act)))
    return seqs
