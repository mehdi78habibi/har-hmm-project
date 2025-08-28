# src/data.py
import os, io, zipfile, requests
import numpy as np
import pandas as pd


UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"


def ensure_har_dataset(data_root: str) -> str:
    """
    اگر دیتاست UCI-HAR نصب نبود، آن را دانلود و extract می‌کند
    و مسیر فولدر اصلی «UCI HAR Dataset» را برمی‌گرداند.
    """
    target = os.path.join(data_root, "UCI HAR Dataset")
    if os.path.isdir(os.path.join(target, "train")):
        return target

    os.makedirs(data_root, exist_ok=True)
    url = os.environ.get("HAR_URL", UCI_URL)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(data_root)

    # اگر نام پوشه دقیقاً همین نبود، یکی را که ساختار train/test دارد پیدا کن
    if os.path.isdir(os.path.join(target, "train")):
        return target
    for dn in os.listdir(data_root):
        cand = os.path.join(data_root, dn)
        if os.path.isdir(os.path.join(cand, "train")) and \
           os.path.isfile(os.path.join(cand, "features.txt")):
            return cand

    raise RuntimeError("UCI-HAR dataset not found after download!")


def _unique_feature_names(names):
    from collections import defaultdict
    seen = defaultdict(int)
    out = []
    for n in names:
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n}.{seen[n]}")
    return out


def load_har_features(dataset_dir: str):
    """
    همه‌ی مسیرها را نسبت به dataset_dir می‌سازد. اگر نبود، دانلود می‌کند.
    خروجی: X_train, y_train, X_test, y_test, subjects(dict), label_map(dict)
    """
    # اطمینان از وجود دیتاست
    if not os.path.isdir(os.path.join(dataset_dir, "train")) \
       and not os.path.isdir(os.path.join(dataset_dir, "Test")):
        dataset_dir = ensure_har_dataset(os.path.dirname(dataset_dir))

    # سازگاری train/Test
    train_dir = os.path.join(dataset_dir, "train")
    test_dir  = os.path.join(dataset_dir, "test")
    if not os.path.isdir(train_dir): train_dir = os.path.join(dataset_dir, "Train")
    if not os.path.isdir(test_dir):  test_dir  = os.path.join(dataset_dir, "Test")

    # لیبل‌ها
    labels_path = os.path.join(dataset_dir, "activity_labels.txt")
    label_map = pd.read_csv(labels_path, sep=r"\s+", header=None,
                            names=["id","name"]).set_index("id")["name"].to_dict()

    # نام ویژگی‌ها و یکتاسازی آن‌ها
    feat_names = pd.read_csv(os.path.join(dataset_dir, "features.txt"),
                             sep=r"\s+", header=None, usecols=[1]) \
                    .squeeze("columns").tolist()
    feat_names = _unique_feature_names(feat_names)

    # X‌ها را با header=None بخوان و بعد ستون‌ها را ست کن
    X_train = pd.read_csv(os.path.join(train_dir, "X_train.txt"),
                          sep=r"\s+", header=None, engine="python")
    X_test  = pd.read_csv(os.path.join(test_dir,  "X_test.txt"),
                          sep=r"\s+", header=None, engine="python")
    X_train.columns = feat_names
    X_test.columns  = feat_names

    # برچسب‌ها و شناسه‌ی افراد
    y_train = pd.read_csv(os.path.join(train_dir, "y_train.txt"),
                          sep=r"\s+", header=None).squeeze("columns")
    y_test  = pd.read_csv(os.path.join(test_dir,  "y_test.txt"),
                          sep=r"\s+", header=None).squeeze("columns")

    # نام‌های متفاوت subject را پوشش بده
    def _read_subj(d, fn1, fn2):
        p = os.path.join(d, fn1)
        if not os.path.isfile(p): p = os.path.join(d, fn2)
        return pd.read_csv(p, sep=r"\s+", header=None).squeeze("columns")

    subj_train = _read_subj(train_dir, "subject_train.txt", "subject_id_train.txt")
    subj_test  = _read_subj(test_dir,  "subject_test.txt",  "subject_id_test.txt")
    subjects = {"train": subj_train.values, "test": subj_test.values}

    # سطرهایی که NaN دارند را حذف کن (برای سازگاری با نسخه‌های جدید pandas/numexpr)
    def _drop_nans(X, y):
        mask = ~X.isna().any(axis=1)
        return X.loc[mask].reset_index(drop=True), np.asarray(y)[mask]

    X_train, y_train = _drop_nans(X_train, y_train)
    X_test,  y_test  = _drop_nans(X_test,  y_test)

    return X_train, y_train, X_test, y_test, subjects, label_map


def build_sequences(X: pd.DataFrame, y: pd.Series, subjects: np.ndarray, max_len: int = 25):
    """
    آرایه‌های دنباله‌ای برای HMM می‌سازد.
    هر قطعه حداقل ۵ نمونه داشته باشد.
    """
    df = X.copy()
    df["y"] = np.asarray(y)
    df["subject"] = subjects
    df["idx"] = np.arange(len(df))
    df.sort_values(["subject", "idx"], inplace=True)

    seqs = []
    for (subj, act), g in df.groupby(["subject", "y"], sort=False):
        arr = g.drop(columns=["y", "subject", "idx"]).values
        for i in range(0, len(arr), max_len):
            chunk = arr[i:i+max_len]
            if len(chunk) >= 5:
                seqs.append((chunk, int(act)))
    return seqs
