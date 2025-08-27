import os, zipfile, io, requests
import numpy as np, pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

def ensure_har_dataset(data_root: str) -> str:
    """اگر دیتاست نبود، دانلود و Extract می‌کند و مسیر فولدر «UCI HAR Dataset» را برمی‌گرداند."""
    target = os.path.join(data_root, "UCI HAR Dataset")
    if os.path.isdir(os.path.join(target, "train")):
        return target

    os.makedirs(data_root, exist_ok=True)
    url = os.environ.get("HAR_URL", UCI_URL)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(data_root)

    if os.path.isdir(os.path.join(target, "train")):
        return target

    # اگر پوشه نام دیگری داشت ولی ساختار train/test داشت
    for dn in os.listdir(data_root):
        cand = os.path.join(data_root, dn)
        if os.path.isdir(os.path.join(cand, "train")) and \
           os.path.isfile(os.path.join(cand, "features.txt")):
            return cand
    raise RuntimeError("UCI-HAR dataset not found after download!")

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
            if chunk:
                f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return out_dir

def load_har_features(dataset_dir: str):
    """X_train, y_train, X_test, y_test, subjects, label_map را برمی‌گرداند."""
    # اگر مسیر معتبر نبود، دانلودش کن
    if not os.path.isdir(os.path.join(dataset_dir, "train")) and \
       not os.path.isdir(os.path.join(dataset_dir, "Test")):
        dataset_dir = ensure_har_dataset(os.path.dirname(dataset_dir))

    # مسیرهای train/test
    train_dir = os.path.join(dataset_dir, "train") if os.path.isdir(os.path.join(dataset_dir, "train")) \
                else os.path.join(dataset_dir, "Train")
    test_dir  = os.path.join(dataset_dir, "test") if os.path.isdir(os.path.join(dataset_dir, "test")) \
                else os.path.join(dataset_dir, "Test")

    # لیبل‌ها
    label_map = pd.read_csv(
        os.path.join(dataset_dir, "activity_labels.txt"),
        sep=r"\s+", header=None, index_col=0
    )[1].to_dict()

    # نام ویژگی‌ها + یکتاسازی
    feat_names = pd.read_csv(
        os.path.join(dataset_dir, "features.txt"),
        sep=r"\s+", header=None, usecols=[1]
    ).squeeze("columns").tolist()

    from collections import defaultdict
    seen = defaultdict(int)
    uniq_names = []
    for nm in feat_names:
        seen[nm] += 1
        uniq_names.append(nm if seen[nm] == 1 else f"{nm}.{seen[nm]}")

    # Xها را بخوان (بدون names) و بعداً ستون‌ها را ست کن
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

    # حذف ردیف‌هایی که NaN دارند (برای جلوگیری از خطای SVC)
    def _drop_nans(X: pd.DataFrame, y: pd.Series):
        mask = ~X.isna().any(axis=1)
        return X.loc[mask].reset_index(drop=True), np.asarray(y)[mask]

    X_train, y_train = _drop_nans(X_train, y_train)
    X_test,  y_test  = _drop_nans(X_test,  y_test)

    subjects = {"train": subj_train.values, "test": subj_test.values}
    return X_train, y_train, X_test, y_test, subjects, label_map

def build_sequences(X: pd.DataFrame, y: pd.Series, subjects: pd.Series, max_len: int = 25):
    df = X.copy()
    df["y"] = np.asarray(y)
    df["subject"] = np.asarray(subjects)
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
