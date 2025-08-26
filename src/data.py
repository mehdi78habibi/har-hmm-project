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

def load_har_features(dataset_dir: str):

    # هم train/Train و هم test/Test را بپذیر
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir  = os.path.join(dataset_dir, 'test')
    if not os.path.isdir(train_dir): train_dir = os.path.join(dataset_dir, 'Train')
    if not os.path.isdir(test_dir):  test_dir  = os.path.join(dataset_dir, 'Test')

    # نام‌گذاری ویژگی‌ها
    features = pd.read_csv(os.path.join(dataset_dir, "features.txt"),
                           sep=r"\s+", header=None, names=["idx","name"]).set_index("idx")
    feat_names = features["name"].tolist()

    # ماتریس‌های ویژگی
    X_train = pd.read_csv(os.path.join(train_dir, "X_train.txt"),
                          sep=r"\s+", header=None, names=feat_names)
    y_train = pd.read_csv(os.path.join(train_dir, "y_train.txt"),
                          sep=r"\s+", header=None)[0]

    X_test = pd.read_csv(os.path.join(test_dir, "X_test.txt"),
                         sep=r"\s+", header=None, names=feat_names)
    y_test = pd.read_csv(os.path.join(test_dir, "y_test.txt"),
                         sep=r"\s+", header=None)[0]

    # مسیرهای subject: هم نامِ UCI-HAR (subject_train.txt)
    # و هم نامِ SBHAR (subject_id_train.txt) را پشتیبانی کن
    subj_train_path = os.path.join(train_dir, "subject_train.txt")
    if not os.path.isfile(subj_train_path):
        subj_train_path = os.path.join(train_dir, "subject_id_train.txt")

    subj_test_path = os.path.join(test_dir, "subject_test.txt")
    if not os.path.isfile(subj_test_path):
        subj_test_path = os.path.join(test_dir, "subject_id_test.txt")

    subj_train = pd.read_csv(subj_train_path, sep=r"\s+", header=None)[0].reset_index(drop=True)
    subj_test  = pd.read_csv(subj_test_path,  sep=r"\s+", header=None)[0].reset_index(drop=True)

    # نگاشت شناسهٔ فعالیت به نام
    labels = pd.read_csv(os.path.join(dataset_dir, "activity_labels.txt"),
                         sep=r"\s+", header=None, names=["id","name"]).set_index("id")["name"].to_dict()

    subjects = {"train": subj_train, "test": subj_test}
    return X_train, y_train, X_test, y_test, subjects, labels


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
