import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def checkCorrect(pred, labels):
    true_num = 0
    for i, value in enumerate(pred):
        ltmp = np.where(labels[i] == 1.0)[0].tolist()
        if value in ltmp:
            true_num += 1
    return true_num


def minAndRange(data):
    _range = np.max(data) - np.min(data)
    return np.min(data), _range


def getThrehold(label, logit):
    max_f1 = 0
    threhold = 0
    print("search for a threhold on val set ...")
    for i in tqdm(range(1, 100)):
        tmp_threhold = i / 100
        vpred = (logit > tmp_threhold)
        tmp_f1 = f1_score(label, vpred, average="weighted")
        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            threhold = tmp_threhold
    return threhold


def normalization(data, min, _range):
    return (data - min) / _range
