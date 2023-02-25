import json
import torch
import numpy as np
from torch.utils.data import Dataset

from PIL import Image


class VQAData(Dataset):
    def __init__(self, setType, nodesNum, transform) -> None:
        self.transform = transform
        self.nodesNum = nodesNum
        with open("data/%s.json" % setType, "rt", encoding="utf8") as f:
            self.data = json.load(f)
        self.textFeature = torch.from_numpy(np.load("features/question_features/%s_sentence_text.npy" % setType)).float()

        self.label = self.labelGenerate([e["label"] for e in self.data])
        assert len(self.data) == self.textFeature.shape[0] == self.label.shape[0]

    def labelGenerate(self, rawLabel):
        label = np.zeros((len(rawLabel), self.nodesNum), np.float64)
        for row in range(len(rawLabel)):
            for i in rawLabel[row]:
                label[row, i] = 1.0
        return torch.from_numpy(label).float()

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        img = Image.open(self.data[index]["imagePath"]).convert("RGB")
        return self.textFeature[index], self.transform(img), self.label[index]