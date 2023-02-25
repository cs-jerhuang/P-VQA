import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from mkbn.model import VQANet, NetConfig
from util.data import VQAData
from util.tools import getThrehold, minAndRange, normalization

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def main():
    opt = NetConfig()
    print(opt)

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    valset = VQAData("val", opt.nodesNum, test_valid_transforms)
    testset = VQAData("test", opt.nodesNum, test_valid_transforms)
    valloader = DataLoader(valset, opt.batchSize, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, opt.batchSize, shuffle=True)

    net = VQANet(opt).cuda()
    
    net.load_state_dict(torch.load(opt.save_path))
    net.eval()
    
    # final test
    test_logit_all = []
    test_true_all = []
    val_logit_all = []
    val_true_all = []
    with torch.no_grad():
        print("infer on val set ...")
        for val_data in tqdm(valloader):
            val_texts, val_images, val_labels = val_data
            logit = net(val_images.cuda(), val_texts.cuda())
            val_logit_all.append(logit.cpu().numpy())
            val_true_all.append(val_labels.cpu().numpy())
        
        print("infer on test set ...")
        for test_data in tqdm(testloader):
            test_texts, test_images, test_labels = test_data
            logit = net(test_images.cuda(), test_texts.cuda())
            test_logit_all.append(logit.cpu().numpy())
            test_true_all.append(test_labels.cpu().numpy())
    
    pred = np.vstack(test_logit_all)
    gtrue = np.vstack(test_true_all)
    val_pred = np.vstack(val_logit_all)
    val_gtrue = np.vstack(val_true_all)
    val_min, val_range = minAndRange(val_pred)
    threhold = getThrehold(val_gtrue, normalization(val_pred, val_min, val_range))
    print("Val min:", val_min, "Val range:", val_range)
    print("Threhold", threhold)
    
    nor_pred = normalization(pred, val_min, val_range)
    tmp = (nor_pred > threhold)
    acc = accuracy_score(gtrue, tmp)
    recall = recall_score(gtrue, tmp, average="weighted", zero_division=0)
    precision = precision_score(gtrue, tmp, average="weighted", zero_division=0)
    f1 = f1_score(gtrue, tmp, average="weighted", zero_division=0)
    print(opt.save_path, "acc:", acc*100, "recall:", recall*100, "precision:", precision*100, "f1:", f1*100)

if __name__ == "__main__":
    main()