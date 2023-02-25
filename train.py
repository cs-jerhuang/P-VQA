import torch
from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from mkbn.model import VQANet, NetConfig
from util.data import VQAData
from util.tools import getThrehold, minAndRange, normalization, checkCorrect

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import random
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def update_learning_rate(epoch, optimizer, config):
    if epoch < len(config.lr_warmup_steps):
        optimizer.param_groups[0]['lr'] = config.lr_warmup_steps[epoch]
    elif epoch in config.lr_decay_epochs:
        optimizer.param_groups[0]['lr'] *= config.lr_decay_rate

def main():
    opt = NetConfig()
    print(opt)

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    trainset = VQAData("train", opt.nodesNum, train_transforms)
    valset = VQAData("val", opt.nodesNum, test_valid_transforms)
    testset = VQAData("test", opt.nodesNum, test_valid_transforms)
    trainloader = DataLoader(trainset, opt.batchSize, shuffle=True, num_workers=8)
    valloader = DataLoader(valset, opt.batchSize, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, opt.batchSize, shuffle=True)

    net = VQANet(opt).cuda()
    criterion = nn.MultiLabelSoftMarginLoss().cuda()

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    if opt.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=opt.lr)
    if opt.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=opt.lr)
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    if opt.optimizer == 'Adamax':
        optimizer = optim.Adamax(net.parameters(), lr=opt.lr)

    best_loss = 9999.9
    
    for epoch in range(opt.epochs):
        estart = time.time()
        update_learning_rate(epoch, optimizer, opt)
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        for step, data in enumerate(trainloader, start=0):
            texts, images, labels = data
            optimizer.zero_grad()
            logit = net(images.cuda(), texts.cuda())
            loss = criterion(logit, labels.cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            predict_y = torch.argmax(logit, dim=1)
            train_acc += checkCorrect(predict_y.cpu().numpy(), labels.cpu().numpy())
            # print train process
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch {}/{} train: {:^3.0f}%[{}->{}] loss: {:.4f}".format(epoch+1, opt.epochs, (rate * 100), a, b, loss), end="")
        train_accurate = train_acc / len(trainset)
        eend = time.time()
        print(" acc: {:.4f} epoch_time: {:.2f}s".format(train_accurate, eend-estart))

        if epoch % opt.val_epoch == 0:
            flag = ""
            net.eval()
            val_acc = 0.0
            val_loss = 0.0
            with torch.no_grad():
                for val_data in valloader:
                    val_texts, val_images, val_labels = val_data
                    logit = net(val_images.cuda(), val_texts.cuda())
                    loss = criterion(logit, val_labels.cuda())
                    val_loss += loss.item()
                    predict_y = torch.argmax(logit, dim=1)
                    val_acc += checkCorrect(predict_y.cpu().numpy(), val_labels.cpu().numpy())
                val_loss = val_loss / len(valloader)
                val_accurate = val_acc / len(valset)

                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(net.state_dict(), opt.save_path)
                    flag = "*"
                print('[epoch %d] train_loss: %.4f train_acc: %.5f val_loss: %.4f  val_accuracy: %.5f' %
                      (epoch + 1, running_loss / step, train_accurate, val_loss, val_accurate) + flag)
    
    net.load_state_dict(torch.load(opt.save_path))
    net.eval()
    
    # final test
    test_logit_all = []
    test_true_all = []
    val_logit_all = []
    val_true_all = []
    with torch.no_grad():
        for val_data in tqdm(valloader):
            val_texts, val_images, val_labels = val_data
            logit = net(val_images.cuda(), val_texts.cuda())
            val_logit_all.append(logit.cpu().numpy())
            val_true_all.append(val_labels.cpu().numpy())
        
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
    # seed = random.randint(0, 1000)
    seed = 336
    setup_seed(seed)
    main()
