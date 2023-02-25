import torch
import numpy as np
import random
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report


'''
Pre-train the imgs of trainset to get img embeddings for embeddings of disease nodes.
'''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(1024)
    train_dir = 'data/train'
    valid_dir = 'data/val'
    test_dir = 'data/test'

    torch.cuda.set_device(0)
    total_epoch = 60
    val_epoch = 3

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

    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_set = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=20)

    net = models.resnet50(pretrained=True).cuda()

    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 20).cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # training
    best_acc = 0.0
    best_loss = 9999.9
    save_path = 'checkpoint/med_img.pth'
    for epoch in range(total_epoch):
        # training
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        for step, data in enumerate(trainloader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.cuda())
            loss = loss_function(logits, labels.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += (predict_y == labels.cuda()).sum().item()
            # print train process
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch {} train: {:^3.0f}%[{}->{}] loss: {:.4f}".format(epoch+1, (rate * 100), a, b, loss), end="")
        train_accurate = train_acc / len(train_set)
        print(" acc: {:.4f}".format(train_accurate))

        if epoch % val_epoch == 0:
            flag = ""
            net.eval()
            val_acc = 0.0
            val_loss = 0.0
            with torch.no_grad():
                for val_data in validloader:
                    val_images, val_labels = val_data
                    outputs = net(val_images.cuda())
                    loss = loss_function(outputs, val_labels.cuda())
                    val_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    val_acc += (predict_y == val_labels.cuda()).sum().item()
                val_loss = val_loss / len(validloader)
                val_accurate = val_acc / len(valid_set)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_loss = val_loss
                    torch.save(net, save_path)
                    flag = "*"
                if val_acc == best_acc and best_loss > val_loss:
                    best_acc = val_acc
                    best_loss = val_loss
                    torch.save(net, save_path)
                    flag = "*"
                print('[epoch %d] train_loss: %.4f train_acc: %.4f val_loss: %.4f  val_accuracy: %.4f' %
                      (epoch + 1, running_loss / step, train_accurate, val_loss, val_accurate) + flag)

    net = torch.load(save_path)
    net.eval()
    acc = 0.0
    test_all_label = []
    predict_all = []
    with torch.no_grad():
        for test_data in testloader:
            test_images, test_labels = test_data
            outputs = net(test_images.cuda())
            predict_y = torch.max(outputs, dim=1)[1]
            predict_all.append(predict_y.cpu().numpy())
            test_all_label.append(test_labels.numpy())
            acc += (predict_y == test_labels.cuda()).sum().item()
        test_all_label = np.hstack(test_all_label)
        predict_all = np.hstack(predict_all)
        test_accurate = acc / len(test_set)
        print(classification_report(test_all_label, predict_all, digits=5))
        print('Final: accuracy: %.3f' % test_accurate)
