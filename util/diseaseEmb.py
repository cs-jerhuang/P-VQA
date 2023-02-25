import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import pickle


'''
Get the disease embeddings according to the pre-trained med_img model.
'''


class IntermediateLayerGetter(nn.ModuleDict):
    """ get the output of certain layers """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

def main():
    train_dir = 'data/train'
    save_path = "checkpoint/med_img.pth"
    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    train_set = datasets.ImageFolder(train_dir, transform=test_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    net = torch.load(save_path).cuda()
    return_layers = {"avgpool": "out"}
    backbone = IntermediateLayerGetter(net, return_layers)

    backbone.eval()
    train_list = []
    train_label_list = []
    with torch.no_grad():
        for train_data in trainloader:
            train_images, train_labels = train_data
            outputs = backbone(train_images.cuda())
            train_list.append(outputs["out"].squeeze())
            train_label_list.append(train_labels)
        train_embedding = torch.cat(train_list, dim=0)
        train_label = torch.cat(train_label_list, dim=0)

    print(train_embedding.shape, train_label.shape, len(train_set))
    train_embedding = train_embedding.cpu().numpy()
    train_label = train_label.cpu().numpy()
    
    # get disease embeddings
    center = {}
    for key in range(20):
        idx = np.where(train_label == key)
        center[key] = np.mean(train_embedding[idx], axis=0)

    result_list = []
    for i in center:
        node_feature = center[i]
        result_list.append(node_feature)
    result_list = np.vstack(result_list)
    print(result_list.shape)
    np.save("features/graph_init_features/diseases.npy", result_list)


if __name__ == "__main__":
    main()
