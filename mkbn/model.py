import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torchvision import models
from mkbn.baseMCA import MCA_ED, AttFlat, LayerNorm
import numpy as np


class NetConfig:
    def __init__(self) -> None:
        self.word_emb_dim = 768

        self.gru_layers = 2
        self.img_dim = 1024
        self.num_hid = 512
        self.gamma = 2

        # train set
        self.optimizer = "Adamax"
        self.lr = 7e-4
        self.lr_decay_step = 2  # Decay every this many epochs
        self.lr_decay_rate = .7
        self.lr_decay_epochs = range(15, 25, self.lr_decay_step)
        self.lr_warmup_steps = [0.5 * self.lr, 1.0 * self.lr, 1.0 * self.lr, 1.5 * self.lr, 2.0 * self.lr]
        self.grad_clip = 50

        self.nodesNum = 419
        self.batchSize = 100
        self.epochs = 100
        self.val_epoch = 1

        # KG path
        self.hgat_layers = 2
        self.diseases_path = "features/graph_init_features/diseases.npy"
        self.rels_path = "features/graph_init_features/rels.npy"
        self.attributes_path = "features/graph_init_features/attributes.npy"
        self.d2d_path = "features/graph_init_features/d2d.npy"
        self.d2a_path = "features/graph_init_features/d2a.npy"
        self.e2a_path = "features/graph_init_features/e2a.npy"
        self.a2a_path = "features/graph_init_features/a2a.npy"

        self.save_path = "checkpoint/MKBN.pth"

    def __str__(self):
        content = "_".join(["%s=%s" % (k, v) for k, v in vars(self).items()])
        return content


class MCANet(nn.Module):
    def __init__(self, num_hid, head_num=4, FF_SIZE=2048, DROPOUT_R=0.1, layers=3):
        super(MCANet, self).__init__()
        self.mca = MCA_ED(num_hid, head_num, FF_SIZE, DROPOUT_R, layers)
        
        # Flatten to vector
        self.attflat_img = AttFlat(num_hid, num_hid, 1, DROPOUT_R, num_hid)
        self.attflat_ques = AttFlat(num_hid, num_hid, 1, DROPOUT_R, num_hid)
    
    def forward(self, img, ques, img_mask=None, ques_mask=None):
        ques_feat, img_feat = self.mca(ques, img, ques_mask, img_mask)
        ques_feat = self.attflat_ques(ques_feat, ques_mask)
        img_feat = self.attflat_img(img_feat, img_mask)
        out = ques_feat+img_feat
        return out


class VQANet(nn.Module):
    def __init__(self, config: NetConfig):
        super(VQANet, self).__init__()
        self.config = config
        self.glimpse = config.gamma

        self.img_enc = models.resnet50(pretrained=True)
        inchannel = self.img_enc.fc.in_features
        self.img_enc.fc = nn.Linear(inchannel, config.img_dim)
        self.img_trans = nn.Linear(config.img_dim, config.num_hid)
        self.q_emb = nn.GRU(config.word_emb_dim, int(config.num_hid / 2), config.gru_layers, batch_first=True, bidirectional=True)
        
        self.vqFus = MCANet(config.num_hid)
        self.vnFus = MCANet(config.num_hid)
        self.qeFus = MCANet(config.num_hid)

        self.disease_embeddings = torch.from_numpy(np.load(config.diseases_path)).cuda()
        self.rel_embeddings = torch.from_numpy(np.load(config.rels_path)).cuda()
        self.attributes_embeddings = torch.from_numpy(np.load(config.attributes_path)).cuda()
        self.d2d = torch.from_numpy(np.load(config.d2d_path)).cuda()
        self.d2a = torch.from_numpy(np.load(config.d2a_path)).cuda()
        self.e2a = torch.from_numpy(np.load(config.e2a_path)).cuda()
        self.a2a = torch.from_numpy(np.load(config.a2a_path)).cuda()

        self.d2a_trans = nn.ModuleList()
        self.d2a_attn = nn.ModuleList()
        self.e2a_trans = nn.ModuleList()
        self.e2a_attn = nn.ModuleList()
        self.a2a_trans = nn.ModuleList()
        self.a2a_attn = nn.ModuleList()
        self.d2d_attn = nn.ModuleList()
        self.diseases_trans = nn.Linear(self.disease_embeddings.shape[1], config.num_hid)
        self.rels_trans = nn.Linear(self.rel_embeddings.shape[1], config.num_hid)
        self.attributes_trans = nn.Linear(self.attributes_embeddings.shape[1], config.num_hid)
        for _ in range(config.hgat_layers):
            self.d2a_trans.append(nn.Linear(config.num_hid, config.num_hid))
            self.d2a_attn.append(nn.Linear(config.num_hid, self.disease_embeddings.shape[0]))
            
            self.e2a_trans.append(nn.Linear(config.num_hid, config.num_hid))
            self.e2a_attn.append(nn.Linear(config.num_hid, self.rel_embeddings.shape[0]))
            
            self.a2a_trans.append(nn.Linear(config.num_hid, config.num_hid))
            self.a2a_attn.append(nn.Linear(config.num_hid, self.attributes_embeddings.shape[0]))

        self.norm = LayerNorm(config.num_hid * 3)
        self.classifier = nn.Linear(config.num_hid * 3, config.nodesNum, config)

    def aggregate(self, nodes, adj, trans_layer, attn_layers):
        trans_feat = trans_layer(nodes)
        attn_adj = attn_layers(nodes)
        new_adj = F.softmax(attn_adj.T * adj, dim=1)
        update_feat = torch.mm(new_adj, trans_feat)
        return update_feat
    
    def aggregateSelf(self, nodes, adj, attn_layers):
        attn_adj = attn_layers(nodes)
        attn_adj = F.softmax(attn_adj.T * adj, dim=1)
        update_feat = torch.mm(attn_adj, nodes)
        return update_feat

    def forward(self, img, ques):
        ques_mask = self.make_mask(ques)
        q_emb, _ = self.q_emb(ques) # [batch, q_len, q_dim]
        # Attention
        v_emb = self.img_trans(self.img_enc(img)).unsqueeze(1).repeat(1, 2, 1)
        img_mask = self.make_mask(v_emb)
        feat = self.vqFus(v_emb, q_emb, img_mask, ques_mask)
        
        disease_embeddings = self.diseases_trans(self.disease_embeddings)
        rel_embeddings = self.rels_trans(self.rel_embeddings)
        attributes_embeddings = self.attributes_trans(self.attributes_embeddings)
        disease_emb_list = [disease_embeddings]
        rel_emb_list = [rel_embeddings]
        attr_emb_list = [attributes_embeddings]
        for i in range(self.config.hgat_layers):
            disease_emb_list.append(disease_emb_list[i] + self.aggregate(attr_emb_list[i], self.d2a, self.d2a_trans[i], self.d2a_attn[i]))
            rel_emb_list.append(rel_emb_list[i] + self.aggregate(attr_emb_list[i], self.e2a, self.e2a_trans[i], self.e2a_attn[i]))
            attr_emb_list.append(attr_emb_list[i] + self.aggregate(attr_emb_list[i], self.a2a, self.a2a_trans[i], self.a2a_attn[i]))
 
        disease_emb = disease_emb_list[0] + disease_emb_list[-1]
        rel_emb = rel_emb_list[0] + rel_emb_list[-1]
        disease_emb = disease_emb.unsqueeze(0).repeat(v_emb.shape[0], 1, 1)
        rel_emb = rel_emb.unsqueeze(0).repeat(v_emb.shape[0], 1, 1)
        disease_mask = self.make_mask(disease_emb)
        rel_mask = self.make_mask(rel_emb)
        disease = self.vnFus(v_emb, disease_emb, img_mask, disease_mask)
        rel = self.qeFus(q_emb, rel_emb, ques_mask, rel_mask)

        out = self.classifier(self.norm(torch.cat([feat, disease, rel], dim=1)))
        return out
    
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
