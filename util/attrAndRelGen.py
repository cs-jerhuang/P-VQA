from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch
import json

def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").cuda()
    attrGen(bert, tokenizer)
    relGen(bert, tokenizer)

def attrGen(bert, tokenizer):
    with open("data/knowledge_graph/graph_nodes.json", "rt", encoding="utf8") as f:
        data = json.load(f)

    result_list = []
    bert.eval()
    with torch.no_grad():
        for index, e in enumerate(data["nodes"]):
            line = e["name"].replace("-", " ").replace("_", " ") + " " + e["description"].replace("-", " ").replace("_", " ")
            tokens = tokenizer.tokenize(line.lower())
            ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).cuda()
            encoded_layers, res = bert(ids, output_all_encoded_layers=True)
            text_embedding = res.cpu().numpy()
            result_list.append(text_embedding)
    result_list = np.concatenate(result_list, axis=0)
    print(result_list.shape)
    np.save("features/graph_init_features/attributes.npy", result_list)


def relGen(bert, tokenizer):
    with open("data/knowledge_graph/graph_edge_type.json", "rt", encoding="utf8") as f:
        data = json.load(f)
    result_list = []
    bert.eval()
    with torch.no_grad():
        for index, e in enumerate(data["edge_type"]):
            line = e["name"].replace("-", " ").replace("_", " ")
            tokens = tokenizer.tokenize(line.lower())
            ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).cuda()
            encoded_layers, res = bert(ids, output_all_encoded_layers=True)
            text_embedding = res.cpu().numpy()
            result_list.append(text_embedding)
    result_list = np.concatenate(result_list, axis=0)
    print(result_list.shape)
    np.save("features/graph_init_features/rels.npy", result_list)


if __name__ == "__main__":
    main()
