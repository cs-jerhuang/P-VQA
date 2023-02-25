import numpy as np
import json


def attr2attr():
    attri = np.load("features/graph_init_features/attributes.npy")
    print("adj of attr2attr shape should be:", attri.shape[0], attri.shape[0])
    adj = np.zeros((attri.shape[0], attri.shape[0]), dtype=np.int64)

    with open("data/knowledge_graph/triple_attributes2attributes.json", "rt", encoding="utf8") as f:
        data = json.load(f)
    for e in data:
        for i in e["nodes"]:
            for j in e["nodes"]:
                adj[i, j] = 1.0
    np.save("features/graph_init_features/a2a.npy", adj)


def rel2attr():
    rel = np.load("features/graph_init_features/rels.npy")
    attri = np.load("features/graph_init_features/attributes.npy")
    print("adj of rel2attr shape should be:", rel.shape[0], attri.shape[0])
    adj = np.zeros((rel.shape[0], attri.shape[0]), dtype=np.int64)

    with open("data/knowledge_graph/triple_attributes2attributes.json", "rt", encoding="utf8") as f:
        data = json.load(f)
    for e in data:
        for j in e["nodes"]:
            adj[e["relation_id"], j] = 1.0
    
    with open("data/knowledge_graph/triple_disease2attributes.json", "rt", encoding="utf8") as f:
        data = json.load(f)
    for e in data["tri-list"]:
        for j in e["tail_id"]:
            adj[e["relation_id"], j] = 1.0
    
    np.save("features/graph_init_features/e2a.npy", adj)


def disease2attr():
    nodes = np.load("features/graph_init_features/diseases.npy")
    attri = np.load("features/graph_init_features/attributes.npy")
    print("adj of disease2attr shape should be:", nodes.shape[0], attri.shape[0])
    adj = np.zeros((nodes.shape[0], attri.shape[0]), dtype=np.int64)
    with open("data/knowledge_graph/triple_disease2attributes.json", "rt", encoding="utf8") as f:
        data = json.load(f)
    for e in data["tri-list"]:
        for t in e["tail_id"]:
            adj[e["head_id"], t] = 1.0
    np.save("features/graph_init_features/d2a.npy", adj)


def disease2disease():
    nodes = np.load("features/graph_init_features/diseases.npy")
    print("adj of disease2disease shape should be:", nodes.shape[0], nodes.shape[0])
    adj = np.zeros((nodes.shape[0], nodes.shape[0]), dtype=np.int64)
    with open("data/knowledge_graph/triple_disease2disease.json", "rt", encoding="utf8") as f:
        data = json.load(f)
    for e in data:
        for i in e["nodes"]:
            for j in e["nodes"]:
                adj[i-1, j-1] = 1.0
    np.save("features/graph_init_features/d2d.npy", adj)


def main():
    attr2attr()
    rel2attr()
    disease2attr()
    disease2disease()


if __name__ == "__main__":
    main()
