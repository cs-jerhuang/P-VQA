from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch
import json
import tqdm


'''
Embed the questions into BERT embeddings.
'''


def padNdarry(embedding):
    axis1 = embedding.shape[1]
    if embedding.shape[1] > 20:
        axis1 = 20

    newOne = np.zeros((1, 20, 768), np.float32)
    newOne[:, :axis1, :] = embedding[:, :axis1, :]
    return newOne


def main(dataset):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").cuda()
    with open("data/%s.json" % dataset, "rt", encoding="utf8") as f:
        data = json.load(f)

    print("Building %s set..., num of data is %s" % (dataset, len(data)))
    qEmbedding = []
    bert.eval()
    with torch.no_grad():
        for index, e in tqdm.tqdm(enumerate(data)):
            tokens = tokenizer.tokenize(e["question"].lower())
            ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).cuda()
            encoded_layers, res = bert(ids, output_all_encoded_layers=True)
            text_embedding = encoded_layers[-1].cpu().numpy()
            qEmbedding.append(padNdarry(text_embedding))
    qEmbedding = np.concatenate(qEmbedding, axis=0)
    print(qEmbedding.shape)
    np.save("features/question_features/%s_sentence_text.npy" % dataset, qEmbedding)


if __name__ == "__main__":
    main("train")
    main("val")
    main("test")
