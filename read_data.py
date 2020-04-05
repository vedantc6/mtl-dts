import json
from collections import Counter
import numpy as np

conll_path = "./data/datasets/conll04/"
ade_path = "./data/datasets/ade/"

conll_data = json.load(open(conll_path + "conll04_dev.json"))
conll_entities = set()
conll_relations = set()
word2idx = {}
idx2word = {}

def create_dataset(path, datatype="conll04"):
    wordseqs = []
    tagseqs = []
    # charseqslist = []
    # charseqs = []
    wordcounter = Counter()
    tagcounter = Counter()
    # charcounter = Counter()
    # if datatype == "conll04":
    data = json.load(open(path))
    for datapoint in data:
        tagseq = []
        for key, val in datapoint.items():
            if key == "entities":
                for entity in val:
                    tagseq.append((entity["start"], entity["end"], entity["type"]))
                    conll_entities.add(entity['type'])
            if key == "relations":
                for relation in val:
                    conll_relations.add(relation['type'])
            if key == "tokens":
                for tokens in val:
                    wordcounter[tokens] += 1
        # if "self-propelled" in datapoint["tokens"]:
        #     print(datapoint["tokens"])

        if tagseq:
            tmp_seq = np.chararray(shape=(len(datapoint["tokens"], )), itemsize=20)
            tmp_seq[:] = "0"
            for tags in tagseq:
                start, end, ent_type = tags
                tmp_seq[start] = "B-" + ent_type
                if end - start > 1:
                    tmp_seq[start+1:end] = "I-" + ent_type
            tmp_seq = np.char.decode(tmp_seq, "utf-8")
        print(datapoint["tokens"])
        wordseqs.append(datapoint["tokens"])
        tagseqs.append(list(tmp_seq))
    
    # for sent, tags in zip(wordseqs, tagseqs):
    #     print(sent, tags)
    assert len(wordseqs) == len(data)   # check for no data is lost
           
if __name__ == "__main__":
    create_dataset(conll_path + "conll04_dev.json")