import json
from collections import Counter
import numpy as np

conll_entities = set()
conll_relations = set()

def load_vertical_tagged_data(path, sort_by_length=True):
    wordseqs = []
    tagseqs = []
    relseqs = []

    wordcounter = Counter()
    tagcounter = Counter()
    relcounter = Counter()

    data = json.load(open(path))
    for datapoint in data:
        tagseq = []
        relseq = []
        for key, val in datapoint.items():
            if key == "entities":
                for entity in val:
                    tagseq.append((entity["start"], entity["end"], entity["type"]))
                    conll_entities.add(entity['type'])
            if key == "relations":
                for relation in val:
                    relseq.append((relation["head"], relation["tail"], relation["type"]))
                    conll_relations.add(relation['type'])
        # print(tagseq, "##############", relseq)
        if tagseq:
            tmp_seq = np.chararray(shape=(len(datapoint["tokens"], )), itemsize=15)
            tmp_seq[:] = "O"
            for tags in tagseq:
                start, end, ent_type = tags
                tmp_seq[start] = "B-" + ent_type
                if end - start > 1:
                    tmp_seq[start+1:end] = "I-" + ent_type
            tmp_seq = list(np.char.decode(tmp_seq, "utf-8"))
            tmp_rel = []
            for rel in relseq:
                start, end, rel_type = rel
                tmp_rel.append((tagseq[start][1]-1, tagseq[end][1]-1, rel_type))

        wordseqs.append(datapoint["tokens"])
        tagseqs.append(tmp_seq)
        relseqs.append(tmp_rel)
    
    for sent, tags, rels in zip(wordseqs, tagseqs, relseqs):
        for word, tag, rel in zip(sent, tags, rels):
            wordcounter[word] += 1
            tagcounter[tag] += 1
            relcounter[rel[2]] += 1
    
    if sort_by_length:
        wordseqs, tagseqs, relseqs = (list(t) for t in zip(*sorted(zip(wordseqs, tagseqs, relseqs), key=lambda x: len(x[0]), reverse=True)))
    assert len(wordseqs) == len(data), "Make sure the data is loading properly and is not lost"
    return wordseqs, tagseqs, relseqs, wordcounter, tagcounter, relcounter