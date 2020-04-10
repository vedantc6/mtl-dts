import json
from collections import Counter
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import os 
import torch

conll_entities = set()
conll_relations = set()

def load_vertical_tagged_data(path, sort_by_length=True):
    wordseqs = []
    tagseqs = []
    relseqs = []
    charseqslist = []
    wordcounter = Counter()
    tagcounter = Counter()
    relcounter = Counter()
    charcounter = Counter()

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
        charseqslist.append([char for words in datapoint["tokens"] for char in words])
        tagseqs.append(tmp_seq)
        relseqs.append(tmp_rel)
    
    for sent, tags, rels, charslist in zip(wordseqs, tagseqs, relseqs, charseqslist):
        for word, tag, rel, chars in zip(sent, tags, rels, charslist):
            wordcounter[word] += 1
            tagcounter[tag] += 1
            relcounter[rel[2]] += 1
            for char in chars:
                charcounter[char] += 1

    if sort_by_length:
        wordseqs, tagseqs, relseqs, charseqslist = (list(t) for t in zip(*sorted(zip(wordseqs, tagseqs, relseqs, charseqslist), \
                                                    key=lambda x: len(x[0]), reverse=True)))
    assert len(wordseqs) == len(data), "Make sure the data is loading properly and is not lost"

    return wordseqs, tagseqs, relseqs, charseqslist, wordcounter, tagcounter, relcounter, charcounter

def load_elmo_embeddings(sentences, num_output_representations=1, dropout=0, mode="single"):
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    if os.path.exists("pretrained_weights/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"):
        weight_file = "pretrained_weights/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    else:
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    
    elmo = Elmo(options_file, weight_file, num_output_representations=num_output_representations, dropout=dropout)

    #Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters (len(batch), max sentence length, max word length).
    character_ids = batch_to_ids(sentences)

    elmo_embedding = elmo(character_ids)
    if mode == "single":
        # use the last layer of elmo embeddings. Size: batch_size, timesteps, embedding_dim
        return elmo_embedding['elmo_representations'][-1]
    else:
        batch_size, timesteps, embed_dim = elmo_embedding['elmo_representations'][-1].shape()
        emb_list = [vect for vect in elmo_embedding['elmo_representations']]
        embs = torch.cat(emb_list, 2).view(batch_size, -1, embed_dim, num_output_representations)
        if mode == "concat_layers":
            # concatenate different output representations of elmo embeddings
            return embs
        else:
            # weighted sum of output representations
            vars = torch.Tensor(num_output_representations, 1).cuda()
            embs = torch.matmul(embs, vars).view(batch_size, -1, embed_dim)
            return embs

def load_glove_embeddings():
    pass

def load_onehot_embeddings(sentences):
    pass


# FOR TESTING
if __name__ == "__main__":
    embeds = load_elmo_embeddings(sentences = [["I", "ate", "an", "apple", "for", "breakfast"],["I", "ate", "an", "orange", "for", "dinner"]])