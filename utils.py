import json
from collections import Counter
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from torchtext.vocab import Vectors
from torchtext.data import Field
import os
import torch
import spacy
import torch.nn as nn


conll_entities = set()
conll_relations = set()

def get_init_weights(init_value):
    def init_weights(m):
        if init_value > 0.0:
            if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_'):
                nn.init.uniform_(m.weight, a=-init_value, b=init_value)
            if hasattr(m, 'bias') and hasattr(m.bias, 'uniform_'):
                nn.init.uniform_(m.bias, a=-init_value, b=init_value)

    return init_weights

def tokenizer(text):
    """

    :param text:
    :return:
    """

    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_vertical_tagged_data(path, sort_by_length=True):
    """

    :param path:
    :param sort_by_length:
    :return:
    """

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
                tmp_rel.append((tagseq[start][1] - 1, tagseq[end][1] - 1, rel_type))

        wordseqs.append(datapoint["tokens"])
        charseqslist.append([char for words in datapoint["tokens"] for char in words])
        tagseqs.append(tmp_seq)
        relseqs.append(tmp_rel)

    for sent, tags, rels, charslist in zip(wordseqs, tagseqs, relseqs, charseqslist):
        for word, tag, rel in zip(sent, tags, rels):
            wordcounter[word] += 1
            tagcounter[tag] += 1
            relcounter[rel[2]] += 1
        for char in charslist:
            charcounter[char] += 1

    if sort_by_length:
        wordseqs, tagseqs, relseqs, charseqslist = (list(t) for t in zip(*sorted(zip(wordseqs, tagseqs, relseqs, charseqslist), \
                                                    key=lambda x: len(x[0]), reverse=True)))
    assert len(wordseqs) == len(data), "Make sure the data is loading properly and is not lost"

    return wordseqs, tagseqs, relseqs, charseqslist, wordcounter, tagcounter, relcounter, charcounter

def load_elmo_embeddings(sentences, num_output_representations=1, dropout=0, mode="single"):
    """
    Converts each word of the sentences to their respective ELMO embeddings.

    :param sentences:
    :param num_output_representations:
    :param dropout:
    :param mode:
    :return:
    """
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    if os.path.exists("pretrained_weights/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"):
        weight_file = "pretrained_weights/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    else:
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    elmo = Elmo(options_file, weight_file, num_output_representations=num_output_representations, dropout=dropout)

    # Converts a batch of tokenized sentences to a tensor representing the sentences with encoded
    # characters (len(batch), max sentence length, max word length).
    character_ids = batch_to_ids(sentences)

    elmo_embedding = elmo(character_ids)
    if mode == "single":
        # use the last layer of elmo embeddings. Size: batch_size, timesteps, embedding_dim
        return elmo_embedding['elmo_representations'][-1]
    else:
        batch_size, timesteps, embed_dim = elmo_embedding['elmo_representations'][-1].shape()
        emb_list = [vect for vect in elmo_embedding['elmo_representations']]
        embeddings = torch.cat(emb_list, 2).view(batch_size, -1, embed_dim, num_output_representations)

        # concatenate different output representations of elmo embeddings
        if mode == "concat_layers":
            return embeddings

        # weighted sum of output representations
        else:
            vars = torch.Tensor(num_output_representations, 1).cuda()
            embeddings = torch.matmul(embeddings, vars).view(batch_size, -1, embed_dim)
            return embeddings

def load_glove_embeddings(sentences):
    """
    Converts each word of the sentences to the respective Glove embeddings.

    :param sentences
    return:
    """

    # Load the glove vectors saved locally.
    glove_vectors = Vectors('glove.6B.300d.txt', './pretrained_weights/')

    # Convert the input sentences to embeddings.
    final_sentences = []
    batch_size = len(sentences)
    max_len = max([len(sentence) for sentence in sentences])
    for sentence in sentences:
        sentence_with_embeddings = glove_vectors.get_vecs_by_tokens(sentence)

        # Add padding for words.
        if len(sentence_with_embeddings) < max_len:
            temp = torch.zeros([max_len - len(sentence), 300]).float()
            sentence_with_embeddings = torch.cat([sentence_with_embeddings, temp], dim=0)

        final_sentences.append(torch.as_tensor(sentence_with_embeddings))
    return torch.stack(final_sentences).view(batch_size, max_len, 300)

def load_onehot_embeddings(sentences):
    pass


# FOR TESTING
if __name__ == "__main__":
    embeds = load_elmo_embeddings(sentences = [["I", "ate", "an", "apple", "for", "breakfast"],["I", "ate", "an", "orange", "for", "dinner"]])