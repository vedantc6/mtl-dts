import json
from collections import Counter
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from torchtext.vocab import Vectors
import os
import torch


conll_entities = set()
conll_relations = set()

def load_vertical_tagged_data(path, sort_by_length=True):
    """

    :param path:
    :param sort_by_length:
    :return:
    """

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

def load_elmo_weights(sentences, num_output_representations=1, dropout=0, mode="single"):
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
    glove_vectors = Vectors('glove.6B.300d.txt', './data/embeddings/')

    # Convert the input sentences to embeddings.
    final_sentences = []
    for sentence in sentences:
        sentence_with_embeddings = glove_vectors.get_vecs_by_tokens(sentence)
        final_sentences.append(sentence_with_embeddings)
    return final_sentences



# FOR TESTING
if __name__ == "__main__":
    embeds = load_glove_embeddings(sentences=[["I", "ate", "an", "apple", "for", "breakfast"],
                                          ["I", "ate", "an", "orange", "for", "dinner"]])