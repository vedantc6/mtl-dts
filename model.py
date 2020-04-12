import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import load_glove_embeddings, load_elmo_embeddings


class MTLArchitecture(nn.Module):
    """

    """

    def __init__(self,
                 num_word_types,
                 shared_layer_size,
                 num_char_types,
                 word_dim,
                 char_dim,
                 hidden_dim,
                 dropout,
                 num_layers,
                 num_tag_types,
                 num_rel_types,
                 activation_type="relu",
                 recurrent_unit="gru"):
        """
        Initialise.

        :param num_word_types: vocabulary size, to be used as input
        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_char_types: vocabulary of characters, to be used for CharRNN
        :param word_dim: word dimension
        :param char_dim: character dimension
        :param hidden_dim: hidden dimensions of biRNN
        :param dropout: dropout values for nodes in biRNN
        :param num_layers: number of biRNN layers
        :param num_tag_types: unique tags of the model, will be used by NER specific layers
        :param num_rel_types: unique relations of the model, will be used by RE specific layers
        :param recurrent_unit: GRU/LSTM
        """
        super(MTLArchitecture, self).__init__()

        self.shared_layers = SharedRNN(num_word_types, shared_layer_size, num_char_types, \
                                        word_dim, char_dim, hidden_dim, dropout, num_layers, \
                                        recurrent_unit)

        self.ner_layers = NERSpecificRNN(shared_layer_size, num_tag_types, hidden_dim, dropout, \
                                        num_layers, activation_type, recurrent_unit)

        self.re_layers = RESpecificRNN(shared_layer_size, num_rel_types, hidden_dim, dropout, \
                                        num_layers, activation_type, recurrent_unit)

        self.loss = nn.CrossEntropyLoss()

class CharRNN(nn.Module):
    """
    Trains character level embeddings via Bidirectional LSTM.
    """
    def __init__(self, cemb, num_layers=1, recurrent_unit="gru"):
        super(CharRNN, self).__init__()
        self.cemb = cemb
        if recurrent_unit == "gru":
            self.birnn = nn.GRU(cemb.embedding_dim, cemb.embedding_dim, num_layers, bidirectional=True)
        else:
            self.birnn = nn.LSTM(cemb.embedding_dim, cemb.embedding_dim, num_layers, bidirectional=True)

    def forward(self, padded_chars, char_lengths):
        """

        :param padded_chars:
        :param char_lengths:
        :return:
        """

        B = len(char_lengths)

        packed = pack_padded_sequence(self.cemb(padded_chars), char_lengths,
                                      batch_first=True, enforce_sorted=False)
        _, (final_h, _) = self.birnn(packed)
        return final_h

class SharedRNN(nn.Module):
    """
    A shared Bidirectional GRU layer that takes in the initial words, converts them to respective embeddings
    and passes them to the respective NER and RE specific layers.
    """

    CharDim = 32
    ELMODim = 1024
    GloveDim = 300
    def __init__(self,
                 num_word_types,
                 shared_layer_size,
                 num_char_types,
                 word_dim,
                 char_dim,
                 hidden_dim,
                 dropout=0.5,
                 num_layers=1,
                 recurrent_unit="gru"):
        super(SharedRNN, self).__init__()
        self.Pad_ind = 0
        word_dim = self.ELMODim + self.GloveDim + 2 * self.CharDim
        self.wemb = nn.Embedding(num_word_types, word_dim, padding_idx=self.Pad_ind)

        # Initialise char-embedding BiRNN
        self.cemb = nn.Embedding(num_char_types, self.CharDim, padding_idx=self.Pad_ind)
        self.charRNN = CharRNN(self.cemb, 1, recurrent_unit)

        if recurrent_unit == "gru":
            self.wordRNN = nn.GRU(word_dim, word_dim, num_layers, bidirectional=True)
        else:
            self.wordRNN = nn.LSTM(word_dim, word_dim, num_layers, bidirectional=True)

    def forward(self, raw_sentences, char_encoded, C_lengths):
        """
        Pass the input sentences through the GRU layers.

        :param X: batch of sentences
        :return:
        """

        batch_size = len(raw_sentences)
        elmo_embeddings = load_elmo_embeddings(raw_sentences)
        glove_embeddings = load_glove_embeddings(raw_sentences)
        char_embeddings = self.charRNN(char_encoded, C_lengths)
        num_words, char_dim = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size, num_words, char_dim)
        final_embeddings = torch.cat([elmo_embeddings, glove_embeddings, char_embeddings], dim=2)
        return final_embeddings

class NERSpecificRNN(nn.Module):
    def __init__(self, shared_layer_size, num_tag_types, hidden_dim, dropout, num_layers, \
                    activation_type="relu", recurrent_unit="gru"):
        super(NERSpecificRNN, self).__init__()

        if recurrent_unit == "gru":
            self.birnn = nn.GRU(shared_layer_size, hidden_dim, num_layers, bidirectional=True)
        else:
            self.birnn = nn.LSTM(shared_layer_size, hidden_dim, num_layers, bidirectional=True)

        if activation_type == "relu":
            self.FFNNe1 = nn.ReLU()
        elif activation_type == "tanh":
            self.FFNNe1 = nn.Tanh()
        elif activation_type == "gelu":
            self.FFNNe1 = nn.GELU()

        self.FFNNe2 = nn.Linear(hidden_dim, num_tag_types)

class RESpecificRNN(nn.Module):
    def __init__(self, shared_layer_size, num_rel_types, hidden_dim, dropout, num_layers, \
                    activation_type="relu", recurrent_unit="gru"):
        super(RESpecificRNN, self).__init__()

        input_ = None
        if activation_type == "relu":
            self.FFNNr1 = nn.ReLU()(input_)
        elif activation_type == "tanh":
            self.FFNNr1 = nn.Tanh()(input_)
        elif activation_type == "gelu":
            self.FFNNr1 = nn.GELU()(input_)

        self.FFNNr2 = nn.Linear(self.FFNNr1, num_rel_types)
