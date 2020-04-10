import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import load_glove_embeddings, load_elmo_weights
from read_data import Dataset


class CharBiRNN(nn.Module):
    """
    Trains character level embeddings via Bidirectional LSTM.
    """

    def __init__(self, cemb, num_layers=1, unit="gru"):
        """

        :param cemb:
        :param num_layers:
        :param unit:
        """

        super(CharBiRNN, self).__init__()
        self.cemb = cemb
        if unit == "gru":
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

        final_h = final_h.view(self.bilstm.num_layers, 2, B,
                               self.bilstm.hidden_size)[-1]       # 2 x BT x d_c
        cembs = final_h.transpose(0, 1).contiguous().view(B, -1)  # BT x 2d_c
        return cembs

class SharedBiRNN(nn.Module):
    """
    A shared Bidirectional GRU layer that takes in the initial words, converts them to respective embeddings
    and passes them to the respective NER and RE specific layers.
    """

    CharDim = 32
    ELMODim = 1024
    GloveDim = 300
    def __init__(self,
                 num_word_types,
                 num_char_types,
                 num_layers=1,
                 recurrent_unit="gru"):
        super(SharedBiRNN, self).__init__()

        self.Pad_ind = 0
        word_dim = self.ELMODim + self.GloveDim + 2 * self.CharDim
        self.wemb = nn.Embedding(num_word_types, word_dim, padding_idx=self.Pad_ind)

        # Initialise char-embedding BiRNN
        self.cemb = nn.Embedding(num_char_types, self.CharDim, padding_idx=self.Pad_ind)
        self.charRNN = CharBiRNN(self.cemb, 1, recurrent_unit)

        # BiRNN
        if recurrent_unit == "gru":
            self.word_birnn = nn.GRU(input_size=word_dim,
                                     hidden_size=word_dim,
                                     num_layers=num_layers,
                                     bidirectional=True)
        else:
            self.word_birnn = nn.LSTM(input_size=word_dim,
                                   hidden_size=word_dim,
                                   num_layers=num_layers,
                                   bidirectional=True)

    def forward(self, X):
        """
        Pass the input sentences through the GRU layers.

        :param X: batch of sentences
        :return:
        """

        elmo_embeddings = load_elmo_weights(X)
        glove_embeddings = load_glove_embeddings(X)
        word_embeddings = torch.cat([elmo_embeddings, glove_embeddings], dim=2)


dataset_loader = Dataset()
shared_rnn = SharedBiRNN(num_word_types=len(dataset_loader.word2x),
                         num_char_types=len(dataset_loader.char2c))
shared_rnn.forward(dataset_loader.wordseqs_train[:2])

class NERSpecificBiRNN(nn.Module):
    """

    """

    def __init__(self,
                 num_rel_types,
                 num_tag_types,
                 hidden_dim,
                 dropout,
                 num_layers,
                 activation_type="relu",
                 recurrent_unit="gru"):
        super(NERSpecificBiRNN, self).__init__()

        input_ = None
        if activation_type == "relu":
            self.FFNNe1 = nn.ReLU()(input_)
        elif activation_type == "tanh":
            self.FFNNe1 = nn.Tanh()(input_)
        elif activation_type == "gelu":
            self.FFNNe1 = nn.GELU()(input_)

        self.FFNNe2 = nn.Linear(self.FFNNe1, num_tag_types)

class RESpecificBiRNN(nn.Module):
    """

    """

    def __init__(self,
                 num_rel_types,
                 hidden_dim,
                 dropout,
                 num_layers,
                 activation_type="relu",
                 recurrent_unit="gru"):
        super(RESpecificBiRNN, self).__init__()

        input_ = None
        if activation_type == "relu":
            self.FFNNr1 = nn.ReLU()(input_)
        elif activation_type == "tanh":
            self.FFNNr1 = nn.Tanh()(input_)
        elif activation_type == "gelu":
            self.FFNNr1 = nn.GELU()(input_)

        self.FFNNr2 = nn.Linear(self.FFNNr1, num_rel_types)

