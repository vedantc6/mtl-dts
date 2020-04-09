import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from .utils import load_glove_embeddings


class SharedBiRNN(nn.Module):
    """
    A shared Bidirectional GRU layer that takes in the initial words, converts them to respective embeddings
    and passes them to the respective NER and RE specific layers.
    """

    CharDim = 32
    ELMODim = 100
    GloveDim = 300
    def __init__(self,
                 num_word_types,
                 num_tag_types,
                 num_char_types,
                 dropout_rate=0.5,
                 num_layers=1,
                 recurrent_unit="gru"):
        super(SharedBiRNN, self).__init__()

        # Initialise char-embedding BiRNN
        self.cemb = nn.Embedding(num_char_types, self.CharDim)
        self.charRNN = CharBiRNN(self.cemb, 1, recurrent_unit)

        # BiRNN
        word_dim = self.ELMODim + self.GloveDim + 2 * self.CharDim
        self.birnn = nn.GRU(input_size=word_dim,
                            hidden_size=word_dim,
                            num_layers=num_layers,
                            bidirectional=True)

    def forward(self, X):
        """

        :param X:
        :return:
        """

        return

class NERSpecificBiRNN(nn.Module):
    """

    """

    def __init__(self,
                 num_word_types,
                 num_tag_types,
                 num_char_types,
                 word_dim,
                 char_dim,
                 hidden_dim,
                 dropout,
                 num_layers,
                 recurrent_unit="gru"):
        super(NERSpecificBiRNN, self).__init__()
        pass

class RESpecificBiRNN(nn.Module):
    """

    """

    def __init__(self,
                 num_word_types,
                 num_tag_types,
                 num_char_types,
                 word_dim,
                 char_dim,
                 hidden_dim,
                 dropout,
                 num_layers,
                 recurrent_unit="gru"):
        super(RESpecificBiRNN, self).__init__()
        pass

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