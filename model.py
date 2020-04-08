import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class SharedRNN(nn.Module):
    def __init__(self, num_word_types, num_tag_types, num_char_types, word_dim, char_dim, hidden_dim, dropout, num_layers, recurrent_unit="gru"):
        self.cemb = nn.Embedding(num_char_types, char_dim)
        self.charRNN = CharRNN(self.cemb, 1, recurrent_unit)

class NERSpecificRNN(nn.Module):
    def __init__(self, num_word_types, num_tag_types, num_char_types, word_dim, char_dim, hidden_dim, dropout, num_layers, recurrent_unit="gru"):
        pass

class RESpecificRNN(nn.Module):
    def __init__(self, num_word_types, num_tag_types, num_char_types, word_dim, char_dim, hidden_dim, dropout, num_layers, recurrent_unit="gru"):
        pass

class CharRNN(nn.Module):
    def __init__(self, cemb, num_layers=1, unit="gru"):
        super(CharRNN, self).__init__()
        self.cemb = cemb
        if unit == "gru":
            self.birnn = nn.GRU(cemb.embedding_dim, cemb.embedding_dim, num_layers, bidirectional=True)
        else:
            self.birnn = nn.LSTM(cemb.embedding_dim, cemb.embedding_dim, num_layers, bidirectional=True)

    def forward(self, padded_chars, char_lengths):
        B = len(char_lengths)

        packed = pack_padded_sequence(self.cemb(padded_chars), char_lengths,
                                      batch_first=True, enforce_sorted=False)
        _, (final_h, _) = self.bilstm(packed)

        final_h = final_h.view(self.bilstm.num_layers, 2, B,
                               self.bilstm.hidden_size)[-1]       # 2 x BT x d_c
        cembs = final_h.transpose(0, 1).contiguous().view(B, -1)  # BT x 2d_c
        return cembs