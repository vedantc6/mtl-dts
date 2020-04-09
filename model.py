import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class SharedRNN(nn.Module):
    def __init__(self, num_word_types, num_tag_types, num_char_types, word_dim, char_dim, hidden_dim, dropout, num_layers, recurrent_unit="gru"):
        self.PAD_ind = 0
        self.wemb = nn.Embedding(num_word_types, word_dim, padding_idx=self.PAD_ind)
        self.cemb = nn.Embedding(num_char_types, char_dim, padding_idx=self.PAD_ind)

        input_dim = word_dim + 2*char_dim
        self.charRNN = CharRNN(self.cemb, 1, recurrent_unit)

        if recurrent_unit == "gru":
            self.wordRNN = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=True)
        else:
            self.wordRNN = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True)

class NERSpecificRNN(nn.Module):
    def __init__(self, num_tag_types, hidden_dim, dropout, num_layers, activation_type="relu", recurrent_unit="gru"):
        input_ = None
        if activation_type == "relu":
            self.FFNNe1 = nn.ReLU()(input_)
        elif activation_type == "tanh":
            self.FFNNe1 = nn.Tanh()(input_) 
        elif activation_type == "gelu":
            self.FFNNe1 = nn.GELU()(input_)
        
        self.FFNNe2 = nn.Linear(self.FFNNe1, num_tag_types)

class RESpecificRNN(nn.Module):
    def __init__(self, num_rel_types, hidden_dim, dropout, num_layers, activation_type="relu", recurrent_unit="gru"):
        input_ = None
        if activation_type == "relu":
            self.FFNNr1 = nn.ReLU()(input_)
        elif activation_type == "tanh":
            self.FFNNr1 = nn.Tanh()(input_) 
        elif activation_type == "gelu":
            self.FFNNr1 = nn.GELU()(input_)
        
        self.FFNNr2 = nn.Linear(self.FFNNr1, num_rel_types)

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
        _, (final_h, _) = self.birnn(packed)

        final_h = final_h.view(self.birnn.num_layers, 2, B, self.birnn.hidden_size)[-1]       # 2 x BT x d_c
        cembs = final_h.transpose(0, 1).contiguous().view(B, -1)  # BT x 2d_c
        return cembs