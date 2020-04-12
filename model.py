import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import load_glove_embeddings, load_elmo_embeddings


class MTLArchitecture(nn.Module):
    """

    """

    def __init__(self, num_word_types, shared_layer_size, num_char_types, \
                        char_dim, hidden_dim, dropout, num_layers_shared, num_layers_ner, num_layers_re, \
                        num_tag_types, num_rel_types, activation_type="relu", recurrent_unit="gru"):
        """
        Initialise.

        :param num_word_types: vocabulary size, to be used as input
        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_char_types: vocabulary of characters, to be used for CharRNN
        :param char_dim: character dimension
        :param hidden_dim: hidden dimensions of biRNN
        :param dropout: dropout values for nodes in biRNN
        :param num_layers_shared: number of shared biRNN layers
        :param num_layers_ner: number of NER biRNN layers
        :param num_layers_re: number of RE biRNN layers
        :param num_tag_types: unique tags of the model, will be used by NER specific layers
        :param num_rel_types: unique relations of the model, will be used by RE specific layers
        :param activation_type: the type of activation function to use
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
        """

        super(MTLArchitecture, self).__init__()

        self.shared_layers = SharedRNN(num_word_types, shared_layer_size, num_char_types, \
                                        char_dim, hidden_dim, dropout, num_layers_shared, recurrent_unit)

        self.ner_layers = NERSpecificRNN(shared_layer_size, num_tag_types, hidden_dim, dropout, \
                                        num_layers_ner, activation_type, recurrent_unit)

        self.re_layers = RESpecificRNN(shared_layer_size, num_rel_types, hidden_dim, dropout, \
                                        num_layers_re, activation_type, recurrent_unit)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents):
        shared_representations = self.shared_layers(C, C_lengths, sents)
        print("Shared representations: ", shared_representations.shape)
        ner_score = self.ner_layers(shared_representations, Y)
        # print(ner_score.shape)

    def do_epoch(self, epoch_num, train_batches, optim, check_interval=200):
        self.train()

        output = {}
        for batch_num, (X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents) in enumerate(train_batches):
            optim.zero_grad()
            forward_result = self.forward(X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents)

class SharedRNN(nn.Module):
    """
    A shared Bidirectional GRU layer that takes in the initial words, converts them to respective embeddings
    and passes them to the respective NER and RE specific layers.
    """

    ELMODim = 1024
    GloveDim = 300
    def __init__(self, num_word_types, shared_layer_size, num_char_types, \
                    char_dim, hidden_dim, dropout, num_layers, recurrent_unit="gru"):
        """

        :param num_word_types:
        :param shared_layer_size:
        :param num_char_types:
        :param char_dim:
        :param hidden_dim:
        :param dropout:
        :param num_layers:
        :param recurrent_unit:
        """

        super(SharedRNN, self).__init__()
        self.CharDim = char_dim
        self.Pad_ind = 0
        word_dim = self.ELMODim + self.GloveDim + self.CharDim
        # self.wemb = nn.Embedding(num_word_types, word_dim, padding_idx=self.Pad_ind)

        # Initialise char-embedding BiRNN
        self.cemb = nn.Embedding(num_char_types, self.CharDim, padding_idx=self.Pad_ind)
        self.charRNN = CharRNN(self.cemb, 1, recurrent_unit)

        if recurrent_unit == "gru":
            self.wordRNN = nn.GRU(word_dim, shared_layer_size, num_layers, bidirectional=True)
        else:
            self.wordRNN = nn.LSTM(word_dim, shared_layer_size, num_layers, bidirectional=True)

    def forward(self, char_encoded, C_lengths, raw_sentences):
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
        char_embeddings = char_embeddings.view(batch_size, num_words // batch_size, char_dim)
        final_embeddings = torch.cat([elmo_embeddings, glove_embeddings, char_embeddings], dim=2)

        shared_output, _ = self.wordRNN(final_embeddings)
        return shared_output

class NERSpecificRNN(nn.Module):
    """

    """

    def __init__(self, shared_layer_size, num_tag_types, hidden_dim, dropout, num_layers, \
                    activation_type="relu", recurrent_unit="gru"):
        """
        Initialise.

        :param shared_layer_size:
        :param num_tag_types:
        :param hidden_dim:
        :param dropout:
        :param num_layers:
        :param activation_type:
        :param recurrent_unit:
        """

        super(NERSpecificRNN, self).__init__()

        if recurrent_unit == "gru":
            self.birnn = nn.GRU(2*shared_layer_size, shared_layer_size, num_layers, bidirectional=True)
        else:
            self.birnn = nn.LSTM(2*shared_layer_size, shared_layer_size, num_layers, bidirectional=True)

        self.FFNNe1 = nn.Linear(2*shared_layer_size, hidden_dim)
        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "tanh":
            self.activation = nn.Tanh()
        elif activation_type == "gelu":
            self.activation = nn.GELU()

        self.FFNNe2 = nn.Linear(hidden_dim, num_tag_types)

    def forward(self, shared_representations, Y):
        ner_representation, _ = self.birnn(shared_representations)
        scores = self.FFNNe2(self.activation(self.FFNNe1(ner_representation)))

        print("Score shape: ", scores.shape, "Y shape: ", Y.shape)

class RESpecificRNN(nn.Module):
    """

    """

    def __init__(self, shared_layer_size, num_rel_types, hidden_dim, dropout, num_layers, \
                    activation_type="relu", recurrent_unit="gru"):
        """

        :param shared_layer_size:
        :param num_rel_types:
        :param hidden_dim:
        :param dropout:
        :param num_layers:
        :param activation_type:
        :param recurrent_unit:
        """

        super(RESpecificRNN, self).__init__()

        if activation_type == "relu":
            self.FFNNr1 = nn.ReLU()
        elif activation_type == "tanh":
            self.FFNNr1 = nn.Tanh()
        elif activation_type == "gelu":
            self.FFNNr1 = nn.GELU()

        self.FFNNr2 = nn.Linear(hidden_dim, num_rel_types)

class CharRNN(nn.Module):
    """
    Trains character level embeddings via Bidirectional LSTM.
    """

    def __init__(self, cemb, num_layers=1, recurrent_unit="gru"):
        """

        :param cemb:
        :param num_layers:
        :param recurrent_unit:
        """

        super(CharRNN, self).__init__()
        self.cemb = cemb
        self.num_layers = num_layers
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