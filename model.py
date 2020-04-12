import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import load_glove_embeddings, load_elmo_embeddings, load_onehot_embeddings
from crf import CRFLoss
import math


class MTLArchitecture(nn.Module):
    """
    The main class where all successive architectures are initialised and a forward pass is done through each
    of them.
    """

    def __init__(self, num_word_types, shared_layer_size, num_char_types, \
                        char_dim, hidden_dim, dropout, num_layers_shared, num_layers_ner, num_layers_re, \
                        num_tag_types, num_rel_types, init, activation_type="relu", recurrent_unit="gru"):
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
        :param init: uniform initialization used in NER CRF
        :param activation_type: the type of activation function to use
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
        """

        super(MTLArchitecture, self).__init__()

        self.shared_layers = SharedRNN(shared_layer_size, num_char_types, \
                                       char_dim, dropout, num_layers_shared, recurrent_unit)

        self.ner_layers = NERSpecificRNN(shared_layer_size, num_tag_types, hidden_dim, dropout, \
                                        num_layers_ner, init, activation_type, recurrent_unit)

        self.re_layers = RESpecificRNN(shared_layer_size, num_rel_types, hidden_dim, dropout, \
                                        num_layers_re, activation_type, recurrent_unit)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents):
        """
        Do a single forwawrd pass on the entire architecture - through all the shared, NER and RE RNNs.

        :param X: encoded sentences
        :param Y: encoded tags
        :param C: encoded characters
        :param C_lengths: lengths of characters in the words
        :param rstartseqs: the start indices of the relations for RE
        :param rendseqs: the end indices of relations for RE
        :param rseqs:
        :param sents: raw non-encoded sentences
        :return:
        """

        shared_representations = self.shared_layers(C, C_lengths, sents)
        ner_score = self.ner_layers(shared_representations, Y)
        return ner_score

    def do_epoch(self, epoch_num, train_batches, clip, optim, check_interval=200):
        """
        Run the forward pass in multiple epochs across training batches.

        :param epoch_num: number of epochs
        :param train_batches: the training data batches
        :param optim: the optimiser used for minimising the loss
        :param check_interval: save the results once after this many intervals
        :return:
        """

        self.train()

        output = {}
        for batch_num, (X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents) in enumerate(train_batches):
            optim.zero_grad()
            forward_result = self.forward(X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents)
            loss1 = forward_result["loss"]
            loss1.backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)
            optim.step()

            for key in forward_result:
                output[key] = forward_result[key] if not key in output else \
                              output[key] + forward_result[key]

            if (batch_num + 1) % check_interval == 0:
                print('Epoch {:3d} | Batch {:5d}/{:5d} | '
                           'Average Loss {:8.4f}'.format(
                               epoch_num, batch_num + 1, len(train_batches),
                               output['loss'] / (batch_num + 1)))
            if math.isnan(output['loss']):
                print('Stopping training since objective is NaN')
                break
        for key in output:
            output[key] /= (batch_num + 1)

        return output

class SharedRNN(nn.Module):
    """
    A shared Bidirectional GRU layer that takes in the initial words, converts them to respective embeddings
    and passes them to the respective NER and RE specific layers.
    """

    ELMODim = 1024
    GloveDim = 300
    OneHotDim = 7
    def __init__(self, shared_layer_size, num_char_types, \
                    char_dim, dropout, num_layers, recurrent_unit="gru"):
        """
        Initialise.

        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_char_types: vocabulary of characters, to be used for CharRNN
        :param char_dim: character dimension
        :param dropout: amount of dropout to use
        :param num_layers: total number of shared layers
        :param recurrent_unit: the type of recurrent unit to use in the bidirectional RNN - GRU or LSTM
        """

        super(SharedRNN, self).__init__()
        self.CharDim = char_dim
        self.Pad_ind = 0
        word_dim = self.ELMODim + self.GloveDim + self.CharDim + self.OneHotDim

        # Initialise char-embedding BiRNN
        self.cemb = nn.Embedding(num_char_types, self.CharDim, padding_idx=self.Pad_ind)
        self.charRNN = CharRNN(self.cemb, 1, recurrent_unit)

        if recurrent_unit == "gru":
            self.wordRNN = nn.GRU(word_dim, shared_layer_size, num_layers, bidirectional=True)
        else:
            self.wordRNN = nn.LSTM(word_dim, shared_layer_size, num_layers, bidirectional=True)

    def forward(self, char_encoded, C_lengths, raw_sentences):
        """
        Do a forward pass by taking the input sentences, generating the respective embeddings and getting the resulting
        shared representations.

        :param char_encoded: tensor of encoded characters
        :param C_lenghts: lengths of the words (number of characters in each word of the sentence)
        :param raw_sentences: list of raw sentences
        :return: tensor of shared representations from biRNN
        """

        batch_size = len(raw_sentences)

        # Calculate the different embeddings.
        elmo_embeddings = load_elmo_embeddings(raw_sentences)
        glove_embeddings = load_glove_embeddings(raw_sentences)
        char_embeddings = self.charRNN(char_encoded, C_lengths)
        one_hot_embeddings = load_onehot_embeddings(raw_sentences)

        # Concatenate the embeddings.
        num_words, char_dim = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size, num_words // batch_size, char_dim)
        final_embeddings = torch.cat([elmo_embeddings, glove_embeddings, char_embeddings, one_hot_embeddings], dim=2)

        # Get the shared layer representations.
        shared_output, _ = self.wordRNN(final_embeddings)
        return shared_output

class NERSpecificRNN(nn.Module):
    """
    NER specific bidirectional GRU layers that take in the shared representations from the shared layers and calculates
    the respective NER scores.
    """

    def __init__(self, shared_layer_size, num_tag_types, hidden_dim, dropout, num_layers, \
                    init, activation_type="relu", recurrent_unit="gru"):
        """
        Initialise.

        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_tag_types: unique tags of the model, will be used by NER specific layers
        :param hidden_dim: the NER biRNN hidden layer dimension
        :param dropout: dropout values for nodes in biRNN
        :param num_layers: number of layers in this biRNN
        :param activation_type: the type of activation function to use
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
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
        self.loss = CRFLoss(num_tag_types, init)

    def forward(self, shared_representations, Y):
        """
        Do a forward pass by taking input from the shared layers and generating the NER scores for the input
        sentences.

        :param shared_representations: tensor of shared representations from the shared layer.
        :param Y: the label NER tags for the input sentences
        :return: NER scores
        """

        ner_representation, _ = self.birnn(shared_representations)
        scores = self.FFNNe2(self.activation(self.FFNNe1(ner_representation)))
        loss = self.loss(scores, Y)
        return {'loss': loss}

class RESpecificRNN(nn.Module):
    """
    RE specific bidirectional GRU layers that take in the shared representations from the shared layers and calculates
    the respective RE scores.
    """

    def __init__(self, shared_layer_size, num_rel_types, hidden_dim, dropout, num_layers, \
                    activation_type="relu", recurrent_unit="gru"):
        """
        Initialise.

        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_layers_re: number of RE biRNN layers
        :param hidden_dim: the NER biRNN hidden layer dimension
        :param dropout: dropout values for nodes in biRNN
        :param num_layers: number of layers in this biRNN
        :param activation_type: the type of activation function to use
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
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
        Initialise.

        :param cemb: nn.Embedding for the characters
        :param num_layers: number of layers in this biRNN
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
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
        Do a forward pass to learn the character embeddings.

        :param padded_chars: the padded character encodings
        :param char_lengths: lengths of the words
        :return: learned character embeddings in the form of biRNN hidden vector
        """

        B = len(char_lengths)
        packed = pack_padded_sequence(self.cemb(padded_chars), char_lengths,
                                      batch_first=True, enforce_sorted=False)
        _, (final_h, _) = self.birnn(packed)
        return final_h
