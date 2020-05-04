import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import load_glove_embeddings, load_elmo_embeddings, load_onehot_embeddings, get_boundaries
from crf import CRFLoss
import math
import itertools
from collections import defaultdict, Counter
import numpy as np
from transformer import TransformerModel

class MTLArchitecture(nn.Module):
    """
    The main class where all successive architectures are initialised and a forward pass is done through each
    of them.
    """

    def __init__(self, num_word_types, shared_layer_size, num_char_types,
                 char_dim, hidden_dim, dropout, re_dropout, num_layers_shared, num_layers_ner, 
                 num_layers_re, num_tag_types, num_rel_types, init, label_embeddings_size, re_ff1_size,
                 re_lambda, e1_activation_type, r1_activation_type, recurrent_unit="gru", device='cuda'):
        """
        Initialise.

        :param num_word_types: vocabulary size, to be used as input
        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_char_types: vocabulary of characters, to be used for CharRNN
        :param char_dim: character dimension
        :param hidden_dim: hidden dimensions of NER FFe1
        :param dropout: dropout values for nodes in biRNN
        :param num_layers_shared: number of shared biRNN layers
        :param num_layers_ner: number of NER biRNN layers
        :param num_layers_re: number of RE biRNN layers
        :param num_tag_types: unique tags of the model, will be used by NER specific layers
        :param num_rel_types: unique relations of the model, will be used by RE specific layers
        :param init: uniform initialization used in NER CRF
        :param label_embeddings_size: label embedding size to be used in NER and RE
        :param activation_type: the type of activation function to use
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
        """

        super(MTLArchitecture, self).__init__()

        self.RELossLambda = re_lambda
        # self.shared_layers = SharedRNN(num_word_types, shared_layer_size, num_char_types,
        #                                char_dim, hidden_dim, dropout, num_layers_shared,
        #                                recurrent_unit, device)
        self.shared_layers = SharedTransformer(shared_layer_size, num_char_types,
                                       char_dim, dropout, num_layers_shared,
                                       recurrent_unit, device)

        # self.ner_layers = NERSpecificRNN(shared_layer_size, num_tag_types, hidden_dim, dropout,
        #                                  num_layers_ner, init, label_embeddings_size,
        #                                  e1_activation_type, recurrent_unit)

        # self.re_layers = RESpecificRNN(shared_layer_size, num_rel_types, hidden_dim, dropout, re_dropout,
        #                                num_layers_re, label_embeddings_size, re_ff1_size,
        #                                r1_activation_type, recurrent_unit, device)

        self.ner_layers = NERSpecificTransformer(shared_layer_size, num_tag_types, hidden_dim, dropout,
                                         num_layers_ner, init, label_embeddings_size,
                                         e1_activation_type, device)

        self.re_layers = RESpecificTransformer(shared_layer_size, num_rel_types, dropout, re_dropout,
                                       num_layers_re, label_embeddings_size, re_ff1_size,
                                       r1_activation_type, device)

        self.loss = nn.CrossEntropyLoss()

    def score(self, X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents):
        """
        Evaluation through all the shared, NER and RE RNNs.

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
        ner_preds, ner_tag_embeddings = self.ner_layers.scorer(shared_representations, Y)
        re_scores = self.re_layers.scorer(shared_representations, ner_tag_embeddings, rstartseqs, rendseqs, rseqs)
        return ner_preds, re_scores

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
        ner_score, ner_tag_embeddings = self.ner_layers(shared_representations, Y)
        re_score = self.re_layers(shared_representations, ner_tag_embeddings, rstartseqs, rendseqs, rseqs)
        return ner_score, re_score

    def do_epoch(self, epoch_num, train_batches, clip, optim, logger=None, check_interval=200):
        """
        Run the forward pass in multiple epochs across training batches.

        :param epoch_num: number of epochs
        :param train_batches: the training data batches
        :param optim: the optimiser used for minimising the loss
        :param check_interval: save the results once after this many intervals
        :return:
        """

        self.train()
        print("\nTraining...")

        output = {}
        for batch_num, (X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents) in enumerate(train_batches):
            optim.zero_grad()
            NER_forward_result, RE_forward_result = self.forward(X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents)
            loss_NER, loss_RE = NER_forward_result["loss"], RE_forward_result["loss"]
            final_loss = loss_NER + self.RELossLambda * loss_RE
            final_loss.backward()

            nn.utils.clip_grad_norm_(self.parameters(), clip)
            optim.step()

            output["loss"] = NER_forward_result['loss'] + RE_forward_result['loss'] if not 'loss' in output else \
                                output["loss"] + (NER_forward_result['loss'] + RE_forward_result['loss'])
            output["ner_loss"] = NER_forward_result['loss'] if not 'ner_loss' in output else \
                                output["ner_loss"] + NER_forward_result['loss']
            output["re_loss"] = RE_forward_result['loss'] if not 're_loss' in output else \
                                output["re_loss"] + RE_forward_result['loss']


            if logger and (batch_num + 1) % check_interval == 0:
                logger.log('Epoch {:3d} | Batch {:5d}/{:5d} | '
                           'Average Loss {:8.4f} | '
                           'Average NER Loss {:8.4f} | '
                           'Average RE Loss {:8.4f} \n'.format(epoch_num, batch_num + 1, len(train_batches), 
                            output['loss'] / (batch_num + 1), output['ner_loss'] / (batch_num + 1), 
                            output['re_loss'] / (batch_num + 1)))

            if math.isnan(output['loss']):
                print('Stopping training since objective is NaN')
                break
        for key in output:
            output[key] /= (batch_num + 1)

        return output

    def evaluate(self, eval_batches, logger=None, tag2y=None, rel2y=None):
        self.eval()
        print("Evaluating...")
        if 'O' in tag2y:
            y2tag = [None for tag in tag2y]
            for tag in tag2y:
                y2tag[tag2y[tag]] = tag
            tp = Counter()
            fp = Counter()
            fn = Counter()

        num_preds = 0
        num_correct = 0
        num_rel_total = 0
        re_tp = 0
        re_fp = 0
        re_fn = 0
        output = dict()
        gold_entities = {}
        for (X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents) in eval_batches:
            try:
                B, T = Y.size()
                ner_preds, re_scores = self.score(X, Y, C, C_lengths, rstartseqs, rendseqs, rseqs, sents)  # B x T x L

                num_preds += B * T
                num_correct += (ner_preds == Y).sum().item()

                if 'O' in tag2y:
                    for i in range(B):
                        gold_bio_labels = [y2tag[Y[i, j].item()]
                                        for j in range(T)]
                        pred_bio_labels = [y2tag[ner_preds[i, j].item()]
                                        for j in range(T)]
                        gold_boundaries = set(get_boundaries(gold_bio_labels))
                        pred_boundaries = set(get_boundaries(pred_bio_labels))
                        for (s, t, entity) in gold_boundaries:
                            gold_entities[entity] = True
                            if (s, t, entity) in pred_boundaries:
                                tp[entity] += 1
                                tp['<all>'] += 1
                            else:
                                fn[entity] += 1
                                fn['<all>'] += 1
                        for (s, t, entity) in pred_boundaries:
                            if not (s, t, entity) in gold_boundaries:
                                fp[entity] += 1
                                fp['<all>'] += 1

                    for ner_actual, ner_pred, rstartseq, rendseq, rseq, re_score in \
                            zip(Y, ner_preds, rstartseqs, rendseqs, rseqs, re_scores):

                        rstart_list = rstartseq.tolist()
                        rend_list = rendseq.tolist()
                        rseq_list = rseq.tolist()
                        num_rel_total += len(rseq_list)
                        gold_bio_labels = [y2tag[ner_actual[j].item()] for j in range(T)]
                        pred_bio_labels = [y2tag[ner_pred[j].item()] for j in range(T)]
                        gold_boundaries = set(get_boundaries(gold_bio_labels))
                        pred_boundaries = set(get_boundaries(pred_bio_labels))
                        for rel_start, rel_end, rel_ind, re_sc in zip(rstart_list, rend_list, \
                                                                        rseq_list, re_score):

                            score = np.asarray(re_sc.tolist())
                            max_score = np.max(score)
                            arg_max = np.argmax(score)

                            first_entity_success, second_entity_success = False, False
                            for (s, t, entity) in gold_boundaries:
                                if t == rel_start and (s, t, entity) in pred_boundaries:
                                    first_entity_success = True
                                if t == rel_end and (s, t, entity) in pred_boundaries:
                                    second_entity_success = True
                            ner_successful = first_entity_success and second_entity_success
                            # print(ner_successful, rel_ind, re_sc.tolist()[rel_ind])

                            if max_score >= 0.9:
                                if arg_max == rel_ind and ner_successful is True:
                                    re_tp += 1
                                else:
                                    re_fp += 1
                            else:
                                re_fn += 1
                
            except Exception as e:
                logger.log('-' * 89)
                logger.log('X {}, Y: {}, C: {}, C_len: {} \n Error: {}'.format(X, Y, C, C_lengths, e))
                continue

        output["ner_acc"] = 100 * num_correct / num_preds
        output["re_precision"] = 100 * re_tp / (re_tp + re_fp + 1e-16)
        output["re_recall"] = 100 * re_tp / (re_tp + re_fn + 1e-16)
        output["re_f1"] = (2*output["re_recall"]*output["re_precision"])/(output["re_recall"] + output["re_precision"] + 1e-16)

        if 'O' in tag2y:
            for e in list(gold_entities) + ['<all>']:
                p_denom = tp[e] + fp[e]
                r_denom = tp[e] + fn[e]
                p_e = 100 * tp[e] / p_denom if p_denom > 0 else 0
                r_e = 100 * tp[e] / r_denom if r_denom > 0 else 0
                f1_denom = p_e + r_e
                f1_e = 2 * p_e * r_e / f1_denom if f1_denom > 0 else 0
                output['ner_p_%s' % e] = p_e
                output['ner_r_%s' % e] = r_e
                output['ner_f1_%s' % e] = f1_e

        logger.log("NER: P {}, R {}, F1 {} | RE: P {}, R {}, F1: {}".format(output['ner_p_<all>'],
                    output['ner_r_<all>'], output['ner_f1_<all>'], output['re_precision'],
                    output['re_recall'], output['re_f1']))
        return output

class SharedTransformer(nn.Module):
    """
    A shared transformer layer that takes in the initial words, converts them to respective embeddings
    and passes them to the respective NER and RE specific layers.
    """
    ELMODim = 1024
    GloveDim = 300
    OneHotDim = 7

    def __init__(self, shared_layer_size, num_char_types, \
                 char_dim, dropout, num_layers, recurrent_unit="gru", \
                 device="cpu"):
        """
        :param num_word_types:
        :param shared_layer_size:
        :param num_char_types:
        :param char_dim:
        :param dropout:
        :param num_layers:
        :param recurrent_unit:
        """

        super(SharedTransformer, self).__init__()
        self.CharDim = char_dim
        self.Pad_ind = 0
        self.device = device
        word_dim = self.ELMODim + self.GloveDim + self.CharDim
        # Initialise char-embedding BiRNN
        self.cemb = nn.Embedding(num_char_types, self.CharDim, padding_idx=self.Pad_ind)
        self.charRNN = CharRNN(self.cemb, 1, recurrent_unit)
        self.dropout = nn.Dropout(p=dropout)

        self.ff = nn.Linear(word_dim, 2*shared_layer_size)
        self.transformer = TransformerModel(word_dim, nhead=12, nhid=word_dim, nlayers=num_layers)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, char_encoded, C_lengths, raw_sentences):
        """
        Pass the input sentences through the GRU layers.
        :param X: batch of sentences
        :return:
        """

        batch_size = len(raw_sentences)
        elmo_embeddings = load_elmo_embeddings(raw_sentences).to(self.device)
        glove_embeddings = load_glove_embeddings(raw_sentences).to(self.device)
        char_embeddings = self.charRNN(char_encoded, C_lengths).to(self.device)
        # one_hot_embeddings = load_onehot_embeddings(raw_sentences).to(self.device)
        num_words, char_dim = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size, num_words // batch_size, char_dim)
        final_embeddings = torch.cat([elmo_embeddings, glove_embeddings, char_embeddings], dim=2)

        shared_output = self.transformer(final_embeddings)
        shared_output = self.ff(shared_output)
        shared_output = self.ff(final_embeddings) + nn.BatchNorm1d(shared_output.shape[2]).to(self.device)(shared_output.transpose(1, 2)).transpose(1, 2)
        return shared_output


class SharedRNN(nn.Module):
    """
    A shared Bidirectional GRU layer that takes in the initial words, converts them to respective embeddings
    and passes them to the respective NER and RE specific layers.
    """

    ELMODim = 1024
    GloveDim = 300
    OneHotDim = 7

    def __init__(self, num_word_types, shared_layer_size, num_char_types, \
                 char_dim, hidden_dim, dropout, num_layers, recurrent_unit="gru", \
                 device="cpu"):
        """
        :param num_word_types:
        :param shared_layer_size:
        :param num_char_types:
        :param char_dim:
        :param dropout:
        :param num_layers:
        :param recurrent_unit:
        """

        super(SharedRNN, self).__init__()
        self.CharDim = char_dim
        self.Pad_ind = 0
        self.device = device
        word_dim = self.ELMODim + self.GloveDim + self.CharDim + self.OneHotDim

        # Initialise char-embedding BiRNN
        self.cemb = nn.Embedding(num_char_types, self.CharDim, padding_idx=self.Pad_ind)
        self.charRNN = CharRNN(self.cemb, 1, recurrent_unit)
        self.dropout = nn.Dropout(p=dropout)

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
        elmo_embeddings = load_elmo_embeddings(raw_sentences).to(self.device)
        glove_embeddings = load_glove_embeddings(raw_sentences).to(self.device)
        char_embeddings = self.charRNN(char_encoded, C_lengths).to(self.device)
        one_hot_embeddings = load_onehot_embeddings(raw_sentences).to(self.device)
        num_words, char_dim = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size, num_words // batch_size, char_dim)
        final_embeddings = torch.cat([elmo_embeddings, glove_embeddings, char_embeddings, one_hot_embeddings], dim=2)
        # Dropout pre BiRNN
        final_embeddings = self.dropout(final_embeddings)
        # Get the shared layer representations.
        shared_output, _ = self.wordRNN(final_embeddings)   # output: B x T x 512 (hidden dim)
        return shared_output

class NERSpecificTransformer(nn.Module):
    """
    NER specific transformer layers that take in the shared representations from the shared layers and calculates
    the respective NER scores.
    """

    def __init__(self, shared_layer_size, num_tag_types, hidden_dim, dropout, num_layers, \
                 init, label_embeddings_size, activation_type="relu", device='cpu'):
        """ 
        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_tag_types: unique tags of the model, will be used by NER specific layers
        :param hidden_dim: the hidden layer dimension for FFe1
        :param dropout: dropout values for nodes in biRNN
        :param num_layers: number of layers in this biRNN
        :label_embeddings_size: label embedding size
        :param activation_type: the type of activation function to use
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
        """

        super(NERSpecificTransformer, self).__init__()

        self.Pad_ind = 0
        self.tag_embeddings = nn.Embedding(num_tag_types, label_embeddings_size, padding_idx=self.Pad_ind)
        nn.init.xavier_uniform_(self.tag_embeddings.weight.data)

        self.device = device
        self.transformer = TransformerModel(2*shared_layer_size, nhead=8, nhid=2*shared_layer_size, nlayers=num_layers)

        self.dropout = nn.Dropout(p=dropout)
        self.FFNNe1 = nn.Linear(2 * shared_layer_size, hidden_dim)
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
        # Dropout before transformer
        # shared_representations = self.dropout(shared_representations)
        shared_output = self.transformer(shared_representations)
        ner_representation = shared_representations + nn.BatchNorm1d(shared_output.shape[2]).to(self.device)(shared_output.transpose(1, 2)).transpose(1, 2)
        scores = self.FFNNe2(self.activation(self.FFNNe1(ner_representation)))
        loss = self.loss(scores, Y)
        tag_embeddings = self.tag_embeddings(Y)
        return {'loss': loss}, tag_embeddings

    def scorer(self, shared_representations, Y):
        """
        Score the representation at evaluation time

        :param shared_representations: tensor of shared representations from the shared layer.
        :param Y: the label NER tags for the input sentences
        :return: NER scores
        """
        shared_output = self.transformer(shared_representations)
        ner_representation = shared_representations + nn.BatchNorm1d(shared_output.shape[2]).to(self.device)(shared_output.transpose(1, 2)).transpose(1, 2)
        scores = self.FFNNe2(self.activation(self.FFNNe1(ner_representation)))
        _, preds = self.loss.decode(scores)  # B x T
        tag_embeddings = self.tag_embeddings(preds)
        return preds, tag_embeddings

class NERSpecificRNN(nn.Module):
    """
    NER specific bidirectional GRU layers that take in the shared representations from the shared layers and calculates
    the respective NER scores.
    """

    def __init__(self, shared_layer_size, num_tag_types, hidden_dim, dropout, num_layers, \
                 init, label_embeddings_size, activation_type="relu", recurrent_unit="gru"):
        """        print(batched[0])
        s
        Initialise.

        :param shared_layer_size: final output size of the shared layers, to be as inputs to task-specific layers
        :param num_tag_types: unique tags of the model, will be used by NER specific layers
        :param hidden_dim: the hidden layer dimension for FFe1
        :param dropout: dropout values for nodes in biRNN
        :param num_layers: number of layers in this biRNN
        :label_embeddings_size: label embedding size
        :param activation_type: the type of activation function to use
        :param recurrent_unit: the type of recurrent unit to use for biRNN - GRU or LSTM
        """

        super(NERSpecificRNN, self).__init__()

        self.Pad_ind = 0
        self.tag_embeddings = nn.Embedding(num_tag_types, label_embeddings_size, padding_idx=self.Pad_ind)
        nn.init.xavier_uniform_(self.tag_embeddings.weight.data)

        if recurrent_unit == "gru":
            self.birnn = nn.GRU(2 * shared_layer_size, shared_layer_size, num_layers, bidirectional=True)
        else:
            self.birnn = nn.LSTM(2 * shared_layer_size, shared_layer_size, num_layers, bidirectional=True)

        self.dropout = nn.Dropout(p=dropout)
        self.FFNNe1 = nn.Linear(2 * shared_layer_size, hidden_dim)
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
        # Dropout before biRNN
        shared_representations = self.dropout(shared_representations)
        ner_representation, _ = self.birnn(shared_representations)
        scores = self.FFNNe2(self.activation(self.FFNNe1(ner_representation)))
        loss = self.loss(scores, Y)
        tag_embeddings = self.tag_embeddings(Y)
        return {'loss': loss}, tag_embeddings

    def scorer(self, shared_representations, Y):
        """
        Score the representation at evaluation time

        :param shared_representations: tensor of shared representations from the shared layer.
        :param Y: the label NER tags for the input sentences
        :return: NER scores
        """
        ner_representation, _ = self.birnn(shared_representations)
        scores = self.FFNNe2(self.activation(self.FFNNe1(ner_representation)))
        _, preds = self.loss.decode(scores)  # B x T
        tag_embeddings = self.tag_embeddings(preds)
        
        return preds, tag_embeddings

class RESpecificTransformer(nn.Module):
    """
    RE specific bidirectional GRU layers that take in the shared representations from the shared layers and calculates
    the respective RE scores.
    """

    def __init__(self, shared_layer_size, num_rel_types, dropout, re_dropout, num_layers, \
                    label_embeddings_size, re_ff1_size, activation_type="relu", device="cpu"):
        """
        Initialise.

        :param shared_layer_size:
        :param num_rel_types:
        :param dropout:
        :param num_layers:
        :param label_embeddings_size:
        :param activation_type:
        :param recurrent_unit:
        """

        super(RESpecificTransformer, self).__init__()

        self.device = device

        self.transformer = TransformerModel(2*shared_layer_size, nhead=8, nhid=2*shared_layer_size, nlayers=num_layers)

        final_re_entity_embedding_size = 2 * shared_layer_size + label_embeddings_size

        self.dropout = nn.Dropout(p=dropout)
        self.re_dropout = nn.Dropout(p=re_dropout)

        self.FFNNr1 = nn.Linear(final_re_entity_embedding_size, re_ff1_size)

        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "tanh":
            self.activation = nn.Tanh()
        elif activation_type == "gelu":
            self.activation = nn.GELU()

        self.FFNNr2 = nn.Linear((2 * re_ff1_size) + 1 + num_rel_types, num_rel_types)

        # Initialise matrix for DistMult score calculation
        self.M = []
        for _ in range(num_rel_types):
            self.M.append(torch.diag(torch.rand(size=(final_re_entity_embedding_size,))))
        self.M = torch.stack(self.M).to(self.device)
        self.M.requires_grad = True

        self.loss = nn.BCELoss()

    def _trim_embeddings(self, embeddings, rstartseqs, rendseqs, rseqs):
        """
        :param embeddings:
        :param rstartseqs:
        :param rendseqs:
        :param rseqs:
        :return:
        """

        B, T, E = embeddings.shape
        batches = []
        for i in range(B):  # Each sentence
            rstart_list = rstartseqs[i].tolist()
            rend_list = rendseqs[i].tolist()
            rseq_list = rseqs[i].tolist()

            end_tokens_of_first_entities = embeddings[i][rstartseqs[i]]
            end_tokens_of_second_entities = embeddings[i][rendseqs[i]]
            all_end_tokens_embeddings = torch.cat([end_tokens_of_first_entities, end_tokens_of_second_entities], dim=0)
            all_end_tokens_indices = rstart_list + rend_list

            # All possible entity pairs
            all_possible_entity_pairs_embeddings = list(itertools.combinations(all_end_tokens_embeddings, 2))
            all_possible_entity_pairs_indices = list(itertools.combinations(all_end_tokens_indices, 2))

            # True RE labels
            true_RE_labels = defaultdict(list)
            for (start, end, relation) in zip(rstart_list, rend_list, rseq_list):
                true_RE_labels[relation].append((start, end))

            batches.append((all_possible_entity_pairs_embeddings, all_possible_entity_pairs_indices, true_RE_labels))

        return batches

    def _RE_scoring_layers(self, first_entity_embedding, second_entity_embedding):
        """

        :param first_entity_embedding:
        :param second_entity_embedding:
        :return:
        """

        # Calculate DistMult score
        first_entity_embedding = first_entity_embedding.unsqueeze(0).to(self.device)  # (1 x p)
        second_entity_embedding = second_entity_embedding.unsqueeze(0).to(self.device)  # (1 x p)
        distmult_scores = torch.matmul(first_entity_embedding, torch.matmul(self.M, second_entity_embedding.T))
        distmult_scores = distmult_scores.squeeze(2)
        distmult_scores = distmult_scores.T  # 1 x num_rel_types

        # Hidden representations of entities
        first_entity_hidden_repr = self.activation(self.FFNNr1(first_entity_embedding))
        second_entity_hidden_repr = self.activation(self.FFNNr1(second_entity_embedding))

        # Cosine distance
        cosine_distance = torch.cosine_similarity(first_entity_hidden_repr, second_entity_hidden_repr)
        cosine_distance = cosine_distance.unsqueeze(1)  # 1 x 1

        # Concatenate everything
        final_embedding = torch.cat([first_entity_hidden_repr, second_entity_hidden_repr,
                                     cosine_distance, distmult_scores], dim=1).to(self.device)

        # RE Scores (Sij)
        RE_scores_for_entity_pair = self.FFNNr2(final_embedding)
        sigmoid_RE_scores = torch.sigmoid(RE_scores_for_entity_pair)
        return sigmoid_RE_scores

    def _calculate_RE_scores(self, batches):
        """
        :param batches:
        :return:
        """

        batch_loss = 0
        for (entity_pairs_embeddings, entity_pairs_indices, true_RE_labels) in batches:
            for i, (first_entity_embedding, second_entity_embedding) in enumerate(entity_pairs_embeddings):
                predicted_RE_scores_for_entity_pair = self._RE_scoring_layers(first_entity_embedding,
                                                                              second_entity_embedding).to(self.device)

                # Create ground truth RE labels for current entity pair
                first_entity_end_index = entity_pairs_indices[i][0]
                second_entity_end_index = entity_pairs_indices[i][1]
                target_RE_Labels_for_entity_pair = torch.zeros(predicted_RE_scores_for_entity_pair.shape).to(self.device)
                for i in range(predicted_RE_scores_for_entity_pair.shape[1]):
                    # RE_score_for_current_relation = torch.stack([1 - predicted_RE_scores_for_entity_pair[:, i],
                    #                                              predicted_RE_scores_for_entity_pair[:, i]])
                    # RE_score_for_current_relation = RE_score_for_current_relation.T

                    # If the particular relation exists between the current entity pair then y = 1 else 0
                    if (first_entity_end_index, second_entity_end_index) in true_RE_labels[i]:
                        target_RE_Labels_for_entity_pair[:, i] = 1

                batch_loss += self.loss(predicted_RE_scores_for_entity_pair, target_RE_Labels_for_entity_pair)
        return batch_loss

    def forward(self, shared_representations, ner_tag_embeddings, rstartseqs, rendseqs, rseqs):
        """
        :param shared_representations:
        :param Y:
        :param tag_embeddings:
        :param rstartseqs:
        :param rendseqs
        :param rseqs
        :return:
        """

        # Pre biRNN dropout
        # shared_representations = self.dropout(shared_representations)
        shared_output = self.transformer(shared_representations)
        re_representation = shared_representations + nn.BatchNorm1d(shared_output.shape[2]).to(self.device)(shared_output.transpose(1, 2)).transpose(1, 2)
        re_representation = torch.cat([re_representation, ner_tag_embeddings], dim=2)

        # Pre RE Scoring Dropout
        re_representation = self.re_dropout(re_representation)

        batches = self._trim_embeddings(re_representation, rstartseqs, rendseqs, rseqs)
        loss = self._calculate_RE_scores(batches)
        return {'loss': loss}

    def scorer(self, shared_representations, ner_tag_embeddings, rstartseqs, rendseqs, rseqs):
        """
        :param shared_representations:
        :param Y:
        :param tag_embeddings:
        :param rstartseqs:
        :param rendseqs
        :param rseqs
        :return:
        """

        shared_output = self.transformer(shared_representations)
        re_representation = shared_representations + nn.BatchNorm1d(shared_output.shape[2]).to(self.device)(shared_output.transpose(1, 2)).transpose(1, 2)
        re_representation = torch.cat([re_representation, ner_tag_embeddings], dim=2)

        # Pre RE Scoring Dropout. In model.eval dropout won't work
        re_representation = self.dropout(re_representation)

        batched = []
        batches = self._trim_embeddings(re_representation, rstartseqs, rendseqs, rseqs)
        for i, (entity_pairs_embeddings, entity_pairs_indices, true_RE_labels) in enumerate(batches):
            rstart_list = rstartseqs[i].tolist()
            rend_list = rendseqs[i].tolist()
            filter_r = set(zip(rstart_list, rend_list))
            pairs = []
            visited = set()
            for j, (first_entity_embedding, second_entity_embedding) in enumerate(entity_pairs_embeddings):
                predicted_RE_scores_for_entity_pair = self._RE_scoring_layers(first_entity_embedding,
                                                                              second_entity_embedding)

                first_entity_end_index = entity_pairs_indices[j][0]
                second_entity_end_index = entity_pairs_indices[j][1]
                if (first_entity_end_index, second_entity_end_index) in filter_r and \
                    (first_entity_end_index, second_entity_end_index) not in visited:
                    pairs.append(predicted_RE_scores_for_entity_pair.squeeze(0))
                visited.add((first_entity_end_index, second_entity_end_index))
            batched.append(pairs)
        return batched

class RESpecificRNN(nn.Module):
    """
    RE specific bidirectional GRU layers that take in the shared representations from the shared layers and calculates
    the respective RE scores.
    """

    def __init__(self, shared_layer_size, num_rel_types, hidden_dim, dropout, re_dropout, num_layers, \
                    label_embeddings_size, re_ff1_size, activation_type="relu", \
                    recurrent_unit="gru", device="cpu"):
        """
        Initialise.

        :param shared_layer_size:
        :param num_rel_types:
        :param dropout:
        :param num_layers:
        :param label_embeddings_size:
        :param activation_type:
        :param recurrent_unit:
        """

        super(RESpecificRNN, self).__init__()

        self.device = device

        # Add check for 0 task-specific layers, it'll be used while hyperparameter tuning
        if recurrent_unit == "gru":
            self.birnn = nn.GRU(2 * shared_layer_size, shared_layer_size, num_layers, bidirectional=True)
        else:
            self.birnn = nn.LSTM(2 * shared_layer_size, shared_layer_size, num_layers, bidirectional=True)

        final_re_entity_embedding_size = 2 * shared_layer_size + label_embeddings_size

        self.dropout = nn.Dropout(p=dropout)
        self.re_dropout = nn.Dropout(p=re_dropout)

        self.FFNNr1 = nn.Linear(final_re_entity_embedding_size, re_ff1_size)

        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "tanh":
            self.activation = nn.Tanh()
        elif activation_type == "gelu":
            self.activation = nn.GELU()

        self.FFNNr2 = nn.Linear((2 * re_ff1_size) + 1 + num_rel_types, num_rel_types)

        # Initialise matrix for DistMult score calculation
        self.M = []
        for _ in range(num_rel_types):
            self.M.append(torch.diag(torch.rand(size=(final_re_entity_embedding_size,))))
        self.M = torch.stack(self.M).to(self.device)
        self.M.requires_grad = True

        self.loss = nn.BCELoss()

    def _trim_embeddings(self, embeddings, rstartseqs, rendseqs, rseqs):
        """
        :param embeddings:
        :param rstartseqs:
        :param rendseqs:
        :param rseqs:
        :return:
        """

        B, T, E = embeddings.shape
        batches = []
        for i in range(B):  # Each sentence
            rstart_list = rstartseqs[i].tolist()
            rend_list = rendseqs[i].tolist()
            rseq_list = rseqs[i].tolist()

            end_tokens_of_first_entities = embeddings[i][rstartseqs[i]]
            end_tokens_of_second_entities = embeddings[i][rendseqs[i]]
            all_end_tokens_embeddings = torch.cat([end_tokens_of_first_entities, end_tokens_of_second_entities], dim=0)
            all_end_tokens_indices = rstart_list + rend_list

            # All possible entity pairs
            all_possible_entity_pairs_embeddings = list(itertools.combinations(all_end_tokens_embeddings, 2))
            all_possible_entity_pairs_indices = list(itertools.combinations(all_end_tokens_indices, 2))

            # True RE labels
            true_RE_labels = defaultdict(list)
            for (start, end, relation) in zip(rstart_list, rend_list, rseq_list):
                true_RE_labels[relation].append((start, end))

            batches.append((all_possible_entity_pairs_embeddings, all_possible_entity_pairs_indices, true_RE_labels))

        return batches

    def _RE_scoring_layers(self, first_entity_embedding, second_entity_embedding):
        """

        :param first_entity_embedding:
        :param second_entity_embedding:
        :return:
        """

        # Calculate DistMult score
        first_entity_embedding = first_entity_embedding.unsqueeze(0).to(self.device)  # (1 x p)
        second_entity_embedding = second_entity_embedding.unsqueeze(0).to(self.device)  # (1 x p)
        distmult_scores = torch.matmul(first_entity_embedding, torch.matmul(self.M, second_entity_embedding.T))
        distmult_scores = distmult_scores.squeeze(2)
        distmult_scores = distmult_scores.T  # 1 x num_rel_types

        # Hidden representations of entities
        first_entity_hidden_repr = self.activation(self.FFNNr1(first_entity_embedding))
        second_entity_hidden_repr = self.activation(self.FFNNr1(second_entity_embedding))

        # Cosine distance
        cosine_distance = torch.cosine_similarity(first_entity_hidden_repr, second_entity_hidden_repr)
        cosine_distance = cosine_distance.unsqueeze(1)  # 1 x 1

        # Concatenate everything
        final_embedding = torch.cat([first_entity_hidden_repr, second_entity_hidden_repr,
                                     cosine_distance, distmult_scores], dim=1).to(self.device)

        # RE Scores (Sij)
        RE_scores_for_entity_pair = self.FFNNr2(final_embedding)
        sigmoid_RE_scores = torch.sigmoid(RE_scores_for_entity_pair)
        return sigmoid_RE_scores

    def _calculate_RE_scores(self, batches):
        """
        :param batches:
        :return:
        """

        batch_loss = 0
        for (entity_pairs_embeddings, entity_pairs_indices, true_RE_labels) in batches:
            for i, (first_entity_embedding, second_entity_embedding) in enumerate(entity_pairs_embeddings):
                predicted_RE_scores_for_entity_pair = self._RE_scoring_layers(first_entity_embedding,
                                                                              second_entity_embedding).to(self.device)

                # Create ground truth RE labels for current entity pair
                first_entity_end_index = entity_pairs_indices[i][0]
                second_entity_end_index = entity_pairs_indices[i][1]
                target_RE_Labels_for_entity_pair = torch.zeros(predicted_RE_scores_for_entity_pair.shape).to(self.device)
                for i in range(predicted_RE_scores_for_entity_pair.shape[1]):
                    # RE_score_for_current_relation = torch.stack([1 - predicted_RE_scores_for_entity_pair[:, i],
                    #                                              predicted_RE_scores_for_entity_pair[:, i]])
                    # RE_score_for_current_relation = RE_score_for_current_relation.T

                    # If the particular relation exists between the current entity pair then y = 1 else 0
                    if (first_entity_end_index, second_entity_end_index) in true_RE_labels[i]:
                        target_RE_Labels_for_entity_pair[:, i] = 1

                batch_loss += self.loss(predicted_RE_scores_for_entity_pair, target_RE_Labels_for_entity_pair)
        return batch_loss

    def forward(self, shared_representations, ner_tag_embeddings, rstartseqs, rendseqs, rseqs):
        """
        :param shared_representations:
        :param Y:
        :param tag_embeddings:
        :param rstartseqs:
        :param rendseqs
        :param rseqs
        :return:
        """

        # Pre biRNN dropout
        shared_representations = self.dropout(shared_representations)
        re_representation, _ = self.birnn(shared_representations)
        re_representation = torch.cat([re_representation, ner_tag_embeddings], dim=2)

        # Pre RE Scoring Dropout
        re_representation = self.re_dropout(re_representation)

        batches = self._trim_embeddings(re_representation, rstartseqs, rendseqs, rseqs)
        loss = self._calculate_RE_scores(batches)
        return {'loss': loss}

    def scorer(self, shared_representations, ner_tag_embeddings, rstartseqs, rendseqs, rseqs):
        """
        :param shared_representations:
        :param Y:
        :param tag_embeddings:
        :param rstartseqs:
        :param rendseqs
        :param rseqs
        :return:
        """

        re_representation, _ = self.birnn(shared_representations)
        re_representation = torch.cat([re_representation, ner_tag_embeddings], dim=2)

        # Pre RE Scoring Dropout. In model.eval dropout won't work
        re_representation = self.dropout(re_representation)

        batched = []
        batches = self._trim_embeddings(re_representation, rstartseqs, rendseqs, rseqs)
        for i, (entity_pairs_embeddings, entity_pairs_indices, true_RE_labels) in enumerate(batches):
            rstart_list = rstartseqs[i].tolist()
            rend_list = rendseqs[i].tolist()
            filter_r = set(zip(rstart_list, rend_list))
            pairs = []
            visited = set()
            for j, (first_entity_embedding, second_entity_embedding) in enumerate(entity_pairs_embeddings):
                predicted_RE_scores_for_entity_pair = self._RE_scoring_layers(first_entity_embedding,
                                                                              second_entity_embedding)

                first_entity_end_index = entity_pairs_indices[j][0]
                second_entity_end_index = entity_pairs_indices[j][1]
                if (first_entity_end_index, second_entity_end_index) in filter_r and \
                    (first_entity_end_index, second_entity_end_index) not in visited:
                    pairs.append(predicted_RE_scores_for_entity_pair.squeeze(0))
                visited.add((first_entity_end_index, second_entity_end_index))
            batched.append(pairs)
        return batched


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
