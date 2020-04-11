import os
from utils import load_vertical_tagged_data
import torch
from torch.nn.utils.rnn import pad_sequence

class Dataset():
    """

    """

    def __init__(self,
                 data_dir='./data/datasets/conll04',
                 data_name='conll04',
                 batch_size=64,
                 device='cpu',
                 lower=True,
                 vocab_size=1000000000,
                 pad='<pad>',
                 unk='<unk>'):
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.device = device
        self.lower = lower
        self.vocab_size = vocab_size
        self.PAD = pad
        self.UNK = unk
        self.PAD_ind = 0
        self.UNK_ind = 1
        self.populate_attributes()

    def populate_attributes(self):
        # Load training portion.
        (self.wordseqs_train, self.tagseqs_train, self.relseqs_train, self.charseqslist_train,\
        self.wordcounter_train, self.tagcounter_train, self.relcounter_train, self.charcounter_train)\
         = load_vertical_tagged_data(os.path.join(self.data_dir, self.data_name + '_train.json'))

        # Create index maps from training portion.
        self.word2x = self.get_imap(self.wordcounter_train, max_size=self.vocab_size, lower=self.lower, pad_unk=True)
        self.tag2y = self.get_imap(self.tagcounter_train, max_size=None, lower=False, pad_unk=True)
        self.relation2y = self.get_imap(self.relcounter_train, max_size=None, lower=False, pad_unk=False)
        self.char2c = self.get_imap(self.charcounter_train, max_size=None, lower=self.lower, pad_unk=True)

        # Load validation and test portions.
        (self.wordseqs_val, self.tagseqs_val, self.relseqs_val, self.charseqslist_val, _, _, _, _) = load_vertical_tagged_data(
                                                                                    os.path.join(self.data_dir, self.data_name + '_dev.json'))
        (self.wordseqs_test, self.tagseqs_test, self.relseqs_test, self.charseqslist_test, _, _, _, _) = load_vertical_tagged_data(
                                                                                    os.path.join(self.data_dir, self.data_name + '_test.json'))

        # Prepare batches.
        self.batches_train = self.batchfy(self.wordseqs_train, self.tagseqs_train, self.relseqs_train, self.charseqslist_train)
        self.batches_val = self.batchfy(self.wordseqs_val, self.tagseqs_val, self.relseqs_val, self.charseqslist_val)
        self.batches_test = self.batchfy(self.wordseqs_test, self.tagseqs_test, self.relseqs_test, self.charseqslist_test)

    def batchfy(self, wordseqs, tagseqs, relseqs, charseqslist):
        batches = []
        def add_batch(xseqs, yseqs, rstartseqs, rendseqs, rseqs, cseqslist, raw_sentence):
            if not xseqs:
                return
            X = torch.stack(xseqs).to(self.device)  # B x T
            Y = torch.stack(yseqs).to(self.device)  # B x T
            flattened_cseqs = [item for sublist in cseqslist for item in sublist]  # List of BT tensors of varying lengths
            C = pad_sequence(flattened_cseqs, padding_value=self.PAD_ind, batch_first=True).to(self.device)  # BT x T_char
            C_lens = torch.LongTensor([s.shape[0] for s in flattened_cseqs]).to(self.device)
            batches.append((X, Y, C, C_lens, rstartseqs, rendseqs, rseqs, raw_sentence))

        xseqs = []
        yseqs = []
        rstartseqs = []
        rendseqs = []
        rseqs = []
        cseqslist = []
        prev_length = float('inf')
        raw_sentence = []

        for i in range(len(wordseqs)):
            length = len(wordseqs[i])
            assert length <= prev_length  # Assume sequences in decr lengths
            wordseq = [word.lower() for word in wordseqs[i]] if self.lower else wordseqs[i]
            raw_sentence.append(wordseqs[i])
            xseq = torch.LongTensor([self.word2x.get(word, self.UNK_ind) for word in wordseq])
            yseq = torch.LongTensor([self.tag2y.get(tag, self.UNK_ind) for tag in tagseqs[i]])
            rstartseq = []
            rendseq = []
            rseq = []
            for rel in relseqs[i]:
                rstartseq.append(rel[0])
                rendseq.append(rel[1])
                rseq.append(self.relation2y[rel[2]])
            rstartseq = torch.LongTensor(rstartseq)
            rendseq = torch.LongTensor(rendseq)
            rseq = torch.LongTensor(rseq)
            cseqs = [torch.LongTensor([self.char2c[c] for c in word if c in self.char2c])  # Skip unknown
                     for word in wordseqs[i]]  # Use original words

            if length < prev_length or len(xseqs) >= self.batch_size:
                add_batch(xseqs, yseqs, rstartseqs, rendseqs, rseqs, cseqslist, raw_sentence)
                xseqs = []
                yseqs = []
                rstartseqs = []
                rendseqs= []
                rseqs = []
                cseqslist = []
                raw_sentence = []

            xseqs.append(xseq)
            yseqs.append(yseq)
            rstartseqs.append(rstartseq)
            rendseqs.append(rendseq)
            rseqs.append(rseq)
            cseqslist.append(cseqs)
            prev_length = length
            raw_sentence.append(wordseqs[i])

        add_batch(xseqs, yseqs, rstartseqs, rendseqs, rseqs, cseqslist, raw_sentence)

        return batches

    def get_imap(self, counter, max_size=None, lower=False, pad_unk=True):
        if pad_unk:
            imap = {self.PAD: self.PAD_ind, self.UNK: self.UNK_ind}
        else:
            imap = {}
        if max_size is None or len(counter) <= max_size:
            strings = counter.keys()
        else:
            strings = list(zip(*sorted(counter.items(), key=lambda x: x[1],
                                       reverse=True)[:max_size]))[0]
        for string in strings:
            if lower:
                string = string.lower()
            if not string in imap:
                imap[string] = len(imap)
        return imap
