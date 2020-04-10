import os
from utils import load_vertical_tagged_data

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
        (self.wordseqs_train, self.tagseqs_train, self.relseqs_train, self.charseqslist_train, self.wordcounter_train, self.tagcounter_train, self.relcounter_train, self.charcounter_train) \
         = load_vertical_tagged_data(os.path.join(self.data_dir, self.data_name + '_train.json'))

        # Create index maps from training portion.
        self.word2x = self.get_imap(self.wordcounter_train, max_size=self.vocab_size, lower=self.lower, pad_unk=True)
        self.tag2y = self.get_imap(self.tagcounter_train, max_size=None, lower=False, pad_unk=True)
        self.relation2y = self.get_imap(self.relcounter_train, max_size=None, lower=False, pad_unk=False)
        self.char2c = self.get_imap(self.charcounter_train, max_size=None, lower=False)

        # Load validation and test portions.
        (self.wordseqs_val, self.tagseqs_val, self.relseqs_val, self.charseqslist_val, _, _, _, _) = load_vertical_tagged_data(
                                                                        os.path.join(self.data_dir, self.data_name + '_dev.json'))
        (self.wordseqs_test, self.tagseqs_test, self.relseqs_test, self.charseqslist_val, _, _, _, _) = load_vertical_tagged_data(
                                                                        os.path.join(self.data_dir, self.data_name + '_test.json'))

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
