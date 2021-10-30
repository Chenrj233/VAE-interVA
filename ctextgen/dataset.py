from torchtext import data, datasets
from torchtext.vocab import GloVe
import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

class SST_Dataset:
    #0: positive
    #1: negative
    def __init__(self, emb_dim=50, mbsize=32, fixlen=16, init_token='<start>', eos_token='<eos>', pad_token="<pad>", unk_token="<unk>"):
        self.fixlen = fixlen
        if self.fixlen != None:
            self.TEXT = data.Field(init_token=init_token, eos_token=eos_token, pad_token=pad_token, unk_token= unk_token,
                                   lower=True, tokenize='spacy', fix_length=self.fixlen+2)
        else:
            self.TEXT = data.Field(init_token=init_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token,
                                   lower=True, tokenize='spacy')
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        if fixlen == None:
            print('none')
            f = lambda ex: ex.label != 'neutral'
            train, val, test = datasets.SST.splits(
                self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
                filter_pred=f
            )
        else:
            f = lambda ex: len(ex.text) <= self.fixlen and ex.label != 'neutral'

            train, val, test = datasets.SST.splits(
                self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
                filter_pred=f
            )
        print(len(train))
        print(len(val))
        print(len(test))

        self.trainlen = len(train)
        self.vallen = len(val)
        self.testlen = len(test)

        self.train = train
        self.val = val
        self.test = test

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1,
            shuffle=False, repeat=False
        )
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)
        self.test_iter = iter(self.test_iter)


    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_test_batch(self, gpu=False):
        batch = next(self.test_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def validation_size(self):
        return self.vallen

    def train_size(self):
        return self.trainlen

    def test_size(self):
        return self.testlen


    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

    def idxs2sentencelist(self, idxs):
        return [self.TEXT.vocab.itos[i] for i in idxs]

    def tokens2sentence(self, tokens):
        return ' '.join([token for token in tokens])
