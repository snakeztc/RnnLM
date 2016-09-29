from abstract_corpus import Corpus
import os
import cPickle as pkl


class PTBCorpus(Corpus):
    def __init__(self, corpus_path, use_char=False):
        """
        :param corpus_path: the folder that contains the PTB
        :param use_char: use word corpus or character level corpus. Default=False
        """
        self._path = corpus_path
        self.use_char = False
        vocab_count = {}
        cache_path = 'ptb-char.p' if use_char else 'ptb-word.p'
        if os.path.exists(cache_path):
            print "Detect cache corpus. Loading from there..."
            temp = pkl.load(open(cache_path, 'rb'))
            self.train_corpus = temp['train']
            self.valid_corpus = temp['valid']
            self.test_corpus = temp['test']
            self.vocab, self.vocab_freq = temp['vocab']
            print "Done loading cache."
            return

        if use_char:
            train_path = os.path.join(self._path, "ptb.char.train.txt")
            valid_path = os.path.join(self._path, "ptb.char.valid.txt")
            test_path = os.path.join(self._path, "ptb.char.test.txt")
        else:
            train_path = os.path.join(self._path, "ptb.train.txt")
            valid_path = os.path.join(self._path, "ptb.valid.txt")
            test_path = os.path.join(self._path, "ptb.test.txt")
        # read train
        self.train_corpus = []
        self.valid_corpus = []
        self.test_corpus = []
        with open(train_path, 'rb') as train_f:
            lines = train_f.readlines()
            for l in lines:
                tokens = l.split()
                self.train_corpus.append(tokens)
                # update vocab
                for t in tokens:
                    cnt = vocab_count.get(t, 0)
                    vocab_count[t] = cnt + 1

        # read validation
        with open(valid_path, 'rb') as valid_f:
            lines = valid_f.readlines()
            for l in lines:
                self.valid_corpus.append(l.split())

        # read test
        with open(test_path, 'rb') as test_f:
            lines = test_f.readlines()
            for l in lines:
                self.test_corpus.append(l.split())

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, test size %d vocabulary size %d"
              % (len(self.train_corpus),len(self.valid_corpus), len(self.test_corpus), len(vocab_count)))

        sorted_vocab = sorted([(cnt, t) for t, cnt in vocab_count.iteritems()], reverse=True)
        self.vocab = [t for cnt, t in sorted_vocab]
        self.vocab_freq = [cnt for cnt, t in sorted_vocab]
        print("Done loading corpus")
        pkl.dump({'train': self.train_corpus,
                  'valid': self.valid_corpus,
                  'test': self.test_corpus,
                  'vocab': (self.vocab, self.vocab_freq)}, open(cache_path, 'wb'))

    def get_corpus(self):
        return {'train': self.train_corpus, 'valid': self.valid_corpus, 'test': self.test_corpus}


