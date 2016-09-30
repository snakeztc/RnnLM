import numpy as np


class SeqDataLoader(object):
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.indexes = range(len(self.data))
        self.data_size = len(data)

        # iteration related
        self.ptr = 0
        self.batch_size = 0
        self.num_batch = None

        # find the longest sequence length
        max_seq_len = np.max([len(row) for row in data])
        print "Longest sentence in corpus is %d" % max_seq_len

    def _shuffle(self):
        np.random.shuffle(self.indexes)

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        input_lens = np.array([len(row)-1 for row in rows], dtype=np.int32)
        max_len = np.max(input_lens)
        inputs = np.zeros((self.batch_size, max_len), dtype=np.int32)
        outputs = np.zeros((self.batch_size, max_len), dtype=np.int32)
        for idx, row in enumerate(rows):
            inputs[idx, 0:input_lens[idx]] = row[0:-1]
            outputs[idx, 0:input_lens[idx]] = row[1:]
        return inputs, input_lens, outputs

    def epoch_init(self, batch_size, shuffle=True):
        if shuffle:
            self._shuffle()
        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size
        print("%s begins training with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.indexes[self.ptr * self.batch_size:min((self.ptr+1)*self.batch_size, self.data_size)]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

