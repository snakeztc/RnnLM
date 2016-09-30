import numpy as np


class SeqDataLoader(object):
    batch_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None

    def __init__(self, name, data, equal_len_batch=False):
        self.name = name
        self.data = data
        self.data_size = len(data)
        self.equal_len_batch = equal_len_batch
        if equal_len_batch:
            all_lens = [len(line) for line in self.data]
            self.indexes = list(np.argsort(all_lens))
        else:
            self.indexes = range(len(self.data))

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

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
        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size

        # if shuffle and we don't want to group lines, shuffle index
        if shuffle and not self.equal_len_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        # if shuffle and we want to group lines, shuffle batch indexes
        if shuffle and self.equal_len_batch:
            self._shuffle_batch_indexes()

        print("%s begins training with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

