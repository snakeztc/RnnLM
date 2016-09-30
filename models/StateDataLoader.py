import numpy as np


class StateDataLoader(object):
    # iteration related
    ptr = 0
    batch_size = 0
    num_batch = None
    batch_indexes = None

    def __init__(self, name, data, curriculum_learning=False):
        self.name = name
        # break data into states
        self.data = []
        for line in data:
            for t_id in range(2, len(line)):
                self.data.append(line[0:t_id])
        if curriculum_learning:
            all_lens = [len(line) for line in self.data]
            self.sorted_indexes = list(np.argsort(all_lens))
        else:
            self.sorted_indexes = range(len(self.data))
        self.data_size = len(self.data)
        print("Create %d sentence state samples" % self.data_size)

    def _shuffle(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        input_lens = np.array([len(row)-1 for row in rows], dtype=np.int32)
        max_len = np.max(input_lens)
        inputs = np.zeros((self.batch_size, max_len), dtype=np.int32)
        outputs = np.zeros(self.batch_size, dtype=np.int32)
        for idx, row in enumerate(rows):
            inputs[idx, 0:input_lens[idx]] = row[0:-1]
            outputs[idx] = row[-1]
        return inputs, input_lens, outputs

    def epoch_init(self, batch_size, shuffle=True):
        # create batch indexes for computation efficiency
        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.sorted_indexes[i*self.batch_size:(i+1)*self.batch_size])
        if shuffle:
            self._shuffle()

        print("%s begins training with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

