import math

import numpy as np
import torch


def create_dataset(total_data):
    n_users = np.max(total_data[:, 0])
    n_items = np.max(total_data[:, 1])

    X = total_data[:, :2] - 1
    y = total_data[:, 2].astype(np.float32) + 2

    return (n_users, n_items), (X, y)


class ReviewsIterator:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k * bs:(k + 1) * bs], self.y[k * bs:(k + 1) * bs]


def batches(X, y, batch_size=32, shuffle=True):
    for xb, yb in ReviewsIterator(X, y, batch_size, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)


if __name__ == '__main__':
    train = np.load(r"E:\RecommendTuanAnh\train.npy")
    test = np.load(r"E:\RecommendTuanAnh\test.npy")
    total = np.concatenate((train, test), axis=0)

    (n_user, n_item), (X, y) = create_dataset(total)
    for x_batch, y_batch in batches(X, y, batch_size=4):
        print(x_batch)
        print(y_batch)
        break
