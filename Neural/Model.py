import torch
import torch.nn as nn
from itertools import zip_longest
from sklearn.model_selection import train_test_split
import copy
import numpy as np
from Neural.make_data import *


def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


RANDOM_State = 1
set_random_seed(RANDOM_State)


class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_items,
                 n_factors=50, embedding_dropout=0.02,
                 hidden=10, dropouts=0.2):
        super().__init__()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)
        n_last = hidden[-1]

        def gen_layers(n_in):
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)

            for n_out, rate in zip_longest(hidden, dropouts):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out

        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_items, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(*list(gen_layers(n_factors * 2)))
        self.fc = nn.Linear(n_last, 1)
        self._init()

    def forward(self, users, items, minmax=None):
        features = torch.cat([self.u(users), self.m(items)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out * (max_rating - min_rating + 1) + min_rating - 0.5
        return out

    def _init(self):
        """
        Setup embeddings and hidden layers with reasonable initial values.
        """

        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)


def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError('layers configuraiton should be a single number or a list of numbers')


if __name__ == '__main__':
    # Load data
    train = np.load(r"E:\RecommendTuanAnh\train.npy")
    test = np.load(r"E:\RecommendTuanAnh\test.npy")
    total = np.concatenate((train, test), axis=0)
    (n_user, n_item), (X, y) = create_dataset(total)
    # Train-Test split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_State)
    datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
    dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}
    # Init model and hyper parameters
    minmax = (1.0, 3.0)
    net = EmbeddingNet(
        n_users=n_user, n_items=n_item,
        n_factors=150, hidden=[250, 250, 250],
        embedding_dropout=0.6, dropouts=[0.6, 0.6, 0.6])
    lr = 1e-3
    wd = 1e-5
    bs = 2000
    n_epochs = 100
    patience = 10
    no_improvements = 0
    best_loss = np.inf
    best_weights = None
    history = []
    lr_history = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net.to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
    # Training
    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}

        for phase in ('train', 'val'):
            training = phase == 'train'
            running_loss = 0.0
            n_batches = 0
            batch_num = 0
            for batch in batches(*datasets[phase], shuffle=training, batch_size=bs):
                x_batch, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()
                # compute gradients only during 'train' phase

                outputs = net(x_batch[:, 0], x_batch[:, 1], minmax)
                loss = criterion(outputs, y_batch)

                # don't update weights and rates when in 'val' phase
                if training:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            stats[phase] = epoch_loss

            # early stopping: save weights of the best model so far
            if phase == 'val':
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(net.state_dict())
                    no_improvements = 0
                else:
                    no_improvements += 1

        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
        if no_improvements >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break
