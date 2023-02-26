from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")


class CF(object):
    def __init__(self, Y_data, total_data, k, dist_func=cosine_similarity, uucF=1):
        self.uucF = uucF
        self.Y_data = Y_data if self.uucF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dis_func = dist_func
        self.Ybar_data = None

        self.n_users = np.max(total_data[:, 0]) + 1
        self.n_items = np.max(total_data[:, 1]) + 1

    def normallize(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy().astype(np.float)
        self.mu = np.zeros((self.n_users,))

        for n in range(0, self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)

            items_ids = self.Y_data[ids, 1]
            ratings = self.Y_data[ids, 2]

            m = np.mean(ratings)
            if np.isnan(m):
                m = 0

            self.mu[n] = m

            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dis_func(self.Ybar.T, self.Ybar.T)

    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normallize()
        self.similarity()

    def fit(self):
        self.refresh()

    def __pred(self, u, i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2:
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:]
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i

        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u, i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        if self.uucF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    def recommend(self, u, item_test, normalized=1):
        """
        Determine all items should be recommended for user u. (uuCF =1)
        or all users who might have interest on item u (uuCF = 0)
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items_rating = []
        for i in item_test:
            rating = self.__pred(u, i)
            recommended_items_rating.append((rating + self.mu[u], i))

        return recommended_items_rating


def mk_test(model, test, n_users):
    total_ans = 0
    for n in range(n_users):
        print(n)
        t = np.where((test[:, 0]) == n)[0]

        item_t = test[t][:, 1]

        r = model.recommend(n, item_t)
        real_r = [i[0] for i in r]

        rating = test[t][:, 2]

        try:
            total_ans += np.abs(rating - real_r).mean()
        except:
            print(n)
            print(rating.shape)
            print(len(real_r))
            exit()
    return total_ans / n_users


if __name__ == '__main__':
    train = np.load('train.npy')
    train[:, :2] = train[:, :2] - 1
    train[:, 2] += 2
    test = np.load('test.npy')
    test[:, :2] = test[:, :2] - 1
    test[:, 2] += 2

    total = np.concatenate((train, test), axis=0)

    model = CF(train, total, k=30)
    model.fit()

    print(mk_test(model, test, np.max(train[:, 0] + 1)))
