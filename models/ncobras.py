from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier


class NCOBRAS:
    def __init__(self, max_queries, epsilon, random_state=None):
        self.max_queries = max_queries
        self.epsilon = epsilon
        self.seed = random_state

        self.labels = None
        self.labels_hist = None
        self.queries_hist = None

    def fit(self, X, y):
        querier = ProbabilisticNoisyQuerier(None, y, self.epsilon, self.max_queries, random_seed=self.seed)
        clusterer = COBRAS(noise_probability=self.epsilon, minimum_approximation_order=3, maximum_approximation_order=6)
        all_clusters, num_queries, runtimes, *_ = clusterer.fit(X, -1, None, querier)
        self.labels = all_clusters[-1]
        self.labels_hist = all_clusters
        self.queries_hist = num_queries
        #runtime_robust = runtimes[-1]
        return self


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import adjusted_rand_score
    X, y = load_iris(return_X_y=True)
    ncobras = NCOBRAS(50, 0.01, random_state=20)
    ncobras.fit(X, y)

    pred = ncobras.labels
    print("ARI:", adjusted_rand_score(y, pred))
    print(ncobras.queries_hist)