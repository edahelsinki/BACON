import numba
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

from models.utils import compute_centers_neighbors, convert_probs_to_labels


def simulate_oracle(i, j, y, epsilon):
    # Simulate asking the noisy oracle
    true_same = 1 if y[i] == y[j] else 0
    # Incorporate oracle noise
    is_flipped = np.random.rand() < epsilon
    if is_flipped:
        y_oracle = 1 - true_same # Flip answer
    else:
        y_oracle = true_same

    return y_oracle, is_flipped


def binary_entropy(p):
    """
    Computes the Shannon entropy for a binary distribution.
    Uses np.clip to prevent log2(0) errors.
    """
    # Clip values to avoid log(0) or log(negative) issues
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def compute_eig_matrix(R, epsilon):
    """
    Computes the Expected Information Gain (EIG) matrix.
    Utilizes the Mutual Information symmetry trick: 
    EIG(i,j) = H_bin(Predictive Prob) - H_bin(Oracle Error Rate)
    """
    # P is our predictive probability that the oracle answers "1" (Same Cluster)
    P = (1.0 - epsilon) * R + epsilon * (1.0 - R)
    
    eig_mat = binary_entropy(P) - binary_entropy(epsilon)
    np.fill_diagonal(eig_mat, 0.0) # Self-edges yield no information
    
    return eig_mat


@numba.njit
def inverse_transform_sample(probs, u):
    """Numba-friendly categorical sampler."""
    # u = rng.rand()
    cum_prob = 0.0
    for i in range(len(probs)):
        cum_prob += probs[i]
        if u < cum_prob:
            return i
    return len(probs) - 1

@numba.njit
def uf_find(parent, x):
    """
    Find with path compression.
    """
    root = x
    while parent[root] != root:
        root = parent[root]

    # Path compression
    while parent[x] != x:
        px = parent[x]
        parent[x] = root
        x = px

    return root


@numba.njit
def uf_union(parent, size, a, b):
    """
    Union by size.
    """
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)

    if ra == rb:
        return

    # Attach smaller tree under larger tree
    if size[ra] < size[rb]:
        ra, rb = rb, ra

    parent[rb] = ra
    size[ra] += size[rb]

@numba.njit
def extract_cluster_labels_from_links(c):
    """
    Extract connected-component labels from a link vector c.

    Parameters
    ----------
    c : 1D int64 array of shape (K,)
        Link vector. c[i] is the node that node i links to.
        Must satisfy 0 <= c[i] < K.

    Returns
    -------
    labels : 1D int64 array of shape (K,)
        Compact cluster labels in {0, 1, ..., n_clusters-1}.
    parent : 1D int64 array of shape (K,)
        Final union-find parent array (roots can be inspected if needed).
    """
    K = c.shape[0]

    # Union-Find initialization
    parent = np.empty(K, dtype=np.int64)
    size = np.empty(K, dtype=np.int64)

    for i in range(K):
        parent[i] = i
        size[i] = 1

    # Union each node with its link target
    for i in range(K):
        j = c[i]
        # Assuming c is valid: 0 <= j < K
        if j != i:
            uf_union(parent, size, i, j)

    # Ensure all nodes point directly to roots
    for i in range(K):
        parent[i] = uf_find(parent, i)

    return parent

    # Compress root IDs into compact labels 0..n_clusters-1
    # labels = np.empty(K, dtype=np.int64)
    # root_to_label = -np.ones(K, dtype=np.int64)

    # next_label = 0
    # for i in range(K):
    #     r = parent[i]
    #     lbl = root_to_label[r]
    #     if lbl == -1:
    #         lbl = next_label
    #         root_to_label[r] = lbl
    #         next_label += 1
    #     labels[i] = lbl

    # return labels, parent


def reindex_labels(labels_old):
    # reindex labels 0, ..., k-1
    uniques = np.unique(labels_old)
    labels = -np.ones_like(labels_old)
    for i,  l in enumerate(uniques):
        labels[labels_old == l] = i
    return labels


def select_query(R, medoid_idx, y, qi, qj, labels, true_epsilon, epsilon_sampled, rng, weighted=False, uncert=False):
    k = medoid_idx.shape[0]
    # uses R_micro between a medoid and a point as the representativeness
    # note that epsilon_sampled does not influence the selection
    EIG = compute_eig_matrix(R, epsilon_sampled)

    if weighted:
        micro_cluster_sizes = np.bincount(labels)
        # size_matrix = micro_cluster_sizes[:, None] + micro_cluster_sizes[None, :]
        size_matrix = np.minimum(micro_cluster_sizes[:, np.newaxis], micro_cluster_sizes)
        EIG = EIG * size_matrix
    EIG_masked = EIG.copy()
    EIG_masked[np.tril_indices(k)] = -np.inf # Lower triangle + diagonal ignored
    if len(qi) > 0:
        for i, j in zip(qi, qj):
            EIG_masked[i, j] = -np.inf
            EIG_masked[j, i] = -np.inf
    
    # Select the pair with the highest Expected Information Gain
    if uncert:
        # print("uncert")
        U = np.abs(R - 0.5)
        U[np.tril_indices(k)] = np.inf 
        if len(qi) > 0:
            for i, j in zip(qi, qj):
                EIG_masked[i, j] = np.inf
                EIG_masked[j, i] = np.inf
        max_indices = np.argwhere(U == np.min(U))
    else:    
        max_indices = np.argwhere(EIG_masked == np.max(EIG_masked))
    
    # Tie-breaker logic: pick randomly if multiple edges have identical max EIG
    if len(max_indices) > 1:
        chosen_idx = rng.choice(len(max_indices))
        best_idx = max_indices[chosen_idx]
    else:
        best_idx = max_indices[0]
        
    opt_i, opt_j = best_idx[0], best_idx[1]
    
    # Ensure i < j for consistency
    if opt_i > opt_j:
        opt_i, opt_j = opt_j, opt_i

    medoid_i = medoid_idx[opt_i]
    medoid_j = medoid_idx[opt_j]
    y_oracle, is_noisy = simulate_oracle(medoid_i, medoid_j, y, true_epsilon)
    return opt_i, opt_j, y_oracle, EIG, is_noisy


def uniform_sample_upper_triagle(arr, rng=None):
    """
    Randomly sample one finite index (i, j) from the upper triangle.
    The diagonal is excluded.
    """
    rng = np.random.default_rng(rng)

    # k=1 excludes the diagonal; use k=0 to include it
    rows, cols = np.triu_indices_from(arr, k=1)

    # Keep values that are not -inf
    valid = ~np.isneginf(arr[rows, cols])

    rows = rows[valid]
    cols = cols[valid]

    if len(rows) == 0:
        raise ValueError("No valid indices exist in the upper triangle.")

    selected = rng.integers(len(rows))
    return rows[selected], cols[selected]


def select_random_query(medoid_idx, y, qi, qj, epsilon, rng):

    k = medoid_idx.shape[0]
    # uses R_micro between a medoid and a point as the representativeness
    
    sampled_masked = np.zeros((k, k))
    sampled_masked[np.tril_indices(k)] = -np.inf # Lower triangle + diagonal ignored
    if len(qi>0):
        for i, j in zip(qi, qj):
            sampled_masked[i, j] = -np.inf
            sampled_masked[j, i] = -np.inf

    opt_i, opt_j = uniform_sample_upper_triagle(sampled_masked, rng)

    # Ensure i < j for consistency
    if opt_i > opt_j:
        opt_i, opt_j = opt_j, opt_i

    medoid_i = medoid_idx[opt_i]
    medoid_j = medoid_idx[opt_j]
    y_oracle, is_noisy = simulate_oracle(medoid_i, medoid_j, y, epsilon)
    
    return opt_i, opt_j, y_oracle, is_noisy


@numba.jit
def mh_alpha_ddcrp_knn(rng, alpha_old, knn, D, lambda2, z, a, b, n, step_size=0.2):
    log_new_alpha = rng.normal(np.log(alpha_old), step_size)
    alpha_new = np.exp(log_new_alpha)

    n_self_links = 0
    for i in range(n):
        if z[i] == i:
            n_self_links +=1
    
    mult = a + n_self_links - 1 

    decay = np.zeros(n)
    for i in range(n):
        for j in knn[i, 1:]:
            decay[i] += np.exp(-D[i,j]/(2*lambda2))

    new_lp = mult*np.log(alpha_new) - alpha_new/b - np.sum(np.log(decay + alpha_new))
    old_lp = mult*np.log(alpha_old) - alpha_old/b - np.sum(np.log(decay + alpha_old))

    log_accept_prob = new_lp - old_lp + log_new_alpha - np.log(alpha_old)
    u = rng.random()
    if np.log(u) < log_accept_prob:
        # accept proposal
        pass
    else:
        alpha_new = alpha_old
    return alpha_new

@numba.njit
def inv_gamma_log_density(lambda2, a, b):
    return (-a - 1)*np.log(lambda2) - b/lambda2

@numba.jit
def mh_lambda2_ddcrp_knn(rng, lambda2_old, knn, D, z, n, alpha, a=1.0, b=1.0, step_size=0.2):
    log_lambda2_new = rng.normal(np.log(lambda2_old), step_size)
    lambda2_new = np.exp(log_lambda2_new)

    decay_sum_old = np.zeros(n)
    decay_sum_new = np.zeros(n)
    for i in range(n):
        for j in knn[i, 1:]:
            decay_sum_old[i] += np.exp(-D[i, j] / (2*lambda2_old))
            decay_sum_new[i] += np.exp(-D[i, j] / (2*lambda2_new))

    lp_old = inv_gamma_log_density(lambda2_old, a, b)
    lp_new = inv_gamma_log_density(lambda2_new, a, b)

    # omega term
    lp_old -= np.sum(np.log(decay_sum_old + alpha))
    lp_new -= np.sum(np.log(decay_sum_new + alpha))

    # numerator decay terms only for non-self links
    for i in range(n):
        if i == z[i]: 
            continue
        lp_old += -D[i, z[i]] / (2*lambda2_old)
        lp_new += -D[i, z[i]] / (2*lambda2_new)

    log_accept_prob = lp_new - lp_old + log_lambda2_new - np.log(lambda2_old)

    if np.log(rng.random()) < log_accept_prob:
        return lambda2_new
    return lambda2_old


@numba.njit
def _log_accept(x, a, b):
    # log f(x) - [log f(1/2) + g (x - 1/2)], with f the unnormalized Beta(a, b)
    # density and g = 2(a - b) its log-slope at 1/2. Always <= 0 when a, b >= 1.
    return ((a - 1.0) * np.log(2.0 * x)
            + (b - 1.0) * np.log(2.0 * (1.0 - x))
            - 2.0 * (a - b) * (x - 0.5))


@numba.njit
def sample_epsilon(m, n_q, a3, b3, rng, n_plain=1000):
    """Exact draw of eps ~ Beta(a3 + m, b3 + n_q - m) truncated to [0, 1/2).

    Requires a3, b3 >= 1 (log-concave density). Always returns a float.
    """
    a = a3 + m
    b = b3 + n_q - m

    # Case 1: a <= b. At least half the Beta mass is below 1/2, so plain
    # rejection accepts with probability >= 1/2.
    if a <= b:
        while True:
            x = rng.beta(a, b)
            if x < 0.5:
                return x

    # Case 2: a > b, mode above 1/2. Try plain rejection a few times first
    # (cheap when the mode is only slightly above 1/2) ...
    for _ in range(n_plain):
        x = rng.beta(a, b)
        if x < 0.5:
            return x

    # ... then use the tangent of the log-density at 1/2 as an envelope:
    # 1/2 - x ~ Exponential(rate g) truncated to [0, 1/2].
    g = 2.0 * (a - b)
    tail = -np.expm1(-0.5 * g)          # 1 - exp(-g/2), accurate for small g
    while True:
        e = -np.log1p(-rng.random() * tail) / g
        x = 0.5 - e
        if x >= 0.5 or x <= 0.0:
            continue
        if np.log(rng.random()) < _log_accept(x, a, b):
            return x


@numba.njit
def mcmc_sampler(D, knn, qi, qj, qy, alpha_init, lambda2_init, epsilon, gamma_a, gamma_b, inv_gamma_a, inv_gamma_b, beta_a, beta_b, num_samples=2000, burn_in=500, init_z=None, rng=None, sample_eps=False):
    n = D.shape[0]
    k = knn.shape[1]

    total_iter = num_samples + burn_in
    log_eps = np.log(epsilon)
    log_eps_inv = np.log(1.0 - epsilon)
    
    qn = len(qi)

    lambda2 = lambda2_init
    alpha = alpha_init

    alpha_trace = np.zeros(total_iter)
    lambda2_trace = np.zeros(total_iter)
    eps_trace = np.zeros(total_iter)
    R_mcmc = np.zeros((n,n))
    all_samples = np.empty((num_samples, n), dtype=np.int64)

    if init_z is None:
        current_z = np.arange(n)
    else:
        current_z = init_z.copy()

    for iter_idx in range(total_iter):
        
        # Iterate over points in a random order
        order = rng.permutation(n)

        for i in order:
            
            proposal_links = knn[i] #np.arange(n)
            log_probs = np.zeros(k)

            # eval proposals for i
            for idx_j, j in enumerate(proposal_links):

                # prior
                if i == j:
                    lp = np.log(alpha)
                else:
                    lp = -D[i, j] / (2 * lambda2)
                
                # temporarily set the link to the proposal j
                current_z[i] = j

                # extract labels from the links
                labels = extract_cluster_labels_from_links(current_z)
                
                # oracle likelihood
                for q in range(qn):
                    queryi = qi[q]
                    queryj = qj[q]
                    queryy = qy[q]

                    is_same = 1 if labels[queryi] == labels[queryj] else 0
                    lp += log_eps_inv if is_same == queryy else log_eps

                log_probs[idx_j] = lp

            # --- E. SOFTMAX & SAMPLE ---
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)
            chosen_idx = inverse_transform_sample(probs, rng.random())
            current_z[i] = proposal_links[chosen_idx]

        # metropolis hastings sample new alpha and lambda
        alpha = mh_alpha_ddcrp_knn(rng, alpha, knn, D, lambda2, current_z, gamma_a, gamma_b, n, step_size=1.0)
        alpha_trace[iter_idx] = alpha

        lambda2 = mh_lambda2_ddcrp_knn(rng, lambda2, knn, D, current_z, n, alpha, inv_gamma_a, inv_gamma_b, step_size=1.0)
        lambda2_trace[iter_idx] = lambda2
        
        # extract and log co-clustering indicators
        labels = extract_cluster_labels_from_links(current_z)

        if sample_eps:
            m = 0
            for q in range(qn):
                queryi = qi[q]
                queryj = qj[q]
                queryy = qy[q]
                is_same = labels[queryi] == labels[queryj]
                if queryy == 1 and not is_same:
                    m += 1
                if queryy == 0 and is_same:
                    m += 1

            epsilon = sample_epsilon(m, qn, a3=beta_a, b3=beta_b, rng=rng)
            log_eps = np.log(epsilon)
            log_eps_inv = np.log(1.0 - epsilon)
        eps_trace[iter_idx] = epsilon

        if iter_idx >= burn_in:
            # This creates an n x n boolean matrix where entry (i,j) is True if they share a cluster
            adjacency = (labels[:, None] == labels[None, :])
            R_mcmc += adjacency
            all_samples[iter_idx-burn_in] = labels.copy()
                
    R_mcmc /= num_samples
    np.fill_diagonal(R_mcmc, 1.0)
    return R_mcmc, all_samples, alpha_trace, lambda2_trace, eps_trace, current_z


class BACON:
    def __init__(self, budget=10, n_samples=2000, n_burnin=200, epsilon=0.1, epsilon_init=0.1, knn=6, gamma_a=1.0, gamma_b=0.5, inv_gamma_a=2.0, inv_gamma_b=1.0, beta_a=1.0, beta_b=5.0, n_super_instances=50, sample_eps=False, weight_query=False, cluster_point_estimate="binder", mode="kmeans", random_queries=False, uncertainty=False, verbose=False, random_state=None):
        self.budget = budget
        self.knn = knn
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        self.inv_gamma_a = inv_gamma_a
        self.inv_gamma_b = inv_gamma_b
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.mode = mode
        self.n_super_instances = n_super_instances
        self.cluster_point_estimate = cluster_point_estimate
        self.weight_query = weight_query
        self.random_state = random_state
        self.verbose = verbose
        self.uncertainty = uncertainty
        self.random_queries = random_queries
        self.sample_eps = sample_eps
        self.epsilon_init = epsilon_init

        self.labels_history = None # point estimates
        self.labels_sample_history = None # full posterior samples
        self.super_instance_labeling = None
        self.super_instance_medoids = None
        self.epsilon_samples = None
        self.n_oracle_errors = None

        if epsilon < 0:
            raise TypeError("Epsilon should be larger or equal to zero.")
        else:
            self.epsilon = epsilon

        self.queries = {'i': [], 'j': [], "y": []}

    def fit(self, X, y):
        self.rng = np.random.default_rng(self.random_state)

        n_queries = 0
        queries = {'i': [], 'j': [], "y": []}

        labels_history = []
        labels_sample_history = []
        epsilon_history = []

        self.n_oracle_errors = 0

        N = X.shape[0]

        if self.mode == "full":
            # run model on the full data
            K = N
        elif self.mode == "kmeans":
            # compress by super-istances
            K = self.n_super_instances
        else:
            raise ValueError(f"Unknown mode={self.mode}.")

        n, d = X.shape

        # initialize estimated params
        alpha_init = 1.0
        lambda2_init = 0.5

        # init variable for the last clustering labels (for initializing the chain)
        init_z = None

        epsilon = self.epsilon_init

        # The active learning sequential loop
        for step in range(self.budget + 1):

            if self.verbose:
                print(f"--- Iteration {step} ---")

            qi = np.array(queries["i"], dtype=int)
            qj = np.array(queries["j"], dtype=int)
            qy = np.array(queries["y"], dtype=int)

            if step == 0:
                if self.verbose:
                    print("Finding super-istances")
                if K == N:
                    labels_super = np.arange(N)
                else:
                    km = KMeans(n_clusters=K, random_state=self.random_state)
                    km.fit(X)
                    labels_super = km.labels_
                    self.super_instance_labeling = labels_super

                unique_labels = np.unique(labels_super)
                current_k = len(unique_labels)
                _, D_centers, knn_idx, medoid_ids = compute_centers_neighbors(X, labels_super, unique_labels, current_k, self.knn, d)
                self.super_instance_medoids = medoid_ids

            # sample macro
            R, cluster_samples, alpha_trace, lambda2_trace, eps_trace, init_z = (
                mcmc_sampler(
                    D_centers,
                    knn_idx,
                    qi,
                    qj,
                    qy,
                    alpha_init,
                    lambda2_init,
                    epsilon,
                    self.gamma_a,
                    self.gamma_b,
                    self.inv_gamma_a,
                    self.inv_gamma_b,
                    self.beta_a,
                    self.beta_b,
                    num_samples=self.n_samples,
                    burn_in=self.n_burnin,
                    init_z=init_z,
                    rng=self.rng,
                    sample_eps=self.sample_eps,
                )
            )
            alpha_init = alpha_trace[-1]
            lambda2_init = lambda2_trace[-1]
            epsilon = eps_trace[-1]
            # point estimates
            labels_point = convert_probs_to_labels(cluster_samples, R, self.cluster_point_estimate)
            labels_point = reindex_labels(labels_point)
            # labels_ = labels_point[labels_super]

            if self.sample_eps:
                # co-clustering probabilities when epsilon is latent
                # C: list of label vectors, E: list of eps draws, same length S
                E = eps_trace[self.n_burnin:]
                R = np.mean([e + (1 - 2*e) * (c[:, None] == c[None, :]) for c, e in zip(cluster_samples, E)], axis=0)

            if self.random_queries:
                # random querying
                opt_i, opt_j, y_oracle, is_noisy = select_random_query(medoid_ids, y, qi, qj, self.epsilon, self.rng)
            else:
                # select query by EIG
                opt_i, opt_j, y_oracle, EIG_macro, is_noisy = select_query(R, medoid_ids, y, qi, qj, labels_super, self.epsilon, epsilon, self.rng, weighted=self.weight_query, uncert=self.uncertainty)
            n_queries +=1

            if is_noisy:
                self.n_oracle_errors +=1

            if self.verbose:
                print(f"Selected Query: Points ({opt_i}, {opt_j}) R-prob: {R[opt_i, opt_j]}")
                print(f"Oracle answers: {y_oracle}")
                print()

            if self.budget >= n_queries:

                # Append to our history
                queries["i"].append(opt_i)
                queries["j"].append(opt_j)
                queries["y"].append(y_oracle)

            labels_history.append(labels_point)
            epsilon_history.append(eps_trace[self.n_burnin:])
            labels_sample_history.append(cluster_samples)

        # self.R = R[labels_super[:, None], labels_super[None, :]]
        self.labels_history = labels_history
        self.labels_sample_history = labels_sample_history
        self.epsilon_history = epsilon_history
        self.queries = queries


def main():

    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    X, true_labels = make_moons(n_samples=80, noise=0.05, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = BACON(budget=11, epsilon=0.1, epsilon_init=0.1, n_samples=2000, n_burnin=200, n_super_instances=30, knn=6, mode="kmeans", random_queries=False, verbose=True, sample_eps=True, random_state=42)
    model.fit(X, true_labels)

    print("Oracle errors:", model.n_oracle_errors)

    for i, macro_labels in enumerate(model.labels_history):
        labels = macro_labels[model.super_instance_labeling]
        mean_eps = np.mean(model.epsilon_history[i])
        print(f"i={i}: ARI={adjusted_rand_score(true_labels, labels)}, EPS={mean_eps}")

if __name__ == "__main__":
    main()
