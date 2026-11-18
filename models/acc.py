import itertools
import math
import os
import pickle
import sys
import time
from hashlib import sha256

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import softmax as scipy_softmax
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    v_measure_score,
)


class QueryStrategy:
    def __init__(self, ac):
        self.ac = ac
        self.info_matrix = None

    def select_batch(self, acq_fn, batch_size):
        if acq_fn == "unif":
            self.info_matrix = np.random.rand(self.ac.N, self.ac.N)
        elif acq_fn == "freq":
            self.info_matrix = -self.ac.feedback_freq
        elif acq_fn == "uncert":
            self.info_matrix = -np.abs(self.ac.pairwise_similarities)
        elif acq_fn == "maxmin":
            if np.random.rand() < self.ac.eps:
                self.info_matrix = -self.ac.feedback_freq
            else:
                self.info_matrix = self.compute_maxmin()
        elif acq_fn == "maxexp":
            if np.random.rand() < self.ac.eps:
                self.info_matrix = -self.ac.feedback_freq
            else:
                self.info_matrix = self.compute_maxexp()
        else:
            raise ValueError(f"Invalid acquisition function: {acq_fn}")

        return self.select_edges(batch_size, self.info_matrix)
           
    def select_edges(self, batch_size, I, use_tau=True):
        I_local = np.copy(I)
        tri_rows, tri_cols = np.tril_indices(n=I_local.shape[0], k=-1)
        informative_scores = I_local[tri_rows, tri_cols]
        if use_tau:
            fq_flat = self.ac.feedback_freq[tri_rows, tri_cols]
            informative_scores[fq_flat > self.ac.tau] = -np.inf

        random_tie_breaker = np.random.rand(len(informative_scores))
        sorted_indices = np.lexsort((random_tie_breaker, -informative_scores))
        top_B_indices = sorted_indices[:batch_size]

        top_row_indices = tri_rows[top_B_indices]
        top_col_indices = tri_cols[top_B_indices]
        top_pairs = np.stack((top_row_indices, top_col_indices), axis=-1)
        return top_pairs
        
    def compute_maxmin(self):
        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
        if self.ac.num_maxmin_edges == 0:
            return self.custom_informativeness
        #max_edges = int(self.ac.num_maxmin_edges * self.ac.N)
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]

        if self.ac.num_maxmin_edges == -1:
            max_edges = self.ac.N
        else:
            num_violations = len(inds)
            max_edges = int(num_violations * self.ac.num_maxmin_edges)

        inds = np.random.choice(inds, np.min([max_edges, len(inds)]), replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]
        N = self.ac.N
        for i, j in zip(a, b):
            if self.violates_clustering(i, j):
                for k in range(N):
                    if k == i or k == j:
                        continue
                    sim_ij = self.ac.pairwise_similarities[i, j]
                    sim_ik = self.ac.pairwise_similarities[i, k]
                    sim_jk = self.ac.pairwise_similarities[j, k]
                    if self.triangle_is_bad(i, j, k):
                        sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                        smallest_sim = self.random_sort(np.array(sims))[0]
                        if smallest_sim == 0:
                            self.custom_informativeness[i, j] = np.abs(sim_ij)
                            self.custom_informativeness[j, i] = np.abs(sim_ij)
                        if smallest_sim == 1:
                            self.custom_informativeness[i, k] = np.abs(sim_ik)
                            self.custom_informativeness[k, i] = np.abs(sim_ik)
                        if smallest_sim == 2:
                            self.custom_informativeness[j, k] = np.abs(sim_jk)
                            self.custom_informativeness[k, j] = np.abs(sim_jk)
        return self.custom_informativeness

    def compute_maxexp(self):
        #max_edges = int(self.ac.num_maxmin_edges * self.ac.N)
        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
        if self.ac.num_maxmin_edges == 0:
            return self.custom_informativeness
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]

        if self.ac.num_maxmin_edges == -1:
            max_edges = self.ac.N
        else:
            num_violations = len(inds)
            max_edges = int(num_violations * self.ac.num_maxmin_edges)
        
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]
        inds = np.random.choice(inds, np.min([max_edges, len(inds)]), replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]
        N = self.ac.N
        for i, j in zip(a, b):
            if self.violates_clustering(i, j):
                for k in range(N):
                    if k == i or k == j:
                        continue
                    sim_ij = self.ac.pairwise_similarities[i, j]
                    sim_ik = self.ac.pairwise_similarities[i, k]
                    sim_jk = self.ac.pairwise_similarities[j, k]

                    if self.triangle_is_bad(i, j, k):
                        expected_loss = self.min_triple_cost(i, j, k, beta=self.ac.beta) 
                        sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                        smallest_sim = self.random_sort(np.array(sims))[0]
                        if smallest_sim == 0:
                            self.custom_informativeness[i, j] = np.maximum(expected_loss, self.custom_informativeness[i, j])
                            self.custom_informativeness[j, i] = np.maximum(expected_loss, self.custom_informativeness[j, i])
                        if smallest_sim == 1:
                            self.custom_informativeness[i, k] = np.maximum(expected_loss, self.custom_informativeness[i, k])
                            self.custom_informativeness[k, i] = np.maximum(expected_loss, self.custom_informativeness[k, i])
                        if smallest_sim == 2:
                            self.custom_informativeness[j, k] = np.maximum(expected_loss, self.custom_informativeness[j, k])
                            self.custom_informativeness[k, j] = np.maximum(expected_loss, self.custom_informativeness[k, j])
        return self.custom_informativeness

    def min_triple_cost(self, i, j, k, beta=1):
        sim_ij = self.ac.pairwise_similarities[i, j]
        sim_ik = self.ac.pairwise_similarities[i, k]
        sim_jk = self.ac.pairwise_similarities[j, k]

        # (i, j, k)
        c1 = 0
        if sim_ij < 0:
            c1 += np.abs(sim_ij)

        if sim_ik < 0:
            c1 += np.abs(sim_ik)

        if sim_jk < 0:
            c1 += np.abs(sim_jk)

        # (i, k), (j)
        c2 = 0
        if sim_ij >= 0:
            c2 += np.abs(sim_ij)

        if sim_ik < 0:
            c2 += np.abs(sim_ik)

        if sim_jk >= 0:
            c2 += np.abs(sim_jk)

        # (i, j), (k)
        c3 = 0
        if sim_ij < 0:
            c3 += np.abs(sim_ij)

        if sim_ik >= 0:
            c3 += np.abs(sim_ik)

        if sim_jk >= 0:
            c3 += np.abs(sim_jk)

        # (k, j), (i)
        c4 = 0
        if sim_ij >= 0:
            c4 += np.abs(sim_ij)

        if sim_ik >= 0:
            c4 += np.abs(sim_ik)

        if sim_jk < 0:
            c4 += np.abs(sim_jk)

        # (i), (j), (k)
        c5 = 0
        if sim_ij >= 0:
            c5 += np.abs(sim_ij)

        if sim_ik >= 0:
            c5 += np.abs(sim_ik)

        if sim_jk >= 0:
            c5 += np.abs(sim_jk)

        costs = np.array([c1, c2, c3, c4, c5])
        probs = scipy_softmax(-beta*costs)
        expected_loss = np.sum(costs*probs)

        return expected_loss

    def triangle_is_bad(self, i, j, k):
        sim_ij = self.ac.pairwise_similarities[i, j]
        sim_ik = self.ac.pairwise_similarities[i, k]
        sim_jk = self.ac.pairwise_similarities[j, k]
        num_pos = 0

        if sim_ij > 0:
            num_pos += 1

        if sim_ik > 0:
            num_pos += 1

        if sim_jk > 0:
            num_pos += 1

        return num_pos == 2

    def in_same_cluster(self, o1, o2):
        return self.ac.clustering_solution[o1] == self.ac.clustering_solution[o2]

    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.ac.pairwise_similarities[o1, o2] > 0) or \
                (self.in_same_cluster(o1, o2) and self.ac.pairwise_similarities[o1, o2] < 0)
    
    def random_sort(self, arr, ascending=True):
        if not ascending:
            arr = -arr
        b = np.random.random(arr.size)
        return np.lexsort((b, arr))

class ExperimentData:
    def __init__(self, Y, repeat_id, **kwargs):
        self.__dict__.update(kwargs)

        self.repeat_id = repeat_id
        self.experiment_params = []
        self.name = ""
        self.dataset_full_name = ""
        for key, value in kwargs.items():
            if key[0] == "_":
                continue
            if key.split("_")[0] == "dataset":
                self.dataset_full_name += str(value) + "_"
            self.name += str(value) + "_"
            self.experiment_params.append(key)
        self.name = self.name[:-1]
        self.name_repeat = self.name + "_" + str(repeat_id)
        self.hashed_name = sha256(self.name.encode("utf-8")).hexdigest()
        self.dataset_full_name = self.dataset_full_name[:-1]

        # ground truth clustering solution
        self.Y = Y

        # metrics
        self.rand = []
        self.ami = []
        self.v_measure = []
        self.num_clusters = []
        self.num_violations = []
        self.time = []
        self.time_select_batch = []
        self.time_update_clustering = []
        self.num_repeat_queries = []

    def is_equal_no_repeat(self, a2):
        params = set(self.experiment_params + a2.experiment_params)
        for param in params:
            if not hasattr(self, param) or not hasattr(a2, param):
                continue
            if getattr(self, param) != getattr(a2, param):
                return False
        return True

    def is_equal(self, a2):
        return self.is_equal_no_repeat(a2) and self.repeat_id == a2.repeat_id

class ExperimentReader:
    def __init__(self, metrics=["rand"]):
        self.metrics = metrics
        self.Y = None

    def get_metric_data(self, exp):
        data = {}
        for metric in self.metrics:
            for e in exp:
                if metric not in data:
                    data[metric] = [getattr(e, metric)]
                else:
                    data[metric].append(getattr(e, metric))
        return data

    def extend_list(self, data, max_size):
        if len(data) < max_size:
            data.extend([data[-1]] * (max_size - len(data)))
        #if len(data) > max_size:                
            #raise ValueError("Length of data larger than max size")
        return data

    def read_all_data(self, folder):
        data = pd.DataFrame()
        for path, subdirs, files in os.walk(folder):
            for name in files:
                if name == "completed_experiments.txt":
                    continue
                with open(path + "/" + name, 'rb') as handle:
                    try:
                        exp = pickle.load(handle)
                    except EOFError:
                        print("EOF error in read data")
                        continue
                    dat = exp[0].__dict__
                    if self.Y is None:
                        self.Y = exp[0].Y
                    wanted_keys = exp[0].experiment_params
                    sub_dat = dict((k, dat[k]) for k in wanted_keys if k in dat)
                    metric_data = self.get_metric_data(exp)
                    for metric in self.metrics:
                        met_data = metric_data[metric]
                        max_size = 0
                        for mt in met_data:
                            max_size = max(max_size, len(mt))
                        for i in range(len(met_data)):
                            met_data[i] = self.extend_list(met_data[i], max_size)
                        sub_dat[metric] = np.array(met_data)
                    sub_dat = pd.DataFrame([sub_dat])
                    data = pd.concat([data, sub_dat], ignore_index=True)
        if len(data) == 0:
            print("no data found in folder")
            return None
        return data

    def flatten_dataframe(self, df, non_data_column_names, data_column_names):
        # List to store each DataFrame
        dataframes = []
        for data_column in data_column_names:
            # Temporarily drop non-required columns
            df_temp = df.drop(columns=[col for col in data_column_names if col != data_column])
            
            # Expand each array into its own DataFrame and merge back with the original DataFrame
            for i in range(df_temp.shape[0]):
                array = df_temp.loc[i, data_column]
                df_expanded = pd.DataFrame(array).unstack().reset_index()
                df_expanded.columns = ['x', 'x2', 'y']
                for col in non_data_column_names:
                    df_expanded[col] = [df_temp.loc[i, col]] * len(df_expanded)
                df_expanded['metric'] = data_column
                dataframes.append(df_expanded)

        # Concatenate all dataframes
        df_flattened = pd.concat(dataframes, ignore_index=True)
        return df_flattened
    
    def extend_list_all(self, data, max_size):
        new_dat = data.tolist()
        for i in range(len(data)):
            new_dat[i] = self.extend_list(new_dat[i], max_size)
        return np.array(new_dat)

    def extend_dataframe(self, df, col, index):
        df[col] = df[col].apply(lambda x: self.extend_list_all(x, index))
        df[col] = df[col].apply(lambda x: x[:, :index+1])
        #df[col] = df[col][:, :index+1]
        return df
    
    def filter_dataframe(self, df, conditions):
        mask = pd.Series(True, index=df.index)
        for column, values in conditions.items():
            if type(values) != list:
                values = [values]
            mask = mask & df[column].isin(values)
        return df[mask]

def max_correlation(S, K, num_iterations):
    N = np.size(S, 0)

    best_objective = -np.finfo(np.float64).max 
    best_solution = np.zeros((N,), dtype=int)
        
    N_iter = 30

    for itr in range(num_iterations):
        current_solution = np.random.randint(0, K, size=N)

        # calc total cost
        current_objective = 0.0
        for i in range(N):
            for j in range(i):
                if current_solution[i] == current_solution[j]:
                    current_objective += S[i, j]

        old_objective = current_objective - 1

        while True:
            if (current_objective - old_objective) <= np.finfo(np.float64).eps or N_iter <= 0:
                break
            N_iter -= 1
            old_objective = current_objective
            order = np.random.permutation(N)
            
            for i in range(N):
                cur_ind = order[i]
                temp_objects = np.zeros((K,), dtype=float)
                
                for j in range(N):
                    if j != cur_ind:
                        temp_objects[current_solution[j]] += S[cur_ind, j]

                sep_Obj = temp_objects[current_solution[cur_ind]]
                temp_objects[current_solution[cur_ind]] = current_objective
                
                for k in range(K):
                    if k != current_solution[cur_ind]:
                        temp_objects[k] = current_objective - sep_Obj + temp_objects[k]
                                             
                temp_max = np.argmax(temp_objects)
                current_solution[cur_ind] = temp_max
                current_objective = temp_objects[temp_max]

        if itr == 0 or current_objective > best_objective:
            best_solution = np.array(current_solution)
            best_objective = current_objective
            
    return best_solution, best_objective

def max_correlation_dynamic_K(S, K, num_iterations):
    N = np.size(S, 0)
    #print("SIZE: ", N)
    K_dyn = K
    
    #print("NUM CLUSTERS: ", K_dyn)

    best_objective = -sys.float_info.max 
    best_solution = np.zeros((N,), dtype=int)

    N_iter = 30

    for itr in range(num_iterations):
        current_solution = np.zeros((N,), dtype=int) 
        
        for i in range(N):
            current_solution[i] = np.random.randint(0, K_dyn)
            
        # to gaurantee non-empty clusters
        temp_indices = np.random.choice(np.arange(0, N), K_dyn, replace=False)
        for k in range(K_dyn):
            current_solution[temp_indices[k]] = k

        current_objective = 0.0
        for k in range(K_dyn):
            inds = np.where(current_solution == k)[0]
            lower_triangle_indices = np.tril_indices(len(inds), -1) 
            current_objective += np.sum(S[np.ix_(inds, inds)][lower_triangle_indices])

        old_objective = current_objective - 1.0

        for _ in range(N_iter): 
            if (current_objective-old_objective) <= sys.float_info.epsilon:
                break
            old_objective = current_objective
            indices = np.arange(0, N)
            np.random.shuffle(indices)
            for i in indices:
                temp_objects = np.zeros(K_dyn)
                for k in range(K_dyn):
                    inds = np.where(current_solution == k)[0]
                    inds = inds[inds != i]
                    temp_objects[k] = np.sum(S[i, inds])

                if np.max(temp_objects) < 0.0:
                    # cerate a new cluster
                        current_objective = current_objective - temp_objects[current_solution[i]]
                        current_solution[i] = K_dyn
                        K_dyn = K_dyn + 1
                else:
                    sep_Obj = temp_objects[current_solution[i]]
                    temp_objects[current_solution[i]] = current_objective
                    for k in range(K_dyn):
                        if k != current_solution[i]:
                            temp_objects[k] = current_objective - sep_Obj + temp_objects[k]

                    temp_old_cluster = current_solution[i]
                    temp_max = np.argmax(temp_objects)
                    current_solution[i] = temp_max
                    current_objective = temp_objects[temp_max]
                        
                    # check the empy cluster, shinke if necessary
                    K_dyn_temp = len(np.unique(current_solution))
                    if K_dyn_temp < K_dyn:
                        for j in range(N):
                            if current_solution[j] > temp_old_cluster:
                                current_solution[j] = current_solution[j] - 1
                        K_dyn = K_dyn - 1

        if itr == 0 or current_objective > best_objective:
            best_solution = np.array(current_solution)
            best_objective = current_objective
            
    return best_solution, best_objective

class ActiveClustering:
    def __init__(self, X, Y, repeat_id, initial_clustering_solution=None, **kwargs):
        self.__dict__.update(kwargs)

        self.X, self.Y = X, Y
        self.repeat_id = repeat_id
        self.initial_clustering_solution = initial_clustering_solution
        self.ac_data = ExperimentData(Y, repeat_id, **kwargs)
        self.qs = QueryStrategy(self)

        self.clustering = None
        self.clustering_solution = None
        self.N = len(self.Y)
        self.n_edges = (self.N*(self.N-1))/2

        if self.batch_size < 1:
            self.batch_size = math.ceil(self.n_edges * self.batch_size)

        np.random.seed(self.repeat_id+self.seed+317421)

        if self.tau == -1:
            self.tau = np.inf
        else:
            self.tau = self.tau

        #if self.acq_fn in ["info_gain_object", "info_gain_pairs"]:
            #self.info_gain_pair_mode = "entropy"

    def run_AL_procedure(self, max_queries):
        clustering_history = []
        self.start_time = time.time()
        self.initialize_ac_procedure()
        self.store_experiment_data(initial=True)
        perfect_rand_count = 0
        early_stopping_count = 0
        old_rand = adjusted_rand_score(self.Y, self.clustering_solution)
        clustering_history.append(self.clustering_solution)
        total_queries = 0
        if self.acq_fn in ["QECC"]:
            stopping_criteria = self.batch_size * 250
        if self.acq_fn in ["nCOBRAS", "COBRAS"]:
            stopping_criteria = self.batch_size * 200
        else:
            stopping_criteria = self.n_edges
        stopping_criteria = self.n_edges
        early_stopping = (self.n_edges/self.batch_size)*0.55
        ii = 1
        while total_queries < stopping_criteria: 
            self.start = time.time()
            ii += 1
            if self.acq_fn == "QECC":
                self.clustering = self.QECC_heur(self.batch_size * ii)
                self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
                self.clustering_solution = self.clustering_solution_from_clustering(self.clustering)
            else:
                start_select_batch = time.time()
                self.edges = self.qs.select_batch(
                    acq_fn=self.acq_fn,
                    batch_size=self.batch_size
                )
                self.time_select_batch = time.time() - start_select_batch
                if len(self.edges) != self.batch_size:
                    raise ValueError(f"Num queried {len(self.edges[0])} not equal to query size {self.batch_size}!!")
                
                self.num_repeat_queries = 0
                for ind1, ind2 in self.edges:
                    self.update_similarity(ind1, ind2)

                time_clustering = time.time()
                self.update_clustering() 
                self.time_clustering = time.time() - time_clustering
                self.compute_violations()

            self.store_experiment_data()
            total_queries += self.batch_size
            clustering_history.append(self.clustering_solution)
            self.total_time_elapsed = time.time() - self.start_time

            #num_hours = self.total_time_elapsed / 3600
            #if num_hours > max_hours:
            #    break

            if self._verbose:
                print("iteration: ", ii)
                print("prop_queried: ", total_queries/self.n_edges)
                print("feedback_freq max: ", np.max(self.feedback_freq))
                print("rand score: ", adjusted_rand_score(self.Y, self.clustering_solution))
                print("time: ", time.time()-self.start)
                if self.acq_fn not in ["QECC", "COBRAS", "nCOBRAS"]:
                    print("num queries: ", len(self.edges))
                    print("TIME SELECT BATCH: ", self.time_select_batch)
                    print("TIME CLUSTERING: ", self.time_clustering)
                print("num clusters: ", self.num_clusters)
                #print("-----------------")
                
            current_rand = adjusted_rand_score(self.Y, self.clustering_solution)

            #if current_rand > 0.98:
                #break

            if total_queries >= max_queries:
                break
            
            if current_rand == old_rand:
                early_stopping_count += 1
            else:
                early_stopping_count = 0
            
            if current_rand >= 1:
                perfect_rand_count += 1
            else:
                perfect_rand_count = 0
            
            old_rand = current_rand

            # if early_stopping_count > early_stopping or perfect_rand_count > 2:
            #     break
        return clustering_history #self.ac_data

    def QECC_heur(self, num_queries):
        nodes = set(np.arange(0, self.N).tolist())
        vertices = nodes.copy()
        query_budget = num_queries
        clusters = []
        while len(vertices) > 1 and query_budget > (len(vertices) - 1):
            unique_pairs = list(itertools.combinations(vertices, 2))
            np.random.shuffle(unique_pairs)
            for (ind1, ind2) in unique_pairs:
                query = self.get_similarity(ind1, ind2)
                query_budget -= 1
                if query >= 0:
                    break

            cluster = {ind1, ind2}
            for nd in vertices - {ind1, ind2}:
                if self.get_similarity(ind2, nd) > 0:
                    cluster.add(nd)
            clusters.append(list(cluster))
            query_budget = query_budget - len(vertices) + 2
            vertices = vertices - cluster
        for nd in vertices:
            clusters.append([nd])
        return clusters

    def get_similarity(self, ind1, ind2):
        if np.random.rand() <= self.noise_level:
            return np.random.uniform(-1.0, 1.0)
        else:
            return self.ground_truth_pairwise_similarities[ind1, ind2]

    def store_experiment_data(self, initial=False):
        self.ac_data.rand.append(adjusted_rand_score(self.Y, self.clustering_solution))
        self.ac_data.ami.append(adjusted_mutual_info_score(self.Y, self.clustering_solution))
        self.ac_data.v_measure.append(v_measure_score(self.Y, self.clustering_solution))
        self.ac_data.num_clusters.append(self.num_clusters)
                
        if initial:
            self.ac_data.Y = self.Y
            self.ac_data.time.append(0.0)
            self.ac_data.time_select_batch.append(0.0)
            self.ac_data.time_update_clustering.append(0.0)
            self.ac_data.num_repeat_queries.append(0)
            self.ac_data.num_violations.append(0)
        else:
            time_now = time.time() 
            self.ac_data.time.append(time_now - self.start)
            if self.acq_fn not in ["QECC", "COBRAS", "nCOBRAS"]:
                self.ac_data.time_select_batch.append(self.time_select_batch)
                self.ac_data.time_update_clustering.append(self.time_clustering)
                count_non_zero_lower = np.count_nonzero(np.tril(self.violations, k=-1))
                self.ac_data.num_violations.append(count_non_zero_lower)
                self.ac_data.num_repeat_queries.append(self.num_repeat_queries)

    def in_same_cluster(self, o1, o2):
        return self.clustering_solution[o1] == self.clustering_solution[o2]

    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] >= 0) or \
                (self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] < 0)

    def initialize_ac_procedure(self):
        self.feedback_freq = np.zeros((self.N, self.N)) + 1
        self.construct_ground_truth_sim_matrix()
        self.construct_initial_sim_matrix()
        self.violations = np.zeros((self.N, self.N))
        self.compute_violations()

    def compute_violations(self):
        for i in range(self.N):
            for j in range(i):
                if self.violates_clustering(i, j):
                    self.violations[i, j] = np.abs(self.pairwise_similarities[i, j])
                    self.violations[j, i] = np.abs(self.pairwise_similarities[j, i])
                else:
                    self.violations[i, j] = 0
                    self.violations[j, i] = 0

    def construct_ground_truth_sim_matrix(self):
        num_clusters = np.max(self.Y) + 1
        self.num_clusters_ground_truth = num_clusters
        self.ground_truth_clustering = [[] for _ in range(num_clusters)]

        for i in range(len(self.Y)):
            self.ground_truth_clustering[self.Y[i]].append(i)

        self.ground_truth_pairwise_similarities = -1*np.ones((self.N, self.N))
        #self.ground_truth_pairwise_similarities = np.zeros((self.N, self.N))
        for cind in self.ground_truth_clustering:
            self.ground_truth_pairwise_similarities[np.ix_(cind, cind)] = 1

    def clustering_from_clustering_solution(self, clustering_solution):
        num_clusters = np.max(clustering_solution) + 1
        clustering = [[] for _ in range(num_clusters)]
        for i in range(len(clustering_solution)):
            clustering[clustering_solution[i]].append(i)
        return clustering, num_clusters

    def find_cluster(self, d, cluster_indices):
        i = 0
        for ds in cluster_indices:
            if d in ds:
                return i
            i += 1
        print("FIND CLUSTER UNREACHABLE")
        return -1

    def clustering_solution_from_clustering(self, clustering):
        clustering_solution = np.zeros(self.N, dtype=np.uint32)
        for u in range(self.N):
            clustering_solution[u] = self.find_cluster(u, clustering)
        return clustering_solution

    def sim_matrix_from_clustering(self, clustering):
        pairwise_similarities = -self.sim_init*np.ones((self.N, self.N))
        for cind in clustering:
            pairwise_similarities[np.ix_(cind, cind)] = self.sim_init
        return pairwise_similarities

    def construct_initial_sim_matrix(self):
        if self.acq_fn in ["QECC", "COBRAS", "nCOBRAS"]:
            self.sim_init_type = "random_clustering"

        if self.sim_init_type == "zeros":
            self.pairwise_similarities = np.zeros((self.N, self.N))
            self.num_clusters = 1
        elif self.sim_init_type == "uniform_random":
            self.pairwise_similarities = np.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.num_clusters = 1
        elif self.sim_init_type == "kmeans":
            self.clustering_solution = np.array(self.initial_clustering_solution)
            self.clustering, self.num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
        elif self.sim_init_type == "random_clustering":
            K = self.K_init
            self.clustering_solution = np.random.randint(0, K, size=self.N)
            self.clustering, self.num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
        else:
            raise ValueError("Invalid sim init type in construct_initial_sim_matrix(...)")
        np.fill_diagonal(self.pairwise_similarities, 0.0)

        pairs = None
        if self.warm_start > 0:
            total_flips = int(self.n_edges * self.warm_start)
            #specific_seed = self.repeat_id + self.seed + 42
            specific_seed = self.seed + 42
            state = np.random.get_state()
            np.random.seed(specific_seed)
            pairs = self.qs.select_batch("unif", total_flips)
            for ind1, ind2 in pairs:
                query = self.ground_truth_pairwise_similarities[ind1, ind2]
                self.pairwise_similarities[ind1, ind2] = query
                self.pairwise_similarities[ind2, ind1] = query
                self.feedback_freq[ind1, ind2] += 1
                self.feedback_freq[ind2, ind1] += 1
        
        self.update_clustering() 

        if self.warm_start > 0:
            np.random.set_state(state)

    def update_clustering(self):
        self.clustering_solution, _ = max_correlation_dynamic_K(self.pairwise_similarities, self.num_clusters, 5)
        self.clustering = self.clustering_from_clustering_solution(self.clustering_solution)[0]
        self.num_clusters = len(self.clustering)
    
    def update_similarity(self, ind1, ind2):
        if self.feedback_freq[ind1, ind2] > 1:
            self.num_repeat_queries += 1

        self.feedback_freq[ind1, ind2] += 1
        self.feedback_freq[ind2, ind1] += 1

        feedback_frequency = self.feedback_freq[ind1, ind2]
        similarity = self.pairwise_similarities[ind1, ind2]

        if np.random.rand() <= self.noise_level:
            #if self.regular_noise:
                #query = np.random.uniform(-1.0, 1.0)
            #else:
            noisy_val = np.random.uniform(0.15, 0.5)
            if np.random.uniform(-1.0, 1.0) > 0:
                query = -noisy_val
            else:
                query = noisy_val
        else:
            query = self.ground_truth_pairwise_similarities[ind1, ind2]

        self.pairwise_similarities[ind1, ind2] = ((feedback_frequency-1) * similarity + query)/(feedback_frequency)
        self.pairwise_similarities[ind2, ind1] = ((feedback_frequency-1) * similarity + query)/(feedback_frequency)


def find_medoids(X, centroids):

    k = centroids.shape[0]
    medoid_idx = np.zeros(k, dtype=int)
    for i in range(k):
        dists_sq = cdist(X, centroids[i].reshape(1, -1), metric="sqeuclidean").flatten()
        medoid_idx[i] = np.argmin(dists_sq)
    return medoid_idx

class ACC:
    """Wrapper for active correlation clustering
    https://github.com/Linusaronsson/Active-Correlation-Clustering/tree/main
    """
    def __init__(self, noise_level=0.1, max_queries=100, aquisition_function="maxexp", init_type="random_clustering", n_super_instances=None, random_state=None):
        self.noise_level = noise_level
        self.max_queries = max_queries
        self.aquisition_function = aquisition_function # "maxmin" or "maxexp"
        self.init_type = init_type # "random_clustering" or "kmeans"
        self.n_super_instances = n_super_instances
        self.random_state = random_state
        self.history = None
    
    def fit(self, X, y):
        km = KMeans(n_clusters=self.n_super_instances, random_state=self.random_state)
        km.fit(X)
        labels_super = km.labels_
        medoid_ids = find_medoids(X, km.cluster_centers_)
        X_sub = X[medoid_ids]
        y_sub = y[medoid_ids]

        if self.random_state is None:
            seed = np.random.choice(99999999)
        else:
            seed = self.random_state

        labels_init = None
        if self.init_type == "kmeans":
            labels_init = KMeans(n_clusters=10).fit_predict(X_sub)
        
        ac = ActiveClustering(X_sub, y_sub, 0, labels_init, batch_size=1, tau=5, acq_fn=self.aquisition_function, sim_init_type=self.init_type, warm_start=0, eps=0.3, beta=1, num_maxmin_edges=-1, noise_level=self.noise_level, _verbose=False, K_init=10, sim_init=0.1, seed=seed)
        hist = ac.run_AL_procedure(self.max_queries)
        self.history = [labels_compressed[labels_super] for labels_compressed in hist]
        return self


if __name__ == "__main__":
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.05, random_state=42)
    acc = ACC(n_super_instances=50, max_queries=50, init_type="kmeans", random_state=42)
    acc.fit(X, y)
    for i, labels in enumerate(acc.history):
        print(f"iter {i} ARI {adjusted_rand_score(y, labels)}")