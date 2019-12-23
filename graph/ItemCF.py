from tqdm import tqdm
import os
from noise import ItemSim, ItemSimSparse
import numpy as np
from Trainer import Trainer
from noise.ItemSimPartition import Cluster, reduce_all_partitions
from utility.utils import np_save, parallel_map
from sklearn.preprocessing import normalize
from numpy.random import laplace


class ItemCF(object):

    def __init__(self, trainer:Trainer, args, max_neighbors, bloom_config=None, sl_path='',
                 load=False, async_fit=False):
        self.train_data = trainer.dataset.train_data
        self.trainer = trainer
        self.args = args
        self.eps = args.eps
        self.num_neighbor = args.num_neighbor
        self.sim_file_name = "eps_{}-N_{}.npz".format(self.eps, self.num_neighbor)
        self.max_neighbors = max_neighbors

        # clustering parameters
        self.num_centroid = args.num_centroid
        self.method = args.clustering_method
        self.feature_type = args.feature_type

        self.bloom_config = bloom_config
        self.sl_path = sl_path
        if not os.path.exists(self.sl_path):
            os.makedirs(self.sl_path)

        self.item_sim, self.neighbors = self.fit(preload=load) if not async_fit else (None, None)

    def show_stat(self):
        ui_mtx = self.train_data
        row_sum = ui_mtx.sum(axis=0)
        col_sum = ui_mtx.sum(axis=1)
        nnz = ui_mtx.nnz

        print('average #ratings per item: {}, max #ratings per item: {}'.format(np.mean(row_sum), np.max(row_sum)))
        print('average #ratings per user: {}'.format(np.mean(col_sum)))
        print('nnz ratio: {}'.format(nnz))

    def load_exist_mtx(self, npz_path):
        neighbors = None
        self.trainer.logger.info("Use Existing Sim-Files From {}".format(npz_path))
        assert os.path.exists(npz_path)

        arr = np.load(npz_path)
        sim = arr['sim']
        if 'neighbors' in arr:
            neighbors = arr['neighbors']
        return sim, neighbors

    def save_mtx(self, npz_path, sim, neighbors):
        return 0 # no need to save internal results now
        # data_dict = {"sim": sim, "neighbors": neighbors}
        # np_save(npz_path, data_dict)

    def fit(self, preload, use_sparse=True, reduce=False):
        sl_path = self.sl_path
        npz_path = os.path.join(sl_path, self.sim_file_name)
        file_exist_flag = os.path.exists(npz_path)

        preload = preload and file_exist_flag

        if preload:
            sim, neighbors = self.load_exist_mtx(npz_path)
        else:
            self.trainer.logger.info("Start Re-Computing Sim-Files")
            if self.num_centroid == 1:
                sim = self._get_sim(use_sparse).toarray()
                self.trainer.logger.info("finish generating similarity matrix")
                neighbors = ItemSim.get_neighbors(sim, self.num_neighbor, True)
            else:
                sim_partitions, mappings = self._get_sim_with_partitions(self.num_centroid, self.method,
                                                                         self.feature_type, reduce=reduce)
                if reduce:
                    # already reduces
                    sim = sim_partitions[0].toarray()
                    neighbors = ItemSim.get_neighbors(sim, self.num_neighbor, True)
                else:
                    num_items = self.train_data.shape[1]
                    neighbors = np.zeros((num_items, self.num_neighbor), dtype=int)
                    sim = np.zeros((num_items, num_items))

                    for part in tqdm(sim_partitions):
                        shift, part_sim = part
                        part_sim = part_sim.toarray()
                        part_neighbors = ItemSim.get_neighbors(part_sim, self.num_neighbor, parallel=False)

                        # re-indexing
                        for row in range(part_neighbors.shape[0]):
                            remap_row = mappings[row + int(shift)]
                            row_neighbors = list(map(lambda x: mappings[x], part_neighbors[row, :] + int(shift)))

                            sim[remap_row, row_neighbors] = part_sim[row, part_neighbors[row, :]]
                            neighbors[remap_row, :] = row_neighbors

        self.save_mtx(npz_path, sim, neighbors)
        return sim, neighbors

    def _get_sim(self, use_sparse):
        sim = ItemSim.get_item_sim(self.train_data, self.eps, self.bloom_config, parallel=True) if not use_sparse \
            else ItemSimSparse.fast_item_sim(self.train_data, self.eps,
                                             neighbors=self.max_neighbors, bloom_config=self.bloom_config)
        return sim

    def _get_sim_with_partitions(self, num_centroids, method, feature_type, reduce=False):
        # the index is different from the original ones to improve the speed
        # reindex is done in the neighborhood-finding part

        cluster = Cluster(data_path=self.trainer.dataset.root, num_centroids=num_centroids)
        labels = cluster.fit(method=method, feature_type=feature_type, load=True).squeeze()
        csc_train = self.train_data.tocsc()
        num_items = self.train_data.shape[1]

        mappings = []
        shift_array = [0]
        partitions = []
        for n in tqdm(range(num_centroids)):
            chosen_cols = np.where(labels == n)[0]
            mappings.extend(chosen_cols)
            shift_array.append(len(chosen_cols))

            part_ui = csc_train[:, chosen_cols]
            part_ui = part_ui.tocsr()
            part_sim = ItemSimSparse.fast_item_sim(part_ui, eps=self.eps, neighbors=self.max_neighbors, bloom_config=None)
            partitions.append((int(np.sum(shift_array[:n+1])), part_sim)) # (shift, part_sim)

        if reduce:
            reduced_sim = [reduce_all_partitions(partitions, index_mapping=mappings, num_items=num_items)]
        else:
            reduced_sim = partitions
        return reduced_sim, mappings

    def from_ranking_to_metric(self, evaluator, rankings):
        rank_file = os.path.join(self.sl_path, "eps_{}_rankings.json".format(self.eps))
        hr_list, ndcg_list = evaluator.ranking_to_metric(rankings, topK=self.args.topK, rank_file=rank_file)
        return hr_list, ndcg_list

    def evaluate(self, evaluator):
        ui_gen = (evaluator.get_test_items_by_idx(idx) for idx in range(evaluator.test_num))

        # # check user
        # user_full_set = set(range(self.trainer.dataset.num_users))
        # user_seen_set = set()
        # for user, items in ui_gen:
        #     user_seen_set.add(user)
        # print(user_full_set.difference(user_seen_set))
        # exit(1)

        rankings = [ItemSim.get_ranking(self.train_data[user], self.neighbors, self.item_sim, items)
                    for user, items in tqdm(ui_gen, total=evaluator.test_num)]

        return self.from_ranking_to_metric(evaluator, rankings)

    def compare_noise(self):
        clean_sim_path = "eps_{}-N_{}.npz".format(1000.0, self.num_neighbor)
        clean_sim_path = os.path.join(self.sl_path, clean_sim_path)
        assert os.path.exists(clean_sim_path), "no-noise version doesn't exists!"

        arr = np.load(clean_sim_path)
        clean_sim, clean_neighbors = arr['sim'], arr['neighbors']

        noisy_sim, noisy_neighbors = self.item_sim, self.neighbors
        assert self.item_sim and self.neighbors

        num_items = noisy_sim.shape[0]
        hr_list = np.zeros((num_items, ))
        for row in range(num_items):
            clean_neighbor, noisy_neighbor = set(list(clean_neighbors[row, :])), set(list(noisy_neighbors[row, :]))
            hr = len(clean_neighbor.intersection(noisy_neighbor)) / clean_neighbors.shape[1]
            hr_list[row] = 1 if hr > 0 else 0

        print("HR SUMMARY: Mean {}, Max {}, Median {}".format(np.mean(hr_list), np.max(hr_list), np.median(hr_list)))

        diff = np.mean((noisy_sim - clean_sim) ** 2)
        print("RMSE OF SIM: {}".format(np.sqrt(diff)))


class ItemCF_with_Laplacian(ItemCF):
    # algorithm from KDD'09 paper: used as a baseline

    def __init__(self, trainer: Trainer, args, max_neighbors, sl_path="", load=False, async_fit=False, bloom_config=None):
        super().__init__(trainer, args, max_neighbors, sl_path=sl_path, load=load, async_fit=True)

        self.norm = args.norm
        self.item_sim = self.fit_with_covariance(preload=load) if not async_fit else (None, None)

    def fit_with_covariance(self, preload):
        sl_path = self.sl_path
        npz_path = os.path.join(sl_path, self.sim_file_name)
        file_exist_flag = os.path.exists(npz_path)

        preload = preload and file_exist_flag
        if preload:
            sim, _ = self.load_exist_mtx(npz_path)

        else:
            ui_mtx = self.train_data.tocsr()
            ui_mtx_T = ui_mtx.T

            assert ui_mtx.format == 'csr', "change train_data format to csr first"
            assert self.norm == 'l1' or self.norm == 'l2', "only l1 or l2 normalization is supported"

            ui_mtx = normalize(ui_mtx, norm=self.norm, copy=False)

            sim = ui_mtx_T * ui_mtx

            if self.eps < ItemSim.MAX_EPS:
                noise_amount = 3 if self.norm == 'l1' else np.sqrt(2)
                self.trainer.logger.info("Adding Noise of Amount {}, because {} normalization is used".
                                         format(noise_amount, self.norm))

                beta = noise_amount / self.eps
                sim = sim + laplace(0, beta, size=sim.shape)
                sim = np.array(sim)
            else:
                sim = np.array(sim.todense())

            data_dict = {"sim": sim}
            np_save(npz_path, data_dict)
        return sim

    def evaluate(self, evaluator, real_neighborhood=True):
        ui_gen = (evaluator.get_test_items_by_idx(idx) for idx in range(evaluator.test_num))
        if real_neighborhood:
            input_tups = ((self.train_data[user], self.num_neighbor, self.item_sim, items)
                        for user, items in ui_gen)
            rankings = parallel_map(ItemSim.neighbor_based_wrapper, tqdm(input_tups, total=evaluator.test_num), parallel=True)
        else:
            neighbors = ItemSim.get_neighbors(self.item_sim, self.num_neighbor, True)
            rankings = [ItemSim.get_ranking(self.train_data[user], neighbors, self.item_sim, items)
                        for user, items in tqdm(ui_gen, total=evaluator.test_num)]

        return self.from_ranking_to_metric(evaluator, rankings)


class ItemCF_Noisy(ItemCF):
    def __init__(self, trainer: Trainer, args, max_neighbors, bloom_config=None, sl_path='',
                 load=False, async_fit=False):
        super().__init__(trainer, args, max_neighbors, bloom_config=bloom_config,
                         sl_path=sl_path, load=load, async_fit=async_fit)

    def fit(self, preload, use_sparse=True, reduce=False):
        sl_path = self.sl_path
        npz_path = os.path.join(sl_path, self.sim_file_name)
        file_exist_flag = os.path.exists(npz_path)

        preload = preload and file_exist_flag
        if preload:
            sim, neighbors = self.load_exist_mtx(npz_path)
        else:
            sim = ItemSimSparse.noisy_item_sim(self.train_data, self.eps, neighbors=self.max_neighbors,
                                               optimal=self.bloom_config['optimal']).toarray()
            self.trainer.logger.info("finish generating similarity matrix (Noisy Version) ")
            neighbors = ItemSim.get_neighbors(sim, self.num_neighbor, True)

        data_dict = {"sim": sim, "neighbors": neighbors}
        np_save(npz_path, data_dict)

        return sim, neighbors


class ItemCF_Optimal(ItemCF):

    def fit(self, preload, use_sparse=True, reduce=False):
        sl_path = self.sl_path
        npz_path = os.path.join(sl_path, self.sim_file_name)
        file_exist_flag = os.path.exists(npz_path)

        preload = preload and file_exist_flag

        if preload:
            sim, neighbors = self.load_exist_mtx(npz_path)
        else:
            self.trainer.logger.info("Start Re-Computing Sim-Files (Optimal Model) ")

            sim = ItemSimSparse.fast_item_sim_optimal(self.train_data, self.eps, neighbors=self.max_neighbors,)
            sim = sim.toarray()

            self.trainer.logger.info("finish generating similarity matrix")
            neighbors = ItemSim.get_neighbors(sim, self.num_neighbor, True)

        self.save_mtx(npz_path, sim, neighbors)
        return sim, neighbors























