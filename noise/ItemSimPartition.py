import os
from os import path
import numpy as np
from scipy.sparse import coo_matrix
from utility.utils import np_save
from collections import defaultdict
import json
from sklearn.cluster import KMeans, AgglomerativeClustering
from tqdm import tqdm
import gensim


def load_dict_from_json(file_path):
    assert path.exists(file_path), "{} doesn't exists".format(file_path)
    with open(file_path) as f:
        res = f.read()
    data_dict = json.loads(res)
    return data_dict


def reduce_all_partitions(partitions, index_mapping, num_items):
    # index_mapping: idx: repl-indices, val: raw-indices
    # shift_array: idx: part-id, val: bias of shift value

    all_rows, all_cols, all_data = [], [], []
    for part_id, part in tqdm(enumerate(partitions)):
        shift, part_sim = part
        part_row, part_col = part_sim.nonzero()
        rows = list(map(lambda x: index_mapping[x + shift], part_row))
        cols = list(map(lambda x: index_mapping[x + shift], part_col))
        all_data.extend(part_sim.data.tolist())
        all_rows.extend(rows)
        all_cols.extend(cols)
    reduced_sim = coo_matrix((all_data, (all_rows, all_cols)), shape=(num_items, num_items))

    return reduced_sim


class Cluster(object):

    def __init__(self, data_path, num_centroids):
        self.data_path = data_path
        self.genre_dict = load_dict_from_json(path.join(data_path, "genres_dict.json"))
        attributes = load_dict_from_json(path.join(data_path, "movie_genres.json"))
        self.item_attributes = attributes['movies']

        mappings = load_dict_from_json(path.join(data_path, "mappings.json"))
        self.user_dict = mappings["user_dict"]
        self.item_dict = mappings["item_dict"]  # old_idx -> new_idx
        self.new_idx_to_old = self.get_new_to_old_array()

        self.feature_size = len(self.genre_dict)  # one-hot encoding for default
        self.sample_size = len(self.new_idx_to_old)
        self.num_centroids = num_centroids

    def get_new_to_old_array(self):
        # return: array, index: new_idx, value: old_idx

        num_items = len(self.item_dict)
        new_idx_list = np.zeros(num_items, dtype=int)

        for old_idx in self.item_dict:
            new_idx = self.item_dict[old_idx]
            new_idx_list[new_idx] = old_idx
        return new_idx_list

    def _create_features(self, feature_type="one-hot"):
        if feature_type == "one-hot":
            feature_array = np.zeros((self.sample_size, self.feature_size))
            for idx, item in enumerate(self.new_idx_to_old):
                item = str(item)
                hit_features = self.item_attributes[item]
                feature_array[idx, hit_features] = 1
        elif feature_type == "word2vec":
            model_path = os.path.join(self.data_path, "../downloaded_dataset/"
                                                      "GoogleNews-vectors-negative300-SLIM.bin")
            model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
            print("finish loading word2vec models")
            features = np.zeros((len(self.item_dict), 300))
            item_lists = sorted(self.item_dict.items(), key=lambda x: x[1])
            genres = sorted(self.genre_dict.items(), key=lambda x: x[1])

            for item, id_ in item_lists:
                sub_genres = self.item_attributes[item]
                tags = list(map(lambda x: genres[x][0], sub_genres))
                word_vecs = list(map(lambda x: model.wv[x], tags))
                mean_vec = np.mean(word_vecs, axis=0)
                features[id_, :] = mean_vec
            feature_array = features

        else:
            feature_array = None
            assert NotImplementedError, "Distance metric {} is not implemented".format(feature_type)

        return feature_array

    def fit(self, method="kmeans", feature_type="word2vec", load=False):
        label_path = "clustering_results/{}/{}/k_{}.npy".format(method, feature_type, self.num_centroids)
        label_path = os.path.join(self.data_path, label_path)

        if load and os.path.exists(label_path):
            labels_ = np.load(label_path)

        else:
            feature_array = self._create_features(feature_type)
            if method == "kmeans":
                kmeans = KMeans(n_clusters=self.num_centroids, )
                labels_ = kmeans.fit_predict(feature_array)
            elif method == "hierarchical":
                agg = AgglomerativeClustering(n_clusters=self.num_centroids, affinity="cosine", linkage="complete")
                labels_ = agg.fit_predict(feature_array)
            else:
                labels_ = None
                assert NotImplementedError, "Method {} is not implemented".format(method)

            # save labels
            np_save(label_path, {"labels": labels_})

        return labels_

    @staticmethod
    def partition_info(labels):
        label_info = defaultdict(int)
        for l in labels:
            label_info[l] += 1
        print(label_info)


if __name__ == "__main__":
    file_path = path.dirname(__file__)
    os.chdir(path.join(file_path, "../data/ml-20m-context/"))
    c = Cluster(os.path.abspath(os.curdir), num_centroids=10)
    labels = c.fit(method="kmeans", feature_type="word2vec")
    c.partition_info(labels)
