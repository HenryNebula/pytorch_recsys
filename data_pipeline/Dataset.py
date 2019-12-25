import os, sys
from data_pipeline.AbstractLoader import AbstractLoader
from utility import utils
from scipy.sparse import coo_matrix
import pandas as pd


class ML1M(AbstractLoader):
    def __init__(self, root, num_neg, implicit):
        super().__init__(root, num_neg, implicit)

    def get_ratings(self):
        train_df = utils.load_rating_from_csv(os.path.join(self.root, "train.rating"), sep="\t")
        test_df = utils.load_rating_from_csv(os.path.join(self.root, "test.rating"), sep="\t")

        unique_user = set(train_df.user.unique()).union(test_df.user.unique())
        unique_item = set(train_df.item.unique()).union(test_df.item.unique())
        num_users, num_items = max(unique_user) + 1, max(unique_item) + 1

        ratings_mtx = []
        for df in [train_df, test_df]:
            mtx = coo_matrix((df.rating,
                            (df.user.to_list(), df.item.to_list())),
                           shape=(num_users, num_items))

            if self.implicit:
                self.explicit_to_implicit(mtx)
            ratings_mtx.append(mtx)

        return ratings_mtx

    def get_test_candidates(self):
        test_candidates_df = utils.load_rating_from_csv(os.path.join(self.root, "test.negative"), sep="\t",
                                                        col_names=["user-item"] + ["cand_{}".format(i) for i in range(99)])
        melted_candidates = pd.melt(test_candidates_df,
                                    id_vars="user-item",
                                    value_vars=list(test_candidates_df.columns)[1:],
                                    var_name="colname",
                                    value_name="cand_id")

        candidates_series = melted_candidates.groupby("user-item")["cand_id"].agg(list).reset_index(drop=True)
        return candidates_series.to_list()

    def get_dataset_name(self):
        return "ML-1M"
