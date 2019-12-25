import numpy as np
from tqdm import tqdm
import heapq
import math
import torch
from torch.autograd import Variable
import time
import datetime
import os
import json
from utility.ranking import ndcg_at, precision_at
import pandas as pd


class Evaluator(object):
    def __init__(self, test_pos, test_neg):
        self.test_pos = test_pos # (user, item, rating) coo-matrix
        self.test_neg = test_neg # [[tc1_1, tc1_2, ...], [tc2_1, tc2_2, ...], ...]
        self.test_num = len(self.test_pos.data)

        self.ground_truth_item = torch.LongTensor(test_pos.col)
        self.ground_truth_rating = torch.DoubleTensor(test_pos.data)
        self.ground_truth_user = torch.LongTensor(test_pos.row)

        combined_items = list(map(lambda t: t[1] + [t[0]], zip(test_pos.col, test_neg)))
        self.combined_items_tensor = torch.LongTensor(combined_items).view(-1)
        self.combined_users_tensor = torch.LongTensor(test_pos.row).view(self.test_num, -1)\
            .expand(self.test_num, len(combined_items[0])).reshape(-1)

    def get_ground_truth_predictions_with_labels(self, model):
        users_tensor = self.ground_truth_user.to(model.device)
        items_tensor = self.ground_truth_item.to(model.device)
        labels_tensor = self.ground_truth_rating.to(model.device)

        predictions_tensor = model.get_predictions([users_tensor, items_tensor])
        predictions_tensor = predictions_tensor.view(-1)

        return predictions_tensor, labels_tensor

    def get_ranking_metrics(self, model):
        users_tensor = self.combined_users_tensor.to(model.device)
        items_tensor = self.combined_items_tensor.to(model.device)

        predictions_tensor = model.get_predictions([users_tensor, items_tensor])
        predictions_tensor = predictions_tensor.view(-1)

        prediction_df = pd.DataFrame({
            "user": self.combined_users_tensor.numpy(),
            "item": self.combined_items_tensor.numpy(),
            "score": predictions_tensor.detach().cpu().numpy()
        })

        prediction_df.sort_values("score", ascending=False, inplace=True)

        ranked_candidates = prediction_df[["user", "item"]].groupby("user").agg(list).reset_index().rename(columns={"item": "item_list"})
        ground_truth = pd.DataFrame({
            "user": self.ground_truth_user.numpy(),
            "item": self.ground_truth_item.numpy()
        })

        ground_truth = ground_truth.groupby("user").agg(list).reset_index().rename(columns={"item": "ground_truth"})

        result = ranked_candidates.merge(ground_truth, on = "user")

        labels = result["ground_truth"].to_list()
        recommendations = result["item_list"].to_list()

        ranking_metrics = {"NDCG@10": ndcg_at(recommendations, labels)}

        return ranking_metrics







