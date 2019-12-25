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
from utility.fast_rank_topK import fast_topK


class Evaluator(object):
    def __init__(self, test_pos, test_neg):
        self.reset()
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.test_num = len(self.test_pos)

        self.hr_epoch = []
        self.ndcg_epoch = []
        self.epoch_list = []

    def reset(self):
        self.hr_list = []
        self.ndcg_list = []

    def getHitRatio(self, ranklist, gtItem):
        return 1 if gtItem in ranklist else 0
    
    def getNDCG(self, ranklist, gtItem):
        try:
            idx = ranklist.index(gtItem)
            ndcg = math.log(2) / math.log(idx+2)
        except ValueError:
            ndcg = 0
        return ndcg

    @staticmethod
    def HR_by_ranking(ranking):
        # rankings: list containing rankings of all test_pos
        return 1 if (ranking >= 0) else 0

    @staticmethod
    def NDCG_by_ranking(ranking):
        return math.log(2) / math.log(ranking+2) if (ranking >= 0) else 0

    def get_test_items_by_idx(self, idx):
        user = self.test_pos[idx][0]
        item_gt = self.test_pos[idx][1]
        items = self.test_neg[idx]
        # try:
        #     items = next(self.test_neg)
        # except StopIteration:
        #     print("error in test_neg generator")
        #     return
        items.append(item_gt)

        return user, items

    def eval_by_index(self, model, idx):
        gen_t = time.time()
        user, items = self.get_test_items_by_idx(idx)
        pred_t = time.time()

        # Get prediction scores
        users_np = np.full(len(items), user, dtype = 'int32')
        items_np = np.array(items)
        users_tensor = Variable(torch.from_numpy(users_np).to(model.device)).long()
        items_tensor = Variable(torch.from_numpy(items_np).to(model.device)).long()
        predictions_gpu = model.get_predictions([users_tensor, items_tensor])
        predictions = predictions_gpu.cpu().data.numpy()

        rank_t = time.time()

        # Evaluate top rank list

        # map_item_score = {}
        # for i in range(len(items)):
        #     map_item_score[items[i]] = predictions[i]
        # ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)

        predictions = predictions.reshape(-1)

        if len(predictions) >= 1000:
            ranklist = fast_topK(predictions, 1000)

        else:
            argsort = np.argsort(-predictions, axis=0) # index using max prediction
            ranklist = list(map(lambda x: items[x], argsort))

        finish_t = time.time()
        # if idx == 0:
        #     print("gen_time: {}, pred_time: {}, rank_time: {}".format(
        #         pred_t - gen_t, rank_t - pred_t, finish_t - rank_t))

        try:
            ranking = ranklist.index(items[-1]) # ground truth is the last elem
        except ValueError:
            ranking = -1

        del predictions_gpu, predictions, users_tensor, items_tensor
        return ranking

    @staticmethod
    def ranking_to_metric(ranking_list, topK=10, rank_file=""):
        topK_list = [10, 5]
        if topK not in topK_list:
            topK_list.append(topK)
        else:
            topK_list.remove(topK)
            topK_list.append(topK)

        hr_list = []
        ndcg_list = []
        for top in topK_list:
            ranking_array = np.array(ranking_list)
            ranking_array[ranking_array >= top] = -1
            ranking_array = ranking_array.reshape(-1).tolist()
            hr_ = list(map(Evaluator.HR_by_ranking, ranking_array))
            ndcg_ = list(map(Evaluator.NDCG_by_ranking, ranking_array))
            hr, ndcg = np.mean(hr_), np.mean(ndcg_)
            hr_list.append(hr)
            ndcg_list.append(ndcg)

        if rank_file:
            rank_dict = {
                "datetime": str(datetime.datetime.now()),
                "rank_list": ranking_list
            }

            with open(rank_file, 'w') as f:
                f.write(json.dumps(rank_dict, indent=0))

            print("Finishing Writing Rankings to File".format(rank_file))

        return hr_list, ndcg_list

    def eval_once(self, model, epoch, topK=10, rank_file=""):
        self.reset()
        ranking_list = [] # -1 stands for out of range (top1000)

        for idx in range(self.test_num):
            ranking = self.eval_by_index(model, idx)
            ranking_list.append(ranking)

        hr_list, ndcg_list = self.ranking_to_metric(ranking_list, topK, rank_file)

        if epoch != -1:
            self.hr_epoch.append(hr_list[-1])
            self.ndcg_epoch.append(ndcg_list[-1])
            self.epoch_list.append(epoch)
        return hr_list, ndcg_list

    def eval_item_pop(self, popularity, topK=10, rank_file=""):
        self.reset()
        ranking_list = []

        for idx in tqdm(range(self.test_num)):
            item_gt = self.test_pos[idx][1]
            ranking_list.append(popularity[item_gt])

        hr_list, ndcg_list = self.ranking_to_metric(ranking_list, topK, rank_file)
        return hr_list, ndcg_list

    def get_best(self):
        idx = int(np.argmax(self.hr_epoch))
        return [self.hr_epoch[idx], self.ndcg_epoch[idx], self.epoch_list[idx]]

    def get_info(self):
        return [np.array(self.hr_epoch), np.array(self.ndcg_epoch), np.array(self.epoch_list)]

    def get_predictions_and_labels(self, model):
        users, items, labels = [], [], []
        for i in range(len(self.test_pos)):
            # pos samples
            temp_items = [self.test_pos[i][1]]

            #neg samples
            temp_items.extend(self.test_neg[i])
            items.extend(temp_items)

            # fill users and labels lists
            users.extend([self.test_pos[i][0]] * len(temp_items))
            labels.extend([1])
            labels.extend([0] * len(self.test_neg[i]))

        assert (len(users) == len(items) and len(users) == len(labels) and (len(items)) == len(labels)),\
            'user:{}, item:{}, label:{}'.format(len(users), len(items), len(labels))
        users_np = np.array(users)
        items_np = np.array(items)
        labels_np = np.array(labels)
        users_tensor = Variable(torch.from_numpy(users_np).to(model.device)).long()
        items_tensor = Variable(torch.from_numpy(items_np).to(model.device)).long()
        labels_tensor = Variable(torch.from_numpy(labels_np).to(model.device)).float()

        predictions_tensor = model.get_predictions([users_tensor, items_tensor])
        predictions_tensor = predictions_tensor.reshape(-1)

        return predictions_tensor, labels_tensor



