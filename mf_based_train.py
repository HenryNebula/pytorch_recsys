import torch
from tqdm import tqdm
import numpy as np
from utility import utils
from Pipeline import Pipeline
import os


def core_training(pipeline:Pipeline):
        epoch_range = range(pipeline.args.epochs)
        device = pipeline.device
        batch_size = pipeline.args.batch_size

        loss_list = [] # loss list for training set

        evaluator = pipeline.evaluator

        for epoch in tqdm(epoch_range):

            loss_ = 0.0
            batch_count = 0

            dataloader = pipeline.train_dataloader if batch_size > 0 else [next(pipeline.train_dataloader)]

            if pipeline.args.loss not in ["BPR", "MG"]:
                test_pred, test_labels = evaluator.get_ground_truth_predictions_with_labels(pipeline.model)
                test_loss = pipeline.model.get_loss(test_pred, test_labels).detach().cpu().numpy()

            else: test_loss = 0.0

            ranking_metrics = evaluator.get_ranking_metrics(pipeline.model)

            for (users, items, labels) in dataloader:
                batch_count += 1

                users = users.view(-1).to(device)
                items = items.view(-1).to(device)
                labels = labels.view(-1).to(device)

                res = pipeline.model([users, items])
                res = res.view(-1)  # [bs*5, 1] -> [bs*5]

                loss = pipeline.model.get_loss(res, labels)

                pipeline.optimizer.zero_grad()
                loss.backward()
                loss_ += loss.detach().cpu().numpy()
                pipeline.optimizer.step()

            # save model when encounter save_interval or use up all epochs
            if (epoch + 1) % pipeline.args.save_interval == 0 or epoch == max(epoch_range):
                pass
                # trainer.diary.save_model(model_dict["model"], model_dict["optimizer"], loss_, epoch)

            if pipeline.config["debug_test"] and (epoch + 1) in pipeline.config["debug_epoch"]:
                pass
                # hr, ndcg = pipeline.test_evaluator.eval_once(pipeline.model, 0, topK=pipeline.args.topK)
                # pipeline.logger.info("TEST_EVAL: epoch = {0}, lr = {1}, HR = {2:.3f}, NDCG = {3:.3f}".format(0, lr_rate, hr[-1], ndcg[-1]))

            print("Epoch [{0}]: Train loss {1:6f}, Test loss {2:6f}; Ranking metrics: {3}"
                  .format(epoch, loss_ / batch_count, test_loss, ranking_metrics))


def init_args(args_):
    print("Task Name: {}".format(args_.task))

    if args_.task == "default":
        """used for debugging"""
        args_.dataset = "ml-1m"
        args_.implicit = False
        args_.method = "MF"
        args_.config = "default_config.json"
        args_.num_neg = 4
        args_.batch_size = -1
        args_.num_factors = 16
        args_.epochs = 50
        args_.lr = 0.5
        args_.reg = 0.0
        args_.loss = "L2"

    paras = "lr_{}-Reg_{}-Embed_{}".format(args_.lr, args_.reg, args_.num_factors)

    args_.log_name = "{}.log".format(paras)
    return args_


if __name__ == "__main__":
    args = utils.parse_args()
    args = init_args(args)
    core_training(Pipeline(args))
