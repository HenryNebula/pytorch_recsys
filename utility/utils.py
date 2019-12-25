import argparse
from data_pipeline.Dataset import ML1M
from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument("--method", type=str, default="GMF",
                        help="choose the method.")

    parser.add_argument("-n", "--task", type=str, default="default",
                        help="task name.")

    parser.add_argument("--config", type=str, default="config.json",
                        help="config file.")

    dataset_choices = ["ml-1m"]
    parser.add_argument("--dataset", type=str, choices=dataset_choices,
                        help="Choose a dataset from {}.".format(dataset_choices))

    parser.add_argument("--implicit", type=bool,
                        help="Whether transform the original rating to implicit form")

    parser.add_argument("--num_neg", type=int, default=4,
                        help="Number of negative instances to pair with a positive instance.")

    parser.add_argument("--topK", type=int, default=10,
                        help="topK for evaluation.")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs.")

    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size.")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")

    parser.add_argument("--loss", type=str, default="default",
                        help="Choose a loss type. [BCE, L2, BPR]")

    parser.add_argument("--reg", type=float, default=0.0,
                        help="L2 regularization of weight decay")

    parser.add_argument("--num_factors", type=int, default=8,
                        help="Embedding size.")

    parser.add_argument("--multiplier", type=float, default=1.0,
                        help="multiplier over the similarity.")

    parser.add_argument("--bias", type=float, default=0.0,
                        help="bias over the similarity.")

    parser.add_argument("--target", type=float, default=1.0,
                        help="soft target within (0.5, 1.0).")

    parser.add_argument("--bpr_margin", type=float, default=0.0,
                        help="margin for bpr. default=0.0")

    parser.add_argument("--mg_margin", type=float, default=0.1,
                        help="margin for bpr. default=0.0")

    # don"t normalize user and item embedding by default
    parser.add_argument("--norm_user", action="store_true",
                        help="Whether to normalize user embedding.")

    parser.add_argument("--norm_item", action="store_true",
                        help="Whether to normalize item embedding.")

    parser.add_argument("--use_user_bias", action="store_true",
                        help="Whether to user bias.")

    parser.add_argument("--use_item_bias", action="store_true",
                        help="Whether to item bias.")

    parser.add_argument("--square_dist", action="store_true",
                        help="Whether to use the squared distance in CML.")

    parser.add_argument("--output", type=str, default="debug_output",
                        help="path to model outputs")

    parser.add_argument("--log_name", type=str, default="debug",
                        help="name of log file")

    parser.add_argument("--save_interval", type=int, default=10000,
                        help="interval for saving the trained model and info.")

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu id")

    args = parser.parse_args()

    return args


def load_dataset(config, dataset, num_neg, implicit):
    if dataset == "ml-1m":
        return ML1M(config["root_ml1m"], num_neg, implicit)
    else:
        raise (Exception("Dataset {0} not found in config file".format(dataset)))


def load_rating_from_csv(fname, sep=",", col_names=("user", "item", "rating", "ts")):

    return pd.read_csv(fname,
                       sep=sep,
                       header=None,
                       names=col_names)
