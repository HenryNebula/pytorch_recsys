import numpy as np
import random
import os
from datetime import datetime
from tqdm import tqdm
from utility.utils import parallel_map
import sys


def txt_to_dict(filename):
    dict_ = {}
    with open(filename) as f:
        for l in f:
            num_list = list(map(int, l.strip().split(" ")))
            user = num_list[0]
            items = num_list[1:]
            if user in dict_:
                dict_[user].extend(items)
            else:
                dict_[user] = items
    for user in dict_:
        dict_[user] = list(set(dict_[user]))
    return dict_


def load_single_file(data_dir="ml-20m-context"):
    file_prefix = "./raw_dataset/{}/".format(data_dir)
    merged_dict_ = txt_to_dict(file_prefix + "data.dat")
    return merged_dict_


def merge_file(data_dir="yelp2018"):
    file_prefix = "./raw_dataset/{}/".format(data_dir)
    train_dict = txt_to_dict(file_prefix + "train.dat")
    test_dict = txt_to_dict(file_prefix + "test.dat")

    # merge and delete repeat ones
    for user in train_dict:
        train_dict[user].extend(test_dict[user])
        train_dict[user] = list(set(train_dict[user]))

    merged_dict_ = train_dict
    return merged_dict_


def re_index(merged_dict):
    def check_compact_set(set_):
        max_val, min_val = np.max(list(set_)), np.min(list(set_))
        return len(set_) == max_val - min_val + 1

    user_set, item_set = set(merged_dict.keys()), set()
    for user in merged_dict:
        for item in merged_dict[user]:
            item_set.add(item)

    if check_compact_set(user_set) and check_compact_set(item_set):
        print("compact sets checked!")
        reindexed_dict = merged_dict
        reindexed_dict["num_user"] = len(user_set)
        reindexed_dict["num_item"] = len(item_set)
    else:
        print("need formatting")
        reindexed_dict = None
        exit(1)
    return reindexed_dict


def generate_neg(pos_sample):
    num_item, num_neg = pos_sample[0], pos_sample[1]
    pos_sample = list(map(lambda x: x[2], pos_sample))
    if num_neg == 1:
        return [(pos_sample + num_item / 3) % num_item]
    else:
        return np.random.choice(list(set(range(num_item)).difference({pos_sample})), size=2 * num_neg, replace=False)


def split(reindexed_dict, dname, num_neg, split_mode="leave_one_out", thresh=0):
    assert type(split_mode) == str
    num_user, num_item = reindexed_dict["num_user"], reindexed_dict["num_item"]
    reindexed_dict.pop("num_user")
    reindexed_dict.pop("num_item")

    def save_file(dict_, fname):
        with open(fname, "w") as f:
            for user in dict_:
                for item in dict_[user]:
                    f.write("{},{}\n".format(user, item))
        print("Finish writing {}".format(fname))

    if not reindexed_dict:
        print("No files created")
        return

    train_dict = {}
    val_dict = {}
    val_neg_dict = {}
    test_dict = {}
    test_neg_dict = {}

    for idx, user in enumerate(tqdm(reindexed_dict)):
        if (idx - 1) % 1000 == 0:
            print("Finishing No. {} users, now: {}".format(idx, datetime.now()))

        items = np.array(reindexed_dict[user])
        np.random.seed(0)
        np.random.shuffle(items)
        length = len(items)

        if split_mode != "leave_one_out":
            # split ratio 6/2/2
            train_dict[user] = items[:int(length * 0.6) + 1]
            val_dict[user] = list(items[int(length * 0.6) + 1: int(length * 0.8) + 1])
            val_neg_dict[user] = parallel_map(generate_neg, val_dict[user])
            test_dict[user] = list(items[int(length * 0.8) + 1:])
        else:
            if length < thresh:
                continue
            if length <= 2:
                train_dict[user] = list(items)
            else:
                items = list(items)
                train_dict[user] = list(items[:-2])

                neg_samples = random.sample(population=range(num_item), k=2*num_neg + len(items))
                neg_samples = list(set(neg_samples).difference(set(items)))[:2*num_neg]
                # neg_samples = np.random.choice(list(set(range(num_item)).difference(set(items))), size=2 * num_neg,
                #                                replace=False)
                val_dict[user] = [items[-2]]
                val_neg_dict[user] = list(neg_samples[:num_neg])
                test_dict[user] = [items[-1]]
                test_neg_dict[user] = list(neg_samples[num_neg:])

    full_dir = "./{}/".format(dname)
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    save_file(train_dict, full_dir + "train.dat")
    save_file(test_dict, full_dir + "test.dat")
    save_file(test_neg_dict, full_dir + "test.negative.dat")
    save_file(val_dict, full_dir + "val.dat")
    save_file(val_neg_dict, full_dir + "val.negative.dat")
    print("finish writing split dataset: {}, Shape: User-{}, Item-{}".format(
        dname, num_user, num_item))


def generate_neg_for_test(dname, item_num):
    def set_difference(arr):
        return "{},{}".format(arr, list(whole_item_set.difference({arr[1]})))

    whole_item_set = set(range(item_num))
    neg_list = []
    with open("./{}/test.dat".format(dname)) as f:
        for line in f:
            arr = map(lambda x: int(x), line.strip().split(","))
            neg_list.append(arr)

    neg_list = map(set_difference, neg_list)

    with open("./{}/test.negative.dat".format(dname), "w") as f:
        f.write("\n".join(neg_list))


if __name__ == "__main__":
    # pool = Pool(5)
    if len(sys.argv) >= 2:
        name_list= [sys.argv[1]]
    else:
        name_list = ["amazon-book", "last-fm", "ml-20m", "yelp2018", "ml-20m-context","ml-10m","app", "yelp"]

    file_path = os.path.dirname(__file__)
    os.chdir(os.path.join(file_path, "../data/"))
    num_neg = 99

    for idx, dataset_name in enumerate([name_list[-1]]):
        print("Start dataset: {}".format(dataset_name))

        if dataset_name not in ["amazon-book", "last-fm", "ml-20m", "yelp2018"]:
            merged_dict = load_single_file(dataset_name)
        else:
            merged_dict = merge_file(dataset_name)

        reindexed_dict = re_index(merged_dict)
        split(reindexed_dict, dataset_name, num_neg, thresh=0)