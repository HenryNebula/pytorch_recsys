import numpy as np


def fast_topK(metric_array:np.ndarray, topK):
    raw_idx = list(range(len(metric_array)))
    old_index = np.argpartition(metric_array, -topK, axis=0)[-topK:]
    metric_partition = metric_array[old_index]
    part_index = np.argsort(-metric_partition, axis=0)
    ranklist = list(map(lambda x: raw_idx[old_index[x]], part_index))
    return ranklist


if __name__ == "__main__":
    arr = np.array(range(1000000))
    print("finish generating")
    print(fast_topK(arr, 10))
