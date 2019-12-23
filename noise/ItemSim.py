import numpy as np
from pybloom_live import BloomFilter
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, coo_matrix
from itertools import product
from tqdm import tqdm
from utility.fast_rank_topK import fast_topK
from utility import utils

MAX_EPS=500


def construct_fake_report_pairs(elem_lists, num_user:int, p:float, bloom_config:dict):
    # col-based data obfuscation
    fake_reports = []
    for elem_list in elem_lists:
        if not bloom_config:
            real_report = np.full((num_user,), fill_value=False, dtype=bool)
            real_report[elem_list] = True
        else:
            capacity = bloom_config["capacity"]
            error_rate = bloom_config["error_rate"]
            real_report = get_bloom_filter_array(elem_list, capacity, error_rate)

        fake_report = random_flipping(real_report, p)
        fake_reports.append(fake_report)

    return fake_reports


def get_bloom_filter_array(elem_list, capacity, error_rate=1e-4):
    bf = BloomFilter(capacity, error_rate)
    [bf.add(x) for x in elem_list]
    bit_arr = bf.bitarray
    mask = [x for x in bit_arr]
    np_bits = np.array(mask)
    if np_bits.dtype != bool:
        np_bits.dtype = bool
    return np_bits


def random_flipping(real_report: np.ndarray, p: float):
    # p: Pr(0 -> 0) = Pr (1 -> 1)
    # q: Pr(0 -> 1) = Pr (1 -> 0) = 1 - p
    # filp binary reports

    if len(real_report.shape) <= 1:
        real_report = real_report.reshape((len(real_report), 1))

    fake_report = real_report.copy()
    row, col = fake_report.shape

    random_noise = np.random.rand(row, col)
    flip_mask = random_noise <= p
    fake_report[flip_mask] = np.logical_not(fake_report[flip_mask])
    return fake_report


def estimate_single_card(fake_report: np.ndarray, p: float, use_bloom:bool):
    length = len(fake_report)
    m0 = fake_report.sum()
    m1 = length - m0
    q = 1 - p

    if not use_bloom:
        estimated_card = length - (q*m0 - p*m1) / (q-p)
    else:
        estimated_card = -length * np.log( (q*m0 - p*m1) / (length * (q-p)) )

    return estimated_card


def estimate_pair_card(fake_report_1: np.ndarray, fake_report_2: np.ndarray, p: float, bloom_slices=0):
    # return union and intersection card

    assert len(fake_report_1) == len(fake_report_2), "length doesn't match between two reports"
    length = len(fake_report_1)

    and_mask = np.logical_and(fake_report_1, fake_report_2)
    union_mask = np.logical_or(fake_report_1, fake_report_2)
    zero_one_mask = np.logical_and(np.logical_not(fake_report_1), fake_report_2)

    m00 = length - union_mask.sum()
    m11 = and_mask.sum()
    m01 = zero_one_mask.sum()
    m10 = length - m00 - m11 - m01
    q = 1 - p

    n00_estimate = ((q**2 * m00 - p*q*(m01+m10) + p**2 * m11) / ((q-p)**2) )
    n11_estimate = ((p**2 * m00 - p*q*(m01+m10) + q**2 * m11) / ((q-p)**2) )

    if not bloom_slices:
        union_card = length - n00_estimate
        intersect_card = n11_estimate
    else:
        card1 = estimate_single_card(fake_report_1, p, True) / bloom_slices
        card2 = estimate_single_card(fake_report_2, p, True) / bloom_slices

        union_card = -length * np.log(n00_estimate / length)
        intersect_card = card1 + card2 - union_card

    return union_card, intersect_card


def jaccard_sim(union_card, intersect_card):
    sim = intersect_card / union_card
    sim = max(sim, 0)
    sim = min(sim, 1)
    return sim


def helper_sim(info_tup):
    if not info_tup:
        return None

    elem_lists, num_user, eps, bloom_config = info_tup

    if eps > MAX_EPS:
        real1: set
        real2: set
        real1, real2 = [set(elem_list) for elem_list in elem_lists]
        union_card= len(real1.union(real2))
        intersect_card = len(real1) + len(real2) - union_card

    else:
        p = 1 / (1 + np.exp(eps))
        bloom_slices = bloom_config['num_slices'] if bloom_config else 0
        fake1, fake2 = construct_fake_report_pairs(elem_lists, num_user, p, bloom_config)
        union_card, intersect_card = estimate_pair_card(fake1, fake2, p, bloom_slices)

    return jaccard_sim(union_card, intersect_card)


def get_item_sim(ui_matrix: csc_matrix, eps, bloom_config:dict=None, parallel=True):

    if ui_matrix.format != 'csc':
        ui_matrix = ui_matrix.tocsc()

    num_user, num_item = ui_matrix.shape
    sim_matrix = np.zeros((num_item, num_item))

    idx_iterator = product(range(num_item), range(num_item))
    elem_iterator = (((ui_matrix[:, i].indices, ui_matrix[:, j].indices), num_user, eps, bloom_config) if i > j else None for i,j in idx_iterator)

    sim_list = utils.parallel_map(helper_sim, tqdm(elem_iterator, total=num_item**2), parallel=parallel, verbose=False)

    for idx, sim in enumerate(sim_list):
        i,j = idx // num_item, idx % num_item
        if i > j:
            sim_matrix[i, j] = sim

    sim_matrix += sim_matrix.T
    sim_matrix += np.eye(num_item)
    return sim_matrix


def fast_rank_wrapper(tup):
    sim_arr, topK = tup
    if isinstance(sim_arr, (csr_matrix, csc_matrix, coo_matrix)):
        sim_arr = sim_arr.toarray().squeeze()

    return fast_topK(sim_arr, topK)


def get_neighbors(sim_matrix: np.ndarray, N=20, parallel=True):
    num_item, _ = sim_matrix.shape
    tup_iterator = map(lambda x: (sim_matrix[x, :], N), range(num_item))

    neighbors_list = utils.parallel_map(fast_rank_wrapper, tqdm(tup_iterator, total=num_item), parallel=parallel)
    neighbors_arr = np.array(neighbors_list)

    return neighbors_arr


def rating_wrapper(tup):
    ratings: csr_matrix
    neighbors, sim_arr, ratings = tup

    sim = np.zeros(ratings.shape[1])
    sim[neighbors] = sim_arr
    sim_sum = np.sum(sim)

    result = ratings * sim / sim_sum if sim_sum != 0 else 0
    return result + random_perturb()


def random_perturb(range_=1e8):
    # increase stability of rankings
    return (np.random.rand()  - 0.5) / range_


def unnormalized_rating_wrapper(tup):
    ratings: csr_matrix
    neighbors, sim_arr, ratings = tup

    sim = np.zeros(ratings.shape[1])
    sim[neighbors] = sim_arr

    return ratings * sim + random_perturb()


def rating_to_ranking(tup_list, item_gt, normalized=False):

    ratings = utils.parallel_map(rating_wrapper, tup_list, parallel=False) if normalized else \
                utils.parallel_map(unnormalized_rating_wrapper, tup_list, parallel=False)

    ratings = np.array(ratings)

    if len(ratings) >= 1000:
        ranklist = fast_topK(ratings, 1000)
        try:
            ranking = ranklist.index(item_gt)
        except ValueError:
            ranking = -1

    else:
        argsort = np.argsort(-ratings, axis=0)  # index using max prediction
        ranking = int(np.where(argsort == len(ratings) - 1)[0])

    return ranking


def get_ranking(rating_arr: csr_matrix, neighbors, sim_mtx, targets):

    if rating_arr.format != 'csr':
        rating_arr = rating_arr.tocsr()

    item_gt = targets[-1]  # ground truth is the last target
    tup_list = [(neighbors[t, :], sim_mtx[t, neighbors[t, :]], rating_arr)
                for t in targets]
    return rating_to_ranking(tup_list, item_gt)


def get_neighborhood_based_ranking(rating_arr: csr_matrix, num_neighbors, sim_mtx, targets):
    if rating_arr.format != 'csr':
        rating_arr = rating_arr.tocsr()

    item_gt = targets[-1]  # ground truth is the last target
    tup_list = []
    for t in targets:

        # index remapping here
        rows, cols = rating_arr.nonzero()
        masked_sim = sim_mtx[t, cols]

        arg_ind = np.argsort(-masked_sim)
        neighbors = cols[arg_ind][:num_neighbors]
        sim_arr = sim_mtx[t, neighbors]
        tup_list.append((neighbors, sim_arr, rating_arr))

    return rating_to_ranking(tup_list, item_gt, normalized=False)


def neighbor_based_wrapper(tup):
    rating_arr, num_neighbors, sim_mtx, targets = tup
    return get_neighborhood_based_ranking(rating_arr, num_neighbors, sim_mtx, targets)






