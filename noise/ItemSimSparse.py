from datetime import datetime

from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
from tqdm import tqdm

MAX_EPS=5e2


def sparse_random_flipping(real_mtx: csr_matrix, p: float) -> csr_matrix:
    # random flipping for symmetric scenario
    # real_mtx: (user, item)
    # fake_mtx: (user, item)

    num_user, num_item = real_mtx.shape
    raw_true = np.full(shape=(num_user, num_item), fill_value=False, dtype=bool)
    rows, cols = real_mtx.nonzero()
    raw_true[rows, cols] = True

    random_noise = np.random.rand(num_user, num_item)
    mask = random_noise <= p
    fake_true = np.logical_xor(raw_true, mask)
    rows, cols = np.where(fake_true)

    fake_mtx = lil_matrix((num_user, num_item))
    fake_mtx[rows, cols] = 1
    return fake_mtx.tocsr()


def sparse_conditional_flipping(real_mtx: csr_matrix, p: float, q: float) -> csr_matrix:
    # random flipping for asymmetric scenario
    # real_mtx: (user, item)
    # fake_mtx: (user, item)

    num_user, num_item = real_mtx.shape
    raw_true = np.full(shape=(num_user, num_item), fill_value=False, dtype=bool)
    rows, cols = real_mtx.nonzero()
    raw_true[rows, cols] = True

    # different strategy for original 0s and 1s
    random_noise = np.random.rand(num_user, num_item)

    # 1 -> 1
    mask = random_noise <= p
    true_true = np.logical_and(raw_true, mask)

    # 0 -> 1
    mask = random_noise <= q
    false_true = np.logical_and(np.logical_not(raw_true), mask)

    fake_true = np.logical_or(true_true, false_true)
    rows, cols = np.where(fake_true)

    fake_mtx = lil_matrix((num_user, num_item))
    fake_mtx[rows, cols] = 1
    return fake_mtx.tocsr()


def flipping_helper(real_mtx: csr_matrix, p: float, q: float) -> csr_matrix:
    if p != 1 - q:
        return sparse_random_flipping(real_mtx, p)
    else:
        return sparse_conditional_flipping(real_mtx, p, q)


def get_intersection_mtx(iu_mtx: csr_matrix, neighbors=-1) -> csr_matrix:
    # get neighboring mtx based on the metric of intersection

    if neighbors == -1:
        sparse_intersect = iu_mtx * iu_mtx.T
    else:
        from implicit.nearest_neighbours import ItemItemRecommender as item_item
        it = item_item(K=neighbors)
        it.fit(iu_mtx)
        sim = it.similarity.tolil()
        rows, cols = sim.nonzero()
        sim[cols, rows] = sim[rows, cols]
        sparse_intersect = sim.tocsr()

    return sparse_intersect


def get_union_mtx(sparse_intersect: csr_matrix, item_cards) -> csr_matrix:
    # sparse_intersect: (item, item)
    # item_cards: (item, ) <- sum ui_mtx by rows

    sparse_union = csr_matrix(sparse_intersect.shape)
    sparse_union = sparse_union.tocoo()

    coo_intersect = sparse_intersect.tocoo()
    rows, cols = coo_intersect.row, coo_intersect.col

    print("Constructing union data...")
    item_item_pair = ((i,j) for i, j in zip(rows, cols))
    union_data = [item_cards[elem[0]] + item_cards[elem[1]] - coo_intersect.data[idx]
                  for idx, elem in tqdm(enumerate(item_item_pair), total=len(rows))]

    sparse_union.row, sparse_union.col = rows, cols
    sparse_union.data = np.array(union_data).reshape(coo_intersect.data.shape)

    return sparse_union.tocsr()


def get_inv_mtx(mtx: csr_matrix):
    mtx_ = mtx.tocoo()
    mtx_.data = 1 / mtx_.data
    return mtx_.tocsr()


def get_sim_mtx(sparse_intersect: csr_matrix, sparse_union: csr_matrix):
    # Jaccard sim for now
    jaccard_sim: csr_matrix
    jaccard_sim = sparse_intersect.multiply(get_inv_mtx(sparse_union))
    jaccard_sim.data[jaccard_sim.data > 1] = 1
    jaccard_sim.data[jaccard_sim.data < 0] = 0
    jaccard_sim.eliminate_zeros()
    return jaccard_sim


def fast_item_sim(ui_matrix: csr_matrix, eps, neighbors, bloom_config:dict=None):

    iu_mtx = ui_matrix.T
    num_user, num_item = ui_matrix.shape

    if eps >= MAX_EPS:
        # no-noise baseline
        sparse_intersect = get_intersection_mtx(iu_mtx, neighbors=neighbors)
        print(sparse_intersect.shape, sparse_intersect.data.shape)
        sparse_union = get_union_mtx(sparse_intersect, np.array(iu_mtx.sum(1)))
        sparse_sim = get_sim_mtx(sparse_intersect, sparse_union)
        print("{} Finish sim calculation".format(str(datetime.now())))

    else:
        if bloom_config:
            capacity = bloom_config["capacity"]
            error_rate = bloom_config["error_rate"]
            assert NotImplementedError
            sparse_sim = None
        else:
            # add noise
            p = 1 / (1 + np.exp(eps))
            q = 1 - p

            flipped = flipping_helper(ui_matrix, p, q)
            flipped = flipped.T  # now flipped is (item, user)

            # original decoding methods
            print("{} Finish flipping".format(str(datetime.now())))

            length_ones = np.ones((num_item, num_item)) * num_user

            m11 = get_intersection_mtx(flipped, neighbors=neighbors)
            # print(m11.shape)
            _m00 = get_union_mtx(m11, np.array(flipped.sum(1)))
            # print(_m00.shape)
            m00 = length_ones - _m00

            m01_m10 = _m00 - m11

            print("{} Finish raw stats computing".format(str(datetime.now())))

            tmp_value = p * q * m01_m10

            n00_estimate = ((q ** 2 * m00 - tmp_value + p ** 2 * m11) / ((q - p) ** 2))
            n11_estimate = ((p ** 2 * m00 - tmp_value + q ** 2 * m11) / ((q - p) ** 2))

            print("{} Finish estimation".format(str(datetime.now())))

            union_card = length_ones - n00_estimate
            intersect_card = n11_estimate

            sparse_sim = get_sim_mtx(csr_matrix(intersect_card), csr_matrix(union_card))

            print("{} Finish sim-mtx computation".format(str(datetime.now())))

    return sparse_sim


def noisy_item_sim(ui_matrix: csr_matrix, eps, neighbors, optimal=False):

    if not optimal:
        p = 1 / (1 + np.exp(eps)) # 0 -> 1 or 1 -> 0
        flipped = sparse_random_flipping(ui_matrix, p)
        flipped = flipped.T  # now flipped is (item, user)
        print("Not Optimal Flipping")
    else:
        p = 1.0
        q = p / (np.exp(eps))
        flipped = sparse_conditional_flipping(ui_matrix, p, q)
        flipped = flipped.T  # now flipped is (item, user)
        print("Optimal Flipping")

    sparse_intersect = get_intersection_mtx(flipped, neighbors=neighbors)
    sparse_union = get_union_mtx(sparse_intersect, np.array(flipped.sum(1)))
    sparse_sim = get_sim_mtx(sparse_intersect, sparse_union)

    opt_str = "With" if optimal else "Without"
    print("{} Finish sim calculation (Noisy Baseline, {} Optimal Settings) ".format(str(datetime.now()), opt_str))
    return sparse_sim


def fast_item_sim_optimal(ui_matrix: csr_matrix, eps, neighbors,):
    # optimal settings for random flipping: p = 1, q = 1/exp(eps)
    num_user, num_item = ui_matrix.shape

    # add noise
    p = 1.0 # 1 -> 1
    q = p / (np.exp(eps))  # 0 -> 1

    flipped = flipping_helper(ui_matrix, p, q)
    flipped = flipped.T  # now flipped is (item, user)

    # original decoding methods
    print("{} Finish flipping".format(str(datetime.now())))

    # 2*4 elements of the first and last rows of the inversion matrix
    det = p**5 * (-1 + q) + q**3 + p**4 * (7 - 9*q + 2*q**2) + \
        p**3 * (-13 + 18*q - 7*q**2 + q**3) + p**2*(7 - 7*q + 2*q**2 + q**3) - p*(1 + 2*q**3)

    k11 = p ** 2 * (1 + p ** 2 + p * (-3 + q)) / det
    k12 = -(-1 + p) * p * (p - q) / det
    k13 = (-1 + p) * p * (1 + p ** 2 + p * (-3 + q)) / det
    k14 = (-1 + p) ** 2 * (p - q) / det

    k21 = (p ** 3 * (-1 + q) + q ** 3 - p ** 2 * (-2 + q + q ** 2) + p * (-1 + 2 * q ** 2 - 2 * q ** 3)) / det
    k22 = ((-1 + p) * (-1 + q) * (p ** 3 * (-1 + q) + q ** 3 + p ** 2 * (3 - 5 * q + 2 * q ** 2)
                                  + p * (-1 + 2 * q - 3 * q ** 2 + q ** 3))) / (
                      p ** 6 * (-1 + q) - q ** 4 + p ** 5 * (7 - 8 * q + q ** 2)
                      - p ** 4 * (13 - 11 * q - 2 * q ** 2 + q ** 3) +
                      p ** 3 * (7 + 6 * q - 16 * q ** 2 + 8 * q ** 3 - q ** 4) -
                      p ** 2 * (1 + 7 * q - 7 * q ** 2 + 4 * q ** 3 + q ** 4) + p * (q + q ** 3 + 2 * q ** 4))

    k23 = ((-1 + q) * (p ** 4 * (-1 + q) + q ** 3 + p ** 3 * (3 - 5 * q + 2 * q ** 2) +
                       p * (1 - 3 * q + q ** 2 - q ** 3) + p ** 2 * (-3 + 8 * q - 5 * q ** 2 + q ** 3))) / (
                      -p ** 6 * (-1 + q) + q ** 4 - p ** 5 * (7 - 8 * q + q ** 2)
                      + p ** 4 * (13 - 11 * q - 2 * q ** 2 + q ** 3) +
                      p ** 3 * (-7 - 6 * q + 16 * q ** 2 - 8 * q ** 3 + q ** 4) +
                      p ** 2 * (1 + 7 * q - 7 * q ** 2 + 4 * q ** 3 + q ** 4) - p * (q + q ** 3 + 2 * q ** 4))

    k24 = ((-1 + 2 * p) * (-1 + q) ** 2 * (-2 + p + q)) / det

    length_ones = np.ones((num_item, num_item)) * num_user

    m11 = get_intersection_mtx(flipped, neighbors=neighbors)
    _m00 = get_union_mtx(m11, np.array(flipped.sum(1)))
    m00 = length_ones - _m00

    reverse_flipped = -flipped + np.ones(flipped.shape) # 1 -> 0 and 0 -> 1
    reverse_flipped = csr_matrix(reverse_flipped)

    # m01 = reverse_flipped * flipped.T
    # m10 = _m00 - m11 - m01

    m10 = flipped * reverse_flipped.T

    print("{} Finish raw stats computing".format(str(datetime.now())))

    # n00_estimate = k11 * m00 + k12 * m01 + k13 * m10 + k14 * m11
    # n11_estimate = k21 * m00 + k22 * m01 + k23 * m10 + k24 * m11

    n00_estimate = k11 * m00
    n11_estimate = k21 * m00 + k23 * m10 + k24 * m11

    print("{} Finish estimation".format(str(datetime.now())))

    union_card = length_ones - n00_estimate
    intersect_card = n11_estimate

    sparse_sim = get_sim_mtx(csr_matrix(intersect_card), csr_matrix(union_card))

    print("{} Finish sim-mtx computation".format(str(datetime.now())))

    return sparse_sim


















