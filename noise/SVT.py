import numpy as np
from numpy.random import laplace
from scipy.sparse import lil_matrix


def laplacian(beta, size=None):
    return laplace(0, beta, size)


def SVT(query_list, thresh_list:list, sens:float, topC:int, eps_1:float, eps_2:float, eps_3:float=0):
    pos_ans = []
    idx_arr = [0] * len(query_list)
    query_idx = -1
    noise_thresh = laplacian(sens/eps_1)
    count = 0
    while len(pos_ans) < topC:
        count += 1
        query_idx += 1
        query_idx = query_idx % len(query_list)

        if idx_arr[query_idx] == 1:
            continue

        query = query_list[query_idx]

        noise = laplacian(2*topC*sens/eps_2)

        if query + noise >= thresh_list[query_idx] + noise_thresh:
            if eps_3 > 0:
                pos_ans.append((query_idx, query + laplacian(topC*sens/eps_3)))
            else:
                pos_ans.append((query_idx, 1))

            idx_arr[query_idx] = 1

    return pos_ans


def direct_report(user_row:lil_matrix):
    user_row = user_row.rows[0]
    user_row_coo = user_row.tocoo()
    query_list = user_row_coo.col.tolist()
    return query_list


def report_with_SVT(user_row:list, eps:float, num_item:int, thresh=0.5, topC=None):
    # user_row actually contains index of 1s
    num_item_bought = len(user_row)
    query_list = np.zeros((num_item,))
    query_list[user_row] = 1
    # query_list = query_list.tolist()
    thresh_list = [thresh] * len(query_list)
    sens = 1
    topC = num_item_bought if not topC else topC

    ratio = (2 * topC) ** (2/3)
    eps_1, eps_2 = 1/(ratio+1) * eps, ratio/(ratio+1) * eps
    pos_ans = SVT(query_list, thresh_list, sens=sens, topC=topC, eps_1=eps_1, eps_2=eps_2)
    return [x[0] for x in pos_ans]


if __name__ == '__main__':
    print(np.mean([laplacian(20) for i in range(20000)]))






