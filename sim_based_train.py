import setproctitle
import utility.utils as utils
from Trainer import Trainer
import os
from graph.ItemCF import ItemCF, ItemCF_with_Laplacian, ItemCF_Noisy, ItemCF_Optimal
from noise.ItemSim import MAX_EPS
import numpy as np


def init_args(args_):

    print("Task Name: {}".format(args_.task))

    if args_.task == 'default':
        args_.dataset = 'ml-20m-context'
        args_.task = 'test_sim'
        args_.topK = 5
        args_.method = 'ItemCF_KDD'
        args_.eps = 1000
        args_.num_neighbor = 10
        args_.config = 'itemsim_config.json'

        args_.num_centroid = 1
        args_.clustering_method = "kmeans"
        args_.feature_type = "word2vec"

    if args_.method == 'ItemCF_KDD':
        output_prefix = 'item_sim_KDD'
        args_.output = '{}/{}/{}-norm/Neighbor_{}'.format(output_prefix, args_.dataset, args_.norm, args_.num_neighbor)

    elif args_.method == 'ItemCF_Noisy':
        output_prefix = 'item_sim_noisy'
        args_.output = '{}/{}/Neighbor_{}'.format(output_prefix, args_.dataset, args_.num_neighbor)

    elif args_.method == 'ItemCF_Noisy_Optimal':
        output_prefix = 'item_sim_noisy_optimal'
        args_.output = '{}/{}/Neighbor_{}'.format(output_prefix, args_.dataset, args_.num_neighbor)

    elif args_.method == 'ItemCF_Optimal':
        output_prefix = 'item_sim_Optimal'
        args_.output = '{}/{}/Neighbor_{}'.format(output_prefix, args_.dataset, args_.num_neighbor)

    else:
        output_prefix = 'item_sim'
        if args_.num_centroid == 1:
            args_.output = '{}/{}/Neighbor_{}'.format(output_prefix, args_.dataset, args_.num_neighbor)
        else:
            args_.output = '{}/{}/{}/{}/Neighbor_{}'.format(output_prefix, args_.dataset, args_.clustering_method,
                                                   args_.feature_type, args_.num_neighbor)

    args_.method = 'ItemCF' if args_.method != 'ItemCF' else args_.method
    args_.log_name = "eps_{}-N_{}.log".format(args_.eps, args_.num_neighbor)
    args_.make_diary = False

    return args_


if __name__ == '__main__':
    args = utils.parse_args()

    args = init_args(args)

    setproctitle.setproctitle(args.task)

    trainer = Trainer(args)

    total_hr_list = np.zeros((args.cycle, ))
    total_ndcg_list = np.zeros((args.cycle, ))

    # print(trainer.val_evaluator.eval_item_pop(trainer.dataset.popularity, topK=5))

    load = True if args.cycle == -1 else False

    for c in range(args.cycle):
        output_path = os.path.join("output/", args.output)
        config = {}

        if 'KDD' in args.output:
            Model = ItemCF_with_Laplacian

        elif 'noisy' in args.output:
            Model = ItemCF_Noisy
            config['optimal'] = True if 'optimal' in args.output else False

        elif 'Optimal' in args.output:
            Model = ItemCF_Optimal

        else:
            Model = ItemCF

        if not load:
            item_cf = Model(trainer, args, max_neighbors=trainer.dataset.num_items, bloom_config=config,
                            sl_path=output_path, load=load, async_fit=False)
            trainer.logger.info("Cycle {} ends!".format(c))
            hr_list, ndcg_list = item_cf.evaluate(trainer.val_evaluator)
            trainer.logger.info([hr_list, ndcg_list])
            total_hr_list[c] = hr_list[-1]
            total_ndcg_list[c] = ndcg_list[-1]
            trainer.logger.info(
                "Summary: Mean HR: {}, Std HR: {}".format(np.mean(total_hr_list), np.std(total_hr_list)))
            trainer.logger.info(
                "Summary: Mean NDCG: {}, Std NDCG: {}".format(np.mean(total_ndcg_list), np.std(total_ndcg_list)))

        else:
            item_cf = Model(trainer, args, max_neighbors=trainer.dataset.num_items, bloom_config=config,
                            sl_path=output_path, load=load, async_fit=False)
            item_cf.compare_noise()






