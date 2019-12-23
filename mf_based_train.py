import setproctitle
import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from utility import utils
from Trainer import Trainer
from torch.utils.data import distributed
from utility.distributed_toolkit import collect_data
from noise.ItemSimSparse import flipping_helper
import os


def core_training(trainer:Trainer, new_data:dict, cycle=-1):
        # same lr, epoch for different sides;
        # different dataloader and training options (fix embeddings or not)
        lr_rate = trainer.lr_rate

        sampler: distributed.DistributedSampler
        loader, sampler = trainer.get_fake_loader(new_data)
        trainer.logger.info("New Sparsity: {}".format(trainer.fake_dataset.sparsity))

        trainer.model.train_with_option(fake=True)
        epoch_range = range(trainer.args.epochs)

        loss_list = [] # loss list for training set

        # early stopping parameters
        start_epoch = 30
        early_stop_buffer = 10
        best_hr, last_hr = 0, 0
        best_epoch = 0

        # model_dict = {"model": None, "optimizer": None}

        for epoch in tqdm(epoch_range):
            if sampler:
                sampler.set_epoch(epoch)

            trainer.fake_dataset.refresh_trainlist()
            loader, sampler = trainer.get_fake_loader({})

            loss_ = 0.0
            batch_count = 0

            for i, (users, items, labels) in enumerate(loader):
                batch_count += 1
                users = Variable(users.to(trainer.model.device)).long()
                items = Variable(items.to(trainer.model.device)).long()
                labels = Variable(labels.to(trainer.model.device)).float()
    
                users, items, labels = users.view(-1), items.view(-1), labels.view(-1)
                res = trainer.model([users, items])
                res = res.reshape(-1)  # [bs*5, 1] -> [bs*5]

                weights = torch.Tensor([1]*len(labels))
                weights = weights.to(trainer.model.device)

                loss = trainer.model.get_loss(res, labels, weights=weights,
                        target=trainer.args.target, bpr_margin=trainer.args.bpr_margin, mg_margin=trainer.args.mg_margin)
                trainer.optimizer.zero_grad()
                loss.backward()
                loss_ += loss.detach().cpu().numpy()
                trainer.optimizer.step()

            loss_ /= batch_count # average training loss in this epoch
            loss_list.append(loss_)

            if epoch + 1 > start_epoch:
                # start early stopping here

                # use HR@topK as metric for early stopping on val set
                val_evaluator = trainer.val_evaluator
                hr, ndcg = val_evaluator.eval_once(trainer.model, epoch, topK=trainer.args.topK)
                info = val_evaluator.get_info()
                present_hr = hr[-1]
                trainer.diary.update_info(trainer.args, trainer.config, info, np.array([loss_list]))
                trainer.logger.info('train-loss@{0}={1:.4f},lr={2},val-HR@{3}={4:.4f}'.format(
                                    epoch, loss_, lr_rate, trainer.args.topK, present_hr))

                if present_hr < last_hr: # performance degrade
                    early_stop_buffer -= 1
                    trainer.logger.warning("Last hr {0:.4f}, Present hr {1:.4f}, buffer left {2}".format(
                        last_hr, present_hr, early_stop_buffer))
                else:
                    early_stop_buffer += 2 # double increment for model upgrading
                    early_stop_buffer = min(early_stop_buffer, 10)

                last_hr = present_hr

                if present_hr > best_hr:
                    trainer.logger.info("Better HR@{}: {}".format(trainer.args.topK, present_hr))
                    best_hr = present_hr
                    best_epoch = epoch
                    early_stop_buffer = 10
                    # only save models on client side
                    # model_dict = {"model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict()}
                    # save rank_file
                    ranking_path = os.path.join(trainer.diary.output_dir, trainer.diary.get_next_folder(), "rankings.json")
                    val_evaluator.eval_once(trainer.model, epoch=-1, topK=trainer.args.topK, rank_file=ranking_path)

            # save model when encounter save_interval or use up all epochs
            if (epoch + 1) % trainer.args.save_interval == 0 or epoch == max(epoch_range):
                # trainer.diary.save_model(model_dict["model"], model_dict["optimizer"], loss_, epoch)
                trainer.logger.info("No early stopping!")

            # lr decay
            if (epoch + 1) in trainer.config['lr_decay_epoch']:
                lr_rate = lr_rate * trainer.config['lr_decay']
                utils.adjust_lr(trainer.optimizer, lr_rate)

            if trainer.config["debug_test"] and (epoch + 1) in trainer.config["debug_epoch"]:
                hr, ndcg = trainer.test_evaluator.eval_once(trainer.model, 0, topK=trainer.args.topK)
                trainer.logger.info(
                    'TEST_EVAL: epoch = {0}, lr = {1}, HR = {2:.3f}, NDCG = {3:.3f}'.format(0, lr_rate, hr[-1], ndcg[-1]))

            if early_stop_buffer < 0 and start_epoch < 0:
                # trainer.diary.save_model(model_dict["model"], model_dict["optimizer"], loss_, best_epoch)
                trainer.logger.info("Early Stopping at epoch {}. Model@Epoch_{} saved!".format(epoch, best_epoch))

            if args.gpu != -1:
                torch.cuda.empty_cache()


def init_args(args_):
    print("Task Name: {}".format(args_.task))

    if args_.task == 'default':
        """used for debugging"""
        args_.dataset = 'ml-20m-context'
        args_.task = 'test_mf'
        args_.topK = 5
        args_.method = 'BaseMF'
        args_.eps = 1
        args_.config = 'itemsim_config.json'

    output_prefix = args_.method if args_.symm else "Asym_" + args_.method
    paras = "lr_{}-Reg_{}-Embed_{}".format(args_.lr, args_.reg, args_.num_factors)

    args_.output = '{}/{}/eps_{}/'.format(output_prefix, args_.dataset, args_.eps)
    args_.log_name = "{}.log".format(paras)
    args_.make_diary = paras + '/'
    return args_


def run_schedule(args, cycles):
    if cycles != -1:
        args.eps = args.eps / cycles
        args.epochs = int(args.epochs / cycles)

    if args.symm:
        p = np.exp(args.eps) / (np.exp(args.eps) + 1)
        q = 1 - p
    else:
        p = 1
        q = p / args.eps

    trainer = Trainer(args)
    real_mtx = trainer.dataset.train_data
    trainer.logger.info("Start random flipping...")
    ui_mtx = flipping_helper(real_mtx, p, q)
    trainer.logger.info("End random flipping...")

    if args.item_pop:
        hr, ndcg = trainer.val_evaluator.eval_item_pop(trainer.dataset.popularity, topK=5)
        print("HR@5:[{}], NDCG@5:[{}]".format(hr, ndcg))
        return

    if trainer.model:
        if cycles == -1:
            new_data = {}
            # centralized baseline
            core_training(trainer, new_data={"train_data": ui_mtx.tolil()}, cycle=-1,)
        else:
            # distributed learning
            new_data = collect_data(trainer, use_embed=False)
            for c in range(cycles):
                core_training(trainer, new_data=new_data, cycle=c+1)


if __name__ == "__main__":
    args = utils.parse_args()
    args = init_args(args)

    setproctitle.setproctitle(args.task)
    processes = []

    if not args.mCore:
        run_schedule(args, cycles=args.cycle)

    else:
        assert False, "Multi-Core version not implemented!"