from torch import optim
from torch.utils.data import DataLoader
import utility.utils as utils
from utility.diary import Diary
from utility.Logger import create_logger
from dataloader.FakeDataset import FakeDataset

from graph.MF import MF
from graph.GMF import GMF
from graph.MLP import MLP
from graph.NeuMF import NeuMF
from graph.LadderGMF import LadderGMF
from graph.DoubleGMF import DoubleGMF
from graph.SharedNeuMF import SharedNeuMF
from graph.PMF import PMF
from graph.CML import CML
from graph.DMF_extension import DMF_A, DMF_B, DMF_C, DMF_D

import json
import GPUtil as gputil
import torch.nn.parallel as parallel
from torch.utils.data import distributed
import os


class Trainer():
    def __init__(self, args):

        self.args = args

        # load logger
        self.diary = Diary(os.path.join(args.output, args.make_diary), makedir=True) if args.make_diary else None
        self.logger = create_logger(os.path.join(args.output, args.log_name))

        self.assign_gpu()

        # load generic config
        with open(os.path.join(os.path.dirname(__file__), 'config_json', args.config)) as f:
            self.config = json.loads(f.read())

        self.logger.debug("start loading dataset!")
        # dataset abstraction
        self.trainloader, self.trainsampler, self.val_evaluator, self.test_evaluator, self.dataset = self.get_trainloader()
        self.logger.debug("finish loading evaluators for both val and test phase")

        self.fake_dataset = FakeDataset(self.dataset)

        # model parameters
        self.model, self.optimizer, self.lr_rate = self.load_model()

    def assign_gpu(self):
        # find optimal GPU
        if self.args.gpu != -1:
            gpu_list = gputil.getAvailable(
                order='memory', limit=10, maxLoad=0.7, maxMemory=0.7, excludeID=[5])
            if not gpu_list:
                raise (Exception("NO Available GPU!"))

            shell_gpu = self.args.gpu
            if shell_gpu not in gpu_list:
                shell_gpu = gpu_list[0]

            os.environ['CUDA_VISIBLE_DEVICES'] = str(shell_gpu)
            self.logger.debug("finish preparing gpu: ({} -> {})".format(self.args.gpu, shell_gpu))

        else:
            self.logger.debug("finishing preparing device as cpu")

    def get_generic_loader(self, dataset):
        # only use distributed version in mCore mode
        # different loader between CPU and GPU training

        args = self.args
        if not args.mCore:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=self.config['shuffle'],
                                     num_workers=8, drop_last=True)
            sampler = None
        else:
            sampler = distributed.DistributedSampler(dataset)
            divided_bs = int(args.batch_size / args.world_size)
            dataloader = DataLoader(dataset, batch_size=divided_bs, shuffle=self.config['shuffle'],
                                     num_workers=4, drop_last=True, sampler=sampler, pin_memory=True)
        return dataloader, sampler

    def get_trainloader(self):
        # Dataloader, evaluator, model and optimizer.
        args = self.args

        dataset = utils.load_dataset(self.config, args.dataset, args.num_neg)

        trainloader, trainsampler = self.get_generic_loader(dataset) if args.make_diary else None, None

        val_evaluator = dataset.get_evaluator(is_val=True)
        test_evaluator = dataset.get_evaluator(is_val=False)

        return trainloader, trainsampler, val_evaluator, test_evaluator, dataset

    def get_fake_loader(self, new_data:dict):
        if "train_data" in new_data and new_data['train_data'].nnz != 0:
            self.fake_dataset.update_data(new_data)

        fakeloader, fakesampler = self.get_generic_loader(self.fake_dataset)

        return fakeloader, fakesampler

    def load_model(self):
        # Model
        args = self.args

        num_users, num_items = self.dataset.count()
        dict_config = {'num_users': num_users, 'num_items': num_items, 'num_factors': args.num_factors,
                       'loss_type': args.loss, 'reg': args.reg, 'device': args.gpu,
                       'norm_user': args.norm_user, 'norm_item': args.norm_item,
                       'use_user_bias': args.use_user_bias, 'use_item_bias': args.use_item_bias,
                       'multiplier': args.multiplier, 'bias': args.bias,
                       'square_dist': args.square_dist, 'mapping': args.gmf_linear, 'fuse': args.fuse}

        if args.method == 'MF':
            model = MF(dict_config)
        elif args.method == 'GMF':
            model = GMF(dict_config)
        elif args.method == 'MLP':
            model = MLP(dict_config)
        elif args.method == 'NeuMF':
            model = NeuMF(dict_config)
        elif args.method == 'LadderGMF':
            model = LadderGMF(dict_config)
        elif args.method == 'DoubleGMF':
            model = DoubleGMF(dict_config)
        elif args.method == 'SharedNeuMF':
            model = SharedNeuMF(dict_config)
        elif args.method == 'PMF':
            model = PMF(dict_config)
        elif args.method == 'DMF_A':
            model = DMF_A(dict_config, norm=args.dist_norm)
        elif args.method == 'DMF_B':
            model = DMF_B(dict_config, norm=args.dist_norm)
        elif args.method == 'DMF_C':
            model = DMF_C(dict_config, norm=args.dist_norm)
        elif args.method == 'DMF_D':
            model = DMF_D(dict_config, norm=args.dist_norm)
        elif args.method == 'CML':
            model = CML(dict_config)
        elif args.item_pop or 'ItemCF' in args.method:
            return None, None, 0
        else:
            raise (Exception('Method {0} not found. Choose the method from '
                             '[GMF, MLP, NeuMF, LadderGMF, DoubleGMF, SharedNeuMF, PMF, DMF_(A,B_C,D), CML].'.format(
                    args.method)))
        model.to(model.device)

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=args.reg)

        # To continue training. Load model from the same class if needed.
        if args.path_to_load_embedding != 'default':
            model.load_embedding_from_file(args.path_to_load_embedding)
        elif args.path_to_load_model != 'default':
            optimizer = model.load_model_from_file(args.path_to_load_model, optimizer=optimizer)

        # Fine-tuning. Load pretrained model from the subclass. (Only for NeuMF and DoubleGMF)
        if args.path_to_subclass_model != ['default', 'default']:
            model.load_pretrained_model(args.path_to_subclass_model)
        if args.path_to_subclass_embedding != ['default', 'default']:
            model.load_pretrained_embedding(args.path_to_subclass_embedding)
        if args.fix_left:
            optimizer = model.fix_left(optimizer)
        if args.fix_right:
            optimizer = model.fix_right(optimizer)

        if args.mCore:
            model = parallel.DistributedDataParallelCPU(model)

        return model, optimizer, lr

## report.
# diary.report_last(type_='max')
# hr, ndcg = test_evaluator.eval_once(model, 0, topK=args.topK)
# print('epoch = {0}, loss={1:.4f}, lr = {2}, HR = {3:.3f}, NDCG = {4:.3f}'.format(0, -1, lr_rate, hr, ndcg))

