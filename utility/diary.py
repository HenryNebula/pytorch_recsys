import os
import time
import pickle
import numpy as np
import utility.utils as utils
from datetime import datetime


class Diary(object):

    def __init__(self, output_dir='output', makedir=False):
        curPath = os.path.dirname(__file__)
        self.output_dir = os.path.join(curPath, '../output', output_dir)
        print("Dir: {}".format(self.output_dir))

        self.dt_format = '%Y-%m-%dT%H:%M:%S'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.file_list = os.listdir(self.output_dir)
        self.next_folder = 'default'
        if makedir:
            self.next_folder = self.get_next_folder()
            os.makedirs(os.path.join(self.output_dir, self.next_folder))
        else:
            if not os.path.exists(self.output_dir):
                raise (Exception('Path {0} not found.'.format(self.output_dir)))

    def get_last_folder(self):
        dir_list = self.iterate_output_root()
        if not dir_list:
            raise('Error! No folder exists in \'{0}\'.'.format(self.output_dir))
        else:
            dir = max(dir_list).strftime(self.dt_format)
            return dir

    def iterate_output_root(self):
        dir_list = [datetime.strptime(k, self.dt_format) for k in self.file_list]
        return dir_list

    def get_next_folder(self):
        if self.next_folder != 'default':
            return self.next_folder
        return datetime.now().strftime(self.dt_format)

    def update_info(self, arguments, config, info, loss_array):
        num_next = self.get_next_folder()
        nowpath = os.path.join(self.output_dir, num_next)
        # save diary
        data = {'args': arguments, 'config': config, 'info': info, 'time': time.ctime(), 'loss': loss_array}
        with open(os.path.join(nowpath, 'diary.pth'), 'wb') as f:
            pickle.dump(data, f)

    def save_model(self, model_state_dict, optimizer_state_dict, loss, epoch):
        num_next = self.get_next_folder()
        curPath = os.path.join(self.output_dir, num_next)
        # save model
        if epoch != -1:
            model_name = 'epoch_{0}_val-loss_{1:.4f}.pth'.format(epoch, loss)
        else:
            model_name = 'default.pth'

        dict_ = {'model': model_state_dict, 'optim': optimizer_state_dict}
        utils.save_model(dict_, curPath, model_name)

    def get_info_from_folder(self, folder):
        nowpath = os.path.join(self.output_dir, folder)
        fname = os.path.join(nowpath, 'diary.pth')
        if not os.path.isfile(fname):
            return 0
        with open(fname, 'rb') as f:
            res = pickle.load(f, encoding='latin1')
        return res

    def get_info(self):
        self.info = {}
        for folder in self.file_list:
            data = self.get_info_from_folder(folder)
            # if data == 0:
            if data == 0 or data['args'].task == 'test':
                continue
            self.info[folder] = data

    def get_last_from_info(self, info):
        hr_list, ndcg_list, epoch_list = info[0], info[1], info[2]
        return hr_list[-1], ndcg_list[-1], epoch_list[-1]

    def get_max_from_info(self, info):
        hr_list, ndcg_list, epoch_list = info[0], info[1], info[2]
        idx = np.argmax(hr_list)
        return hr_list[idx], ndcg_list[idx], epoch_list[idx]

    def report_info(self, folder, data, type_='last'):
        info = data['info']

        epoch_interval = info[2][1] - info[2][0] if len(info[2]) > 1 else 1
        epoch_num = info[2][0] + epoch_interval * (len(info[2]) - 1)

        if type_ == 'last':
            hr, ndcg, epoch = self.get_last_from_info(info)
        elif type_ == 'max':
            hr, ndcg, epoch = self.get_max_from_info(info)
        else:
            raise(Exception('Invalid type {0}.'.format(type_)))

        args = data['args']
        print('Folder: {0},\nArgs: {1}\nepoch = {2:3d}, size = {3:2d}\nHR@{4:d}= {5:.3f}, NDCG@{4:d} = {6:.3f}'.
              format(folder, args, epoch_num, epoch, args.topK, hr, ndcg))

    def report_last(self, type_='last'):
        print('------------ REPORT ---------------')
        folder = self.get_next_folder()
        data = self.get_info_from_folder(folder)
        self.report_info(folder, data, type_)
        
    def report_all(self, type_='last'):
        self.__init__(self.output_dir)
        print('------------ DIARY ---------------')
        self.get_info()
        key_list = list(self.info.keys())
        num_list = [datetime.strptime(k, self.dt_format) for k in key_list]
        num_list.sort()
        folder_list = [t.strftime(self.dt_format) for t in num_list]
        for folder in folder_list:
            data = self.info[folder]
            self.report_info(folder, data, type_)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for the diary.")
    parser.add_argument('--path', type=str, default='Cent-MF-lr0.0005-reg0.001', help='path to the output folder.')
    args = parser.parse_args()

    diary = Diary(args.path)
    diary.report_all('max')

