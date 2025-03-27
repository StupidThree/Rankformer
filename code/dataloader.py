import torch
from torch.utils.data import Dataset
import pandas as pd
from parse import args
import numpy as np
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, train_file, valid_file, test_file, device):
        self.device = device
        # train dataset
        train_data = pd.read_table(train_file, header=None, sep=' ')
        train_data = torch.from_numpy(train_data.values).to(self.device)
        self.train_data = train_data[torch.argsort(train_data[:, 0]), :]
        self.train_user, self.train_item = self.train_data[:, 0], self.train_data[:, 1]
        # valid dataset
        valid_data = pd.read_table(valid_file, header=None, sep=' ')
        valid_data = torch.from_numpy(valid_data.values).to(self.device)
        self.valid_data = valid_data[torch.argsort(valid_data[:, 0]), :]
        self.valid_user, self.valid_item = self.valid_data[:, 0], self.valid_data[:, 1]
        # test dataset
        test_data = pd.read_table(test_file, header=None, sep=' ')
        test_data = torch.from_numpy(test_data.values).to(self.device)
        self.test_data = test_data[torch.argsort(test_data[:, 0]), :]
        self.test_user, self.test_item = self.test_data[:, 0], self.test_data[:, 1]
        self.num_users = max(self.train_user.max(), self.valid_user.max(), self.test_user.max()).cpu()+1
        self.num_items = max(self.train_item.max(), self.valid_item.max(), self.test_item.max()).cpu()+1
        self.num_nodes = self.num_users+self.num_items
        print(f'{self.num_users:d} users, {self.num_items:d} items.')
        print(f'train: {self.train_user.shape[0]:d}, valid: {self.valid_user.shape[0]:d}, test: {self.test_user.shape[0]:d}.')
        self.build_batch()

    def build_batch(self):
        self.train_degree = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.train_user, torch.ones_like(self.train_user))
        self.test_degree = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.test_user, torch.ones_like(self.test_user))
        self.valid_degree = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.valid_user, torch.ones_like(self.valid_user))
        self.batch_users = [torch.arange(i, min(i+args.test_batch_size, self.num_users)).to(args.device) for i in range(0, self.num_users, args.test_batch_size)]
        self.train_batch = list(self.train_data.split([self.train_degree[batch_user].sum() for batch_user in self.batch_users]))
        self.test_batch = list(self.test_data.split([self.test_degree[batch_user].sum() for batch_user in self.batch_users]))
        self.valid_batch = list(self.valid_data.split([self.valid_degree[batch_user].sum() for batch_user in self.batch_users]))


dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)
