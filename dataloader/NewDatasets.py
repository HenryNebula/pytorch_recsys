from dataloader.NewAbstracter import NewAbstracter


class ML20M (NewAbstracter):
    def __init__(self, root, num_neg):
        super(ML20M, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset ML20M loaded.')
        self.num_users, self.num_items = 53220, 14561
        print('num_users: {0}, num_items: {1}.'.format(self.num_users, self.num_items))
        print('-----------------------------------')


class ML20MSparse (NewAbstracter):
    def __init__(self, root, num_neg):
        super(ML20MSparse, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset ML20M-Sparse loaded.')
        self.load_meta()
        # self.num_users, self.num_items = 51634, 8532
        print('num_users: {0}, num_items: {1}, sparsity: {2}.'.format(self.num_users, self.num_items, self.get_sparsity()))
        print('-----------------------------------')


class ML20MContext (NewAbstracter):
    def __init__(self, root, num_neg):
        super(ML20MContext, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset ML20M-Context loaded.')
        self.num_users, self.num_items = 51634, 8532
        print('num_users: {0}, num_items: {1}, sparsity: {2}.'.format(self.num_users, self.num_items, self.get_sparsity()))
        print('-----------------------------------')


class ML10M (NewAbstracter):
    def __init__(self, root, num_neg):
        super(ML10M, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset ML10M loaded.')
        self.num_users, self.num_items = 67959, 8882
        print('num_users: {0}, num_items: {1}, sparsity: {2}.'.format(self.num_users, self.num_items, self.get_sparsity()))
        print('-----------------------------------')


class App (NewAbstracter):
    def __init__(self, root, num_neg):
        super(App, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset App loaded.')
        self.num_users, self.num_items = 871, 1682
        print('num_users: {0}, num_items: {1}, sparsity: {2}.'.format(self.num_users, self.num_items, self.get_sparsity()))
        print('-----------------------------------')


class Yelp(NewAbstracter):
    def __init__(self, root, num_neg):
        super(Yelp, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset Yelp loaded.')
        self.num_users, self.num_items = 6828, 9132
        print('num_users: {0}, num_items: {1}, sparsity: {2}.'.format(self.num_users, self.num_items, self.get_sparsity()))
        print('-----------------------------------')


class AmazonBook (NewAbstracter):
    def __init__(self, root, num_neg):
        super(AmazonBook, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset AmazonBook loaded.')
        self.num_users, self.num_items = 52643, 91599
        print('num_users: {0}, num_items: {1}.'.format(self.num_users, self.num_items))
        print('-----------------------------------')


class LastFM (NewAbstracter):
    def __init__(self, root, num_neg):
        super(LastFM, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset LastFM loaded.')
        self.num_users, self.num_items = 23566, 48123
        print('num_users: {0}, num_items: {1}.'.format(self.num_users, self.num_items))
        print('-----------------------------------')


class Yelp2018 (NewAbstracter):
    def __init__(self, root, num_neg):
        super(Yelp2018, self).__init__(root, num_neg)
        print('-----------------------------------')
        print('Dataset Yelp loaded.')
        self.num_users, self.num_items = 31831, 40841
        print('num_users: {0}, num_items: {1}.'.format(self.num_users, self.num_items))
        print('-----------------------------------')
