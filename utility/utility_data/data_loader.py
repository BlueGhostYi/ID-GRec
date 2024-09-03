import pickle
import numpy as np
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')


class Data(object):
    def __init__(self, path, config):
        self.path = path
        self.num_users = 0
        self.num_items = 0
        self.num_entities = 0
        self.num_relations = 0
        self.num_nodes = 0
        self.num_train = 0
        self.num_test = 0
        self.config = config

        self.load_data()
        if config:
            self.split_test_dict = None
            self.split_state = None
            if "sparsity_test" in config.keys() and int(config["sparsity_test"]) == 1:
                self.split_test_dict, self.split_state = self.create_sparsity_split()

    def load_data(self):
        train_path = self.path + "/train.txt"
        test_path = self.path + "/test.txt"

        train_user, self.train_user, self.train_item, self.num_train, self.pos_length = self.read_ratings(train_path)
        test_user, self.test_user, self.test_item, self.num_test, _ = self.read_ratings(test_path)

        self.num_users += 1
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items

        self.data_statistics()

        assert len(self.train_user) == len(self.train_item)

        self.user_item_net = sp.csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                           shape=(self.num_users, self.num_items))

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))
        self.test_dict = self.build_test()

    def read_ratings(self, file_name):
        inter_users, inter_items, unique_users = [], [], []
        inter_num = 0
        pos_length = []
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]
                unique_users.append(user_id)
                if len(pos_id) < 1:
                    line = f.readline()
                    continue
                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))
                inter_users.extend([user_id] * len(pos_id))
                pos_length.append(len(pos_id))
                inter_items.extend(pos_id)
                inter_num += len(pos_id)
                line = f.readline()

        return np.array(unique_users), np.array(inter_users), np.array(inter_items), inter_num, pos_length

    def data_statistics(self):
        print("\t num_users:", self.num_users)
        print("\t num_items:", self.num_items)
        print("\t num_nodes:", self.num_nodes)
        print("\t num_train:", self.num_train)
        print("\t num_test: ", self.num_test)
        print("\t sparisty: ", 1 - (self.num_train + self.num_test) / self.num_users / self.num_items)

    def get_statistics(self):
        strs = "dataset:" + self.config['dataset'] + "\t"
        strs += "num_users:%d, num_items:%d \t" % (self.num_users,self.num_items)
        strs += ("|num_train:%d, num_test:%d, sparsity: %.6f"
                 % (self.num_train, self.num_test, 1 - (self.num_train + self.num_test) / self.num_users / self.num_items))

        return strs

    # random sampling from official implementation of LightGCN
    def sample_data_to_train_random(self):
        users = np.random.randint(0, self.num_users, len(self.train_user))
        sample_list = []
        for i, user in enumerate(users):
            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue
            positive_index = np.random.randint(0, len(positive_items))
            positive_item = positive_items[positive_index]
            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    continue
                else:
                    break
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)

    def sample_data_to_train_all(self):
        sample_list = []
        for i in range(len(self.train_user)):
            user = self.train_user[i]

            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue

            positive_item = self.train_item[i]

            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    continue
                else:
                    break
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)

    def get_user_pos_items(self, users):
        positive_items = []
        for user in users:
            positive_items.append(self.user_item_net[user].nonzero()[1])
        return positive_items

    def get_user_n_neg_items(self, users, n):
        negative_items = []
        for user in users:
            negative_list = []
            for i in range(n):
                while True:
                    negative_item = np.random.randint(0, self.num_items)
                    if negative_item in self.all_positive[user]:
                        continue
                    else:
                        negative_list.append(negative_item)
                        break
            negative_items.append(negative_list)

        return negative_items

    def build_test(self):
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = dict()

        for uid in all_users:
            train_iids = self.all_positive[uid]
            test_iids = self.test_dict[uid]

            num_iids = len(train_iids) + len(test_iids)

            if num_iids not in user_n_iid.keys():
                user_n_iid[num_iids] = [uid]
            else:
                user_n_iid[num_iids].append(uid)

        split_uids = list()
        temp = []
        count = 1
        fold = 4
        n_count = self.num_train + self.num_test
        n_rates = 0
        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.num_train + self.num_test):
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
