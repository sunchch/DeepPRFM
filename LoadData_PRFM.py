'''
Utilities for Loading data.
'''

class LoadData(object):
    def __init__(self, path, dataset):
        self.path = path + dataset + "/"
        self.trainfile = self.path + "train.csv"
        self.testfile = self.path + "test.csv"
        self.user_field_M, self.item_field_M = self.get_length()
        print("user_field_M", self.user_field_M)
        print("item_field_M", self.item_field_M)
        print("field_M", self.user_field_M + self.item_field_M)
        self.item_bind_M = self.bind_item()
        self.user_bind_M = self.bind_user()
        print("item_bind_M", len(self.binded_items.values()))
        print("user_bind_M", len(self.binded_users.values()))
        self.user_positive_list = self.get_positive_list(self.trainfile)
        self.Train_data, self.Test_data = self.construct_data()

    def get_length(self):
        length_user = 0
        length_item = 0
        f = open(self.trainfile)
        line = f.readline()
        while line:
            user_features = line.strip().split(',')[0].split('-')
            item_features = line.strip().split(',')[1].split('-')
            for user_feature in user_features:
                feature = int(user_feature)
                if feature > length_user:
                    length_user = feature
            for item_feature in item_features:
                feature = int(item_feature)
                if feature > length_item:
                    length_item = feature
            line = f.readline()
        f.close()
        return length_user + 1, length_item + 1

    def bind_item(self):
        self.binded_items = {}  # dic{feature: id}
        self.item_map = {}  # dic{id: feature}
        self.bind_i(self.trainfile)
        self.bind_i(self.testfile)
        return len(self.binded_items)

    def bind_i(self, file):
        f = open(file)
        line = f.readline()
        i = len(self.binded_items)
        while line:
            features = line.strip().split(',')
            item_features = features[1]
            if item_features not in self.binded_items:
                self.binded_items[item_features] = i
                self.item_map[i] = item_features
                i = i + 1
            line = f.readline()
        f.close()

    def bind_user(self):
        self.binded_users = {}
        self.bind_u(self.trainfile)
        self.bind_u(self.testfile)
        return len(self.binded_users)

    def bind_u(self, file):
        f = open(file)
        line = f.readline()
        i = len(self.binded_users)
        while line:
            features = line.strip().split(',')
            user_features = features[0]
            if user_features not in self.binded_users:
                self.binded_users[user_features] = i
                i = i + 1
            line = f.readline()
        f.close()

    def get_positive_list(self, file):
        f = open(file)
        line = f.readline()
        user_positive_list = {}
        while line:
            features = line.strip().split(',')
            user_id = self.binded_users[features[0]]
            item_id = self.binded_items[features[1]]
            if user_id in user_positive_list:
                user_positive_list[user_id].append(item_id)
            else:
                user_positive_list[user_id] = [item_id]
            line = f.readline()
        f.close()
        return user_positive_list

    def construct_data(self):
        X_user, X_item = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(X_user, X_item)
        print("# of training:", len(X_user))

        X_user, X_item = self.read_data(self.testfile)
        Test_data = self.construct_dataset(X_user, X_item)
        print("# of test:", len(X_user))
        return Train_data, Test_data

    def read_data(self, file):
        f = open(file)
        X_user = []
        X_item = []
        line = f.readline()
        while line:
            features = line.strip().split(',')
            user_features = features[0].split('-')
            X_user.append([int(item) for item in user_features[0:]])
            item_features = features[1].split('-')
            X_item.append([int(item) for item in item_features[0:]])
            line = f.readline()
        f.close()
        return X_user, X_item

    def construct_dataset(self, X_user, X_item):
        Data_Dic = {}
        indexs = range(len(X_user))
        Data_Dic['X_user'] = [X_user[i] for i in indexs]
        Data_Dic['X_item'] = [X_item[i] for i in indexs]
        return Data_Dic
