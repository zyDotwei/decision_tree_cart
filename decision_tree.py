import numpy as np
from collections import Counter


class DecisionTree:

    def __init__(self, criterion='gini', attr_set=None):
        self.attr_set = attr_set
        self.criterion = criterion
        self.cart_tree = None

    def _judge_same_values(self, x_data):
        """
        :param x_data:
        :return: True 所有样本在所有属性取值上相同
                False 不相等
        """
        temp = x_data[0]
        for data in x_data:
            b_judge = temp == data
            if False in b_judge:
                return False
        return True

    def _get_major_class(self, label_data):

        label_key = 0
        value = 0
        label_count = Counter(label_data)
        for k, v in label_count.items():
            if v > value:
                value = v
                label_key = k
        return label_key

    def gini(self, y_data):
        counter = Counter(y_data)
        res = 1.0
        for num in counter.values():
            p = num / len(y_data)
            res -= p ** 2
        return res

    def _split_dataset(self, data, y_data, axis, value):
        x_sub_data = []
        y_sub_data = []
        x_other_data = []
        y_other_data = []
        for featVec, label in zip(data, y_data):  # 抽取符合划分特征的值
            if featVec[axis] == value:
                x_sub_data.append(featVec)
                y_sub_data.append(label)
            else:   # 非value数据
                x_other_data.append(featVec)
                y_other_data.append(label)

        return np.array(x_sub_data), np.array(y_sub_data), np.array(x_other_data), np.array(y_other_data)

    def _cart_best_split(self, x_data, y_data):

        best_gini = float('inf')
        best_feature = -1
        best_values = -1
        for d in range(x_data.shape[1]):
            feature_value = x_data[:, d]
            feature_value = sorted(set(feature_value))
            for value in feature_value:
                x_sub, y_sub, x_other, y_other = self._split_dataset(x_data, y_data, d, value)
                gini = len(y_sub)/len(y_data) * self.gini(y_sub) + len(y_other)/len(y_data) * self.gini(y_other)
                # print('第{}个维度，切分值为{}，gini系数为{:.4f}'.format(d+1, value, gini))
                if gini < best_gini:
                    best_gini, best_feature, best_values = gini, d, value
        return best_gini, best_feature, best_values

    def cart_create_tree(self, x_data, label_data):

        uni_lens = len(np.unique(label_data))
        if uni_lens == 1:
            # 类别全部都属于同一类
            return label_data[0]
        elif len(self.attr_set) == 0 or self._judge_same_values(x_data):
            return self._get_major_class(label_data)

        best_gini, best_feature, best_values = self._cart_best_split(x_data, label_data)
        best_feature_attr = self.attr_set[best_feature]  # 最好的属性
        x_sub, y_sub, x_other, y_other = self._split_dataset(x_data, label_data, best_feature, best_values)
        tree_root = best_feature_attr + '=={}?'.format(best_values)
        cart_tree = {tree_root: {}}  # 树结构

        if len(x_sub) == 0:
            cart_tree[tree_root][1] = self._get_major_class(label_data)
        else:
            cart_tree[tree_root][1] = self.cart_create_tree(x_sub, y_sub)

        if len(x_other) == 0:
            cart_tree[tree_root][0] = self._get_major_class(label_data)
        else:
            cart_tree[tree_root][0] = self.cart_create_tree(x_other, y_other)

        return cart_tree

    def fit(self, x_data, label_data):
        self.cart_tree = self.cart_create_tree(x_data, label_data)
        return self.cart_tree

    def classify(self, decesion_tree, feature_label, data):
        """
        输入：决策树，分类标签，测试数据
        输出：决策结果
        """
        root_node = list(decesion_tree.keys())[0]
        next_node = decesion_tree[root_node]
        cur_feature, feature_val = str(root_node).split('?')[0].split('==')
        feature_indx = feature_label.index(cur_feature)

        class_label = None
        if data[feature_indx] == int(feature_val):
            if type(next_node[1]).__name__ == 'dict':
                class_label = self.classify(next_node[1], feature_label, data)
            else:
                class_label = next_node[1]
        else:
            if type(next_node[0]).__name__ == 'dict':
                class_label = self.classify(next_node[0], feature_label, data)
            else:
                class_label = next_node[0]

        return class_label

    def predict(self, x_test):

        predicted_label = []
        for data in x_test:
            predicted_label.append(self.classify(self.cart_tree, self.attr_set, data))
        return predicted_label

    def accuracy_score(self, y_true, y_predict):
        """计算y_true和y_predict之间的准确率"""
        assert len(y_true) == len(y_predict), \
            "the size of y_true must be equal to the size of y_predict"

        return np.sum(y_true == y_predict) / len(y_true)

    def score(self, x_test, y_test):
        predicted_label = []
        for data in x_test:
            predicted_label.append(self.classify(self.cart_tree, self.attr_set, data))

        return self.accuracy_score(y_test, predicted_label)


