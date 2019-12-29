import pandas as pd
import pandas as np
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from tree_plotter import plot_cart_tree


def class_mapping(data=None, label=None):

    if data is not None:
        price_mapping = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
        dors_mapping = {'2': 0, '3': 1, '4': 2, '5more': 3}
        persons_mapping = {'2': 0, '4': 1, 'more': 2}
        lug_boot_mapping = {'small': 0, 'med': 1, 'big': 2}
        safety_mapping = {'low': 0, 'med': 1, 'high': 2}

        data.loc[:, 'buying'] = data.loc[:, 'buying'].map(price_mapping)
        data.loc[:, 'maint'] = data.loc[:, 'maint'].map(price_mapping)
        data.loc[:, 'doors'] = data.loc[:, 'doors'].map(dors_mapping)
        data.loc[:, 'persons'] = data.loc[:, 'persons'].map(persons_mapping)
        data.loc[:, 'lug_boot'] = data.loc[:, 'lug_boot'].map(lug_boot_mapping)
        data.loc[:, 'safety'] = data.loc[:, 'safety'].map(safety_mapping)
    if label is not None:
        label_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        label = label.map(label_mapping)

    return data, label, data.columns.tolist()


def read_trainDataset(path):
    train_df = pd.read_csv(path)
    data = train_df.loc[:, :'safety']
    label = train_df.loc[:, 'class']
    data, label, attr_set = class_mapping(data, label)

    return data.values, label.values, attr_set


def read_testDataset(path):
    data = pd.read_csv(path)
    data, _, _ = class_mapping(data)

    return data.values


if __name__ == '__main__':

    X_data, label, attr_set = read_trainDataset('./data/car_train.csv')
    test_data = read_testDataset('./data/car_test.csv')
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_data, label, test_size=0.2, random_state=99)

    dtree = DecisionTree(attr_set=attr_set)
    cart_decision_tree = dtree.fit(X_train, Y_train)
    print(cart_decision_tree)
    plot_cart_tree(cart_decision_tree)
    predect_label = dtree.predict(X_dev)
    print(predect_label)
    acc = dtree.accuracy_score(Y_dev, predect_label)
    print('acc is {}'.format(acc))

    dtree = DecisionTree(attr_set=attr_set)
    cart_decision_tree = dtree.fit(X_data, label)
    plot_cart_tree(cart_decision_tree)
    predect_label = dtree.predict(test_data)
    df = pd.DataFrame({'id': [i_d for i_d in range(len(predect_label))], 'result': predect_label})
    df.to_csv('decesion_result.csv', encoding='utf-8', index=0)




