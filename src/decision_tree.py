import math
from operator import itemgetter


class DecisionTree:
    def __init__(self, data, target, positive, parent_value='', parent=None):
        self.data = data
        self.target = target
        self.positive = positive
        self.parent_value = parent_value
        self.parent = parent
        self.decision = None
        self.children = []

        self.columns = [col for col in self.data.columns if col != self.target]
        self.entropy = self._entropy(self.data)
        if self.entropy != 0 and len(self.data.columns) > 1:
            self.gains = [(column, self._information_gain(column)) for column in self.columns]
            self.split_column = max(self.gains, key=itemgetter(1))[0]
            columns_left = [k for k in self.data.columns if k != self.split_column]
            for val in self.data[self.split_column].unique():
                df_tmp = self.data[self.data[self.split_column] == val][columns_left]
                tmp_node = DecisionTree(df_tmp, self.target, self.positive, val, self)
                self.children.append(tmp_node)
        else:
            self.decision = self.data[self.target].mode()[0]
            #self.data[self.target].iloc[0]

    # Расчет энтропии
    def _entropy(self, data):
        p = sum(data[self.target] == self.positive)
        n = data.shape[0] - p
        p_ratio = p / (p + n)
        n_ratio = 1 - p_ratio
        entropy_p = -p_ratio * math.log2(p_ratio) if p_ratio != 0 else 0
        entropy_n = -n_ratio * math.log2(n_ratio) if n_ratio != 0 else 0
        return entropy_p + entropy_n

    # Информативность
    def _information_gain(self, feat):
        avg_info = 0
        for val in self.data[feat].unique():
            avg_info += self._entropy(self.data[self.data[feat] == val]) * sum(self.data[feat] == val) / \
                        self.data.shape[0]
        return self._entropy(self.data) - avg_info

    def predict(self, data):
        if self.decision:
            return self.decision

        next_node = None
        for child in self.children:
            if child.parent_value == data[child.parent.split_column]:
                next_node = child

        if next_node:
            decision = next_node.predict(data)
        else:
            decision = self._trivial_decision()

        return decision

    def _trivial_decision(self):
        return self.data[self.target].mode()[0]
