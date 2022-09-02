import datetime

class Condition:

    def __init__(self, attribute_name, threshold, greater):
        self.attribute_name = attribute_name
        self.threshold = threshold
        self.greater = greater

    def __str__(self):
        if self.greater:
            threshold_sign = ">"
        else:
            threshold_sign = "<="
        return "{attribute_name} {threshold_sign} {threshold}".format(attribute_name=self.attribute_name, threshold_sign=threshold_sign, threshold=self.threshold)

    def __repr__(self):
        if self.greater:
            threshold_sign = ">"
        else:
            threshold_sign = "<="
        return "{attribute_name} {threshold_sign} {threshold}".format(attribute_name=self.attribute_name, threshold_sign=threshold_sign, threshold=self.threshold)

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)

class Node:
    def __init__(self, node_id, n_samples, conditions, is_leaf):
        self.node_id = node_id
        self.n_samples = n_samples
        self.conditions = conditions
        self.is_leaf = is_leaf

    def __str__(self):
        if self.is_leaf:
            return "[{node_id}]- {n_samples}: {conditions}".format(node_id=self.node_id, n_samples=self.n_samples, conditions=self.conditions)
        return "({node_id})- {n_samples}: {conditions}".format(node_id=self.node_id, n_samples=self.n_samples, conditions=self.conditions)

    def __repr__(self):
        if self.is_leaf:
            return "[{node_id}] - {n_samples}: {conditions}".format(node_id=self.node_id, n_samples=self.n_samples, conditions=self.conditions)
        return "({node_id}) - {n_samples}: {conditions}".format(node_id=self.node_id, n_samples=self.n_samples, conditions=self.conditions)


class SurprisingInstance:

    def __init__(self, id, data, target_attribute, target_data, actual_data, leaf_id, categorical=False, conditions=[]):
        self.id = id
        self.data = data
        self.target_attribute = target_attribute
        self.target_data = target_data
        self.actual_data = actual_data
        self.leaf_id = leaf_id
        self.categorical = categorical
        self.conditions = conditions

    def calculateDifference(self):
        if self.target_data > self.actual_data:
            return round(self.target_data - self.actual_data, 2)
        return round(self.actual_data - self.target_data, 2)

    def convertToDatetime(self):
        return datetime.timedelta(seconds=round(self.calculateDifference(), 0))

    def convertActualDataToDatetime(self):
        return datetime.timedelta(seconds=round(self.actual_data, 0))

    def convertTargetDataToDatetime(self):
        return datetime.timedelta(seconds=round(self.target_data, 0))

    def __str__(self):
        return "Case {id} | leaf {leaf_id}: [{conditions}]".format(id=self.id, leaf_id=self.leaf_id ,conditions=self.conditions)

    def __repr__(self):
        return "Case {id} | leaf {leaf_id}: [{conditions}]".format(id=self.id, leaf_id=self.leaf_id ,conditions=self.conditions)