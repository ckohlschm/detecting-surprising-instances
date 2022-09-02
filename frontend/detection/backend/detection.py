import numpy as np
import random
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from pm4py.objects.log.util import get_class_representation
from pm4py.visualization.decisiontree import visualizer as dectree_visualizer


from .models import SurprisingInstance, Condition, Node

def train_dt_classifier(pd_data, descriptive_feature_names, target_feature_name, maxDepth):
    # Filter descriptive data
    descriptive_attributes = pd_data.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    # Filter target data
    target_attributes = pd_data.filter(items=target_feature_name)
    target_feature_name = target_attributes.columns.tolist()
    target_feature_data = target_attributes.values.tolist()

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)
    print('Target Classes: ' + str(dict(zip(unique, counts))) + '|MaxDepth: ' + str(maxDepth))

    # Create DT Classifier
    #clf = RandomForestClassifier(max_depth=7)
    clf = tree.DecisionTreeClassifier(max_depth=maxDepth, random_state=11)
    
    clf.fit(descriptive_feature_data, target_feature_data)
    return clf, unique

def find_conditions_for_instance(clf, attributes_list, descriptive_feature_names):
    conditions = []
    
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    decision_path = clf.decision_path(attributes_list)
    leaf_id = clf.apply(attributes_list)
    
    #print('Leaf ID: ' + str(leaf_id))
    #print('Number of nodes in leaf: ' + str(clf.tree_.n_node_samples[leaf_id]))
    #print(decision_path)
    
    # Follow the decision path for the the surprising instance
    sample_id = 0
    node_index = decision_path.indices[decision_path.indptr[sample_id] : decision_path.indptr[sample_id + 1]]
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if float(attributes_list[0][feature[node_id]]) <= float(threshold[node_id]):
            threshold_sign = "<="
            greater = False
        else:
            threshold_sign = ">"
            greater = True
        
        condition = Condition(attribute_name=descriptive_feature_names[feature[node_id]], threshold=threshold[node_id], greater=greater)
        conditions.append(condition)
        #print(
        #    "decision node {node} : (Feature: {featurename}[{sample}, {feature}] = {value}) "
        #    "{inequality} {threshold})".format(
        #        node=node_id,
        #        featurename=descriptive_feature_names[feature[node_id]],
        #        sample=sample_id,
        #        feature=feature[node_id],
        #        value=attributes_list[0][feature[node_id]],
        #        inequality=threshold_sign,
        #        threshold=threshold[node_id],
        #    )
        #)

    return conditions



def classify_dt(pd_data_event_log, descriptive_feature_names, target_feature_name, maxDepth, add_conditions):
    #parameters["add_case_identifier_column"] = True
    #parameters["enable_case_duration"] = True
    #parameters["enable_succ_def_representation"] = False
    if '@@caseDuration' in descriptive_feature_names:
        descriptive_feature_names.remove('@@caseDuration')
    # Filter descriptive data
    descriptive_attributes = pd_data_event_log.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    # Filter target data
    target_feature_name_list = []
    target_feature_name_list.append(target_feature_name)
    target_attributes = pd_data_event_log.filter(items=target_feature_name_list)
    target_feature_name = target_attributes.columns.tolist()
    target_feature_data = target_attributes.values.tolist()

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)
    #print('Target Classes: ' + str(dict(zip(unique, counts))))

    #pd_data, feature_names = extract_event_log_features(event_log, parameters)
    #print("Chosen lower threshold: " + str(lower_threshold) + " upper threshold: " + str(upper_threshold))
    #if add_delay:
    #    pd_data = add_delay(pd_data=pd_data, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
    #    feature_names.append('@@delay')

    #if feature_extracion == 'apriori':
    #    selected_features_for_apriori = descriptive_feature_names
    #    selected_features_for_apriori.append('@@delay')
    #    pd_data_filtered = pd_data.filter(items=selected_features_for_apriori)
#
    #    df_binary = _make_df_binary(pd_data_filtered, 10, '_')
    #    print(df_binary.head())
    #    frequent_itemsets = apriori(df_binary, min_support= 0.8, use_colnames=True)
    #    print('Done frequent itemsets: ' + str(len(frequent_itemsets)))
    #    mined_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
    #    mined_rules["antecedent_len"] = mined_rules["antecedents"].apply(lambda x: len(x))
    #    rules_for_target_attribute = mined_rules[mined_rules['consequents'] == {'@@delay_True'}]
    #    print(rules_for_target_attribute.head())
    #    rules_for_target_attribute = mined_rules[mined_rules['consequents'] == {'@@delay_False'}]
    #    print(rules_for_target_attribute.head())
    #elif feature_extracion == 'pearson':
    #    correlation = pd_data.corr(method='pearson')
    #    print(correlation)
    #    print(correlation['@@caseDuration'])
    #elif num_selected_attributes > 0:
    #    descriptive_feature_names_for_first_tree = random.sample(descriptive_feature_names, num_selected_attributes)
    #    print("Chosen Features: " + str(descriptive_feature_names_for_first_tree))
    #else:
    #    descriptive_feature_names_for_first_tree = descriptive_feature_names.copy()
    #    print("Choosing all features")

    clf, unique = train_dt_classifier(pd_data=pd_data_event_log, descriptive_feature_names=descriptive_feature_names, target_feature_name=target_feature_name, maxDepth=maxDepth)
    
    surprising_instances = []
    case_data = pd_data_event_log.values.tolist()
    features = pd_data_event_log.columns.tolist()

    leaf_nodes = {}
    leaf_node_ids = []

    for case in case_data:
        case_feature_data = pd.DataFrame([case], columns=features)
        descriptive_attributes = case_feature_data.filter(items=descriptive_feature_names)
        attributes_list = descriptive_attributes.values.tolist()
        result = clf.predict(attributes_list)
        target_feature_index = features.index(target_feature_name[0])
        # Predicted value is different than actual value
        if result != case[target_feature_index]:
            # Count how many surprising instances are in leaf nodes
            leaf_id = clf.apply(attributes_list)
            if str(leaf_id) in leaf_nodes:
                leaf_nodes[str(leaf_id)].append(case[0])
            else: 
                leaf_node_ids.append(leaf_id)
                leaf_nodes[str(leaf_id)] = []
                leaf_nodes[str(leaf_id)].append(case[0])
            conditions = []
            if add_conditions:
                conditions = find_conditions_for_instance(clf, attributes_list, descriptive_feature_names)
            #print(conditions)
            instance = SurprisingInstance(case[0], case, target_feature_name[0], result[0], case[target_feature_index], leaf_id[0], True, conditions)
            surprising_instances.append(instance)

    leaf_map = {}
    # For evaluation of all instances in leafs
    np_leaf_node_ids = np.array(leaf_node_ids)
    for leaf_node_id in np.unique(np_leaf_node_ids):
        leaf_string = '[' + str(leaf_node_id) + ']'
        total_num_instances_in_leaf = clf.tree_.n_node_samples[leaf_node_id]
        falsely_classified_instances = len(leaf_nodes[leaf_string])
        correctly_classified_instances = total_num_instances_in_leaf - falsely_classified_instances
        ratio = falsely_classified_instances / correctly_classified_instances
        leaf_map[str(leaf_node_id)] = (correctly_classified_instances, falsely_classified_instances, ratio)
        #print('Surprising instances in leaf ' + str(leaf_node_id) + ': (' + str(falsely_classified_instances) + ' : ' + str(correctly_classified_instances) + ' - ' + str(leaf_nodes[leaf_string]) + ')')

    sorted_leaf_list = sorted(leaf_map.items(), key=lambda item: item[1][2])
    #for k,v in sorted_leaf_list:
    #    print("Leaf: " + str(k) + " (" + str(v[0]) + "/" + str(v[1]) + "): " + str(v[2]))

    #print(sorted_leaf_list[0])
    
    #first_instance_to_investigate = None
    #second_instance_to_investigate = None

    #for instance in surprising_instances:
    #    print(instance)
    #    if str(instance.leaf_id) == sorted_leaf_list[0][0]:
    #        print("First instance: " + str(instance))
    #        first_instance_to_investigate = instance
    #        break

    #for instance in surprising_instances:
    #    if instance.leaf_id == sorted_leaf_list[1][0]:
    #        print("Second instance: " + str(instance))
    #        second_instance_to_investigate = instance
    #        break

    # Visualize DT
    classes = []
    for item in unique:
        classes.append(str(item))

    # graphviz does not like special characters
    new_feature_names = [] 
    for feature_name in descriptive_feature_names:
        feature_name = feature_name.replace('<', 'l')
        feature_name = feature_name.replace('>', 'g')
        new_feature_names.append(feature_name)

    gviz = dectree_visualizer.apply(clf, new_feature_names, classes)
    decision_tree_output_path = 'detection/static/detection/figures/decision_tree.png'
    dectree_visualizer.save(gviz, decision_tree_output_path)
    #dectree_visualizer.view(gviz)

    return surprising_instances

def traverse_tree(clf, feature_names):
    nodes = []

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    n_node_samples = clf.tree_.n_node_samples

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0, [])]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth, conditions = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            node = Node(node_id=node_id, n_samples=n_node_samples[node_id], conditions=conditions, is_leaf=False)
            # nodes.append(node)
            conditions_left = conditions.copy()
            conditions_left.append(Condition(attribute_name=feature_names[feature[node_id]], threshold=threshold[node_id], greater=False))
            conditions_right = conditions.copy()
            conditions_right.append(Condition(attribute_name=feature_names[feature[node_id]], threshold=threshold[node_id], greater=True))
            stack.append((children_left[node_id], depth + 1, conditions_left))
            stack.append((children_right[node_id], depth + 1, conditions_right))
        else:
            node = Node(node_id=node_id, n_samples=n_node_samples[node_id], conditions=conditions, is_leaf=True)
            nodes.append(node)
            is_leaves[node_id] = True

    #print(
    #    "The binary tree structure has {n} nodes and has "
    #    "the following tree structure:\n".format(n=n_nodes)
    #)
    #for i in range(n_nodes):
    #    if is_leaves[i]:
    #        print(
    #            "{space}node={node} is a leaf node.".format(
    #                space=node_depth[i] * "\t", node=i
    #            )
    #        )
    #    else:
    #        print(
    #            "{space}node={node} is a split node: "
    #            "go to node {left} if X[:, {feature}] <= {threshold} "
    #            "else to node {right}.".format(
    #                space=node_depth[i] * "\t",
    #                node=i,
    #                left=children_left[i],
    #                feature=feature_names[i],
    #                threshold=threshold[i],
    #                right=children_right[i],
    #            )
    #        )
    return nodes


def filter_data_for_conditions(all_data, conditions):
    query_string = ""
    for condition in conditions:
        query_string += "`" + str(condition.attribute_name) + "`"
        if condition.greater:
            query_string += " > "
        else:
            query_string += " <= "
        query_string += str(condition.threshold)
        query_string += " & "
    query_string = query_string.rsplit('&', 1)[0]
    print("Query string: " + query_string)
    if query_string == "":
        pd_data_by_conditions = all_data.copy()
    else:
        pd_data_by_conditions = all_data.query(query_string)
    #print(pd_data_by_conditions.head())
    return pd_data_by_conditions

def calculate_surprisingness_index_better(row, target_feature_name, p_avg, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * affectedInstances 

def calculate_surprisingness_index_worse(row, target_feature_name, p_avg, vicinitySize, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * (vicinitySize - affectedInstances)

def calculate_relevance_worse(row, vicinitySize):
    return row['surprisingnessWorseIndex'] * vicinitySize

def calculate_relevance_better(row, vicinitySize):
    return row['surprisingnessBetterIndex'] * vicinitySize

def find_outliers_for_node_threshold(filtered_data, target_feature_name, conditions, descriptive_feature_names, clf, threshold):
    mean_value = filtered_data[target_feature_name].mean()

    lower_bound = mean_value - threshold
    upper_bound = mean_value + threshold

    filter_better = (filtered_data[target_feature_name] < lower_bound)
    event_log_filter_better = filtered_data.loc[filter_better]
    surprising_instances_better = []
    if len(event_log_filter_better) > 0:
        event_log_data_better = event_log_filter_better.values.tolist()
        event_log_features_better = event_log_filter_better.columns.tolist()
        for case in event_log_data_better:
            case_feature_data = pd.DataFrame([case], columns=event_log_features_better)
            descriptive_attributes = case_feature_data.filter(items=descriptive_feature_names)
            descriptive_attributes_list = descriptive_attributes.values.tolist()
            target_feature_index = event_log_features_better.index(target_feature_name)
            leaf_id = clf.apply(descriptive_attributes_list)
            instance = SurprisingInstance(case[0], case, target_feature_name, mean_value, case[target_feature_index], 0, False, conditions)
            surprising_instances_better.append(instance)

        other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_better.index)]
        #print("All: " + str(len(filtered_data))+ " Better: " + str(len(event_log_filter_better)) + " Other: " + str(len(other_instances_in_vicinity)))
        p_avg = other_instances_in_vicinity[target_feature_name].mean()
        affectedInstances = len(event_log_filter_better)
        vicinitySize = len(filtered_data)
        event_log_filter_better['surprisingnessBetterIndex'] = event_log_filter_better.apply(lambda row: calculate_surprisingness_index_better(row=row, target_feature_name=target_feature_name, p_avg= p_avg, affectedInstances=affectedInstances), axis=1)
        event_log_filter_better['RelevanceIndex'] = event_log_filter_better.apply(lambda row: calculate_relevance_better(row=row, vicinitySize=vicinitySize), axis=1)
    print("There are " + str(len(event_log_filter_better)) + " better performing instances")

    filter_worse = (filtered_data[target_feature_name] > upper_bound)
    event_log_filter_worse = filtered_data.loc[filter_worse]
    surprising_instances_worse = []
    if len(event_log_filter_worse) > 0:
        event_log_data_worse = event_log_filter_worse.values.tolist()
        event_log_features_worse = event_log_filter_worse.columns.tolist()
        for case in event_log_data_worse:
            case_feature_data = pd.DataFrame([case], columns=event_log_features_worse)
            descriptive_attributes = case_feature_data.filter(items=descriptive_feature_names)
            descriptive_attributes_list = descriptive_attributes.values.tolist()
            target_feature_index = event_log_features_worse.index(target_feature_name)
            leaf_id = clf.apply(descriptive_attributes_list)
            instance = SurprisingInstance(case[0], case, target_feature_name, mean_value, case[target_feature_index], leaf_id[0], False, conditions)
            surprising_instances_worse.append(instance)
        
        other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_worse.index)]
        #print("All: " + str(len(filtered_data))+ " Worse: " + str(len(event_log_filter_worse)) + " Other: " + str(len(other_instances_in_vicinity)))
        #print("All: " + str(filtered_data[target_feature_name].mean())+ " Worse: " + str(event_log_filter_worse[target_feature_name].mean()) + " Other: " + str(other_instances_in_vicinity[target_feature_name].mean()))
        p_avg = other_instances_in_vicinity[target_feature_name].mean()
        affectedInstances = len(event_log_filter_worse)
        vicinitySize = len(filtered_data)
        event_log_filter_worse['surprisingnessWorseIndex'] = event_log_filter_worse.apply(lambda row: calculate_surprisingness_index_worse(row=row, target_feature_name=target_feature_name, p_avg= p_avg, vicinitySize=vicinitySize, affectedInstances=affectedInstances), axis=1)
        event_log_filter_worse['RelevanceIndex'] = event_log_filter_worse.apply(lambda row: calculate_relevance_worse(row=row, vicinitySize=vicinitySize), axis=1)
    print("There are " + str(len(event_log_filter_worse)) + " worse performing instances")

    return event_log_filter_better, event_log_filter_worse, surprising_instances_better, surprising_instances_worse


def find_outliers_for_node(filtered_data, target_feature_name, conditions, descriptive_feature_names, clf):
    Q1 = filtered_data[target_feature_name].quantile(0.25)
    Q3 = filtered_data[target_feature_name].quantile(0.75)
    IQR = Q3 - Q1

    mean_value = filtered_data[target_feature_name].mean()

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 *IQR

    filter_better = (filtered_data[target_feature_name] < lower_bound)
    event_log_filter_better = filtered_data.loc[filter_better]
    surprising_instances_better = []
    if len(event_log_filter_better) > 0:
        event_log_data_better = event_log_filter_better.values.tolist()
        event_log_features_better = event_log_filter_better.columns.tolist()
        for case in event_log_data_better:
            case_feature_data = pd.DataFrame([case], columns=event_log_features_better)
            descriptive_attributes = case_feature_data.filter(items=descriptive_feature_names)
            descriptive_attributes_list = descriptive_attributes.values.tolist()
            target_feature_index = event_log_features_better.index(target_feature_name)
            leaf_id = clf.apply(descriptive_attributes_list)
            instance = SurprisingInstance(case[0], case, target_feature_name, mean_value, case[target_feature_index], 0, False, conditions)
            surprising_instances_better.append(instance)

        other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_better.index)]
        #print("All: " + str(len(filtered_data))+ " Better: " + str(len(event_log_filter_better)) + " Other: " + str(len(other_instances_in_vicinity)))
        p_avg = other_instances_in_vicinity[target_feature_name].mean()
        affectedInstances = len(event_log_filter_better)
        vicinitySize = len(filtered_data)
        event_log_filter_better['surprisingnessBetterIndex'] = event_log_filter_better.apply(lambda row: calculate_surprisingness_index_better(row=row, target_feature_name=target_feature_name, p_avg= p_avg, affectedInstances=affectedInstances), axis=1)
        event_log_filter_better['RelevanceIndex'] = event_log_filter_better.apply(lambda row: calculate_relevance_better(row=row, vicinitySize=vicinitySize), axis=1)
    print("There are " + str(len(event_log_filter_better)) + " better performing instances")

    filter_worse = (filtered_data[target_feature_name] > upper_bound)
    event_log_filter_worse = filtered_data.loc[filter_worse]
    surprising_instances_worse = []
    if len(event_log_filter_worse) > 0:
        event_log_data_worse = event_log_filter_worse.values.tolist()
        event_log_features_worse = event_log_filter_worse.columns.tolist()
        for case in event_log_data_worse:
            case_feature_data = pd.DataFrame([case], columns=event_log_features_worse)
            descriptive_attributes = case_feature_data.filter(items=descriptive_feature_names)
            descriptive_attributes_list = descriptive_attributes.values.tolist()
            target_feature_index = event_log_features_worse.index(target_feature_name)
            leaf_id = clf.apply(descriptive_attributes_list)
            instance = SurprisingInstance(case[0], case, target_feature_name, mean_value, case[target_feature_index], leaf_id[0], False, conditions)
            surprising_instances_worse.append(instance)
        
        other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_worse.index)]
        #print("All: " + str(len(filtered_data))+ " Worse: " + str(len(event_log_filter_worse)) + " Other: " + str(len(other_instances_in_vicinity)))
        #print("All: " + str(filtered_data[target_feature_name].mean())+ " Worse: " + str(event_log_filter_worse[target_feature_name].mean()) + " Other: " + str(other_instances_in_vicinity[target_feature_name].mean()))
        p_avg = other_instances_in_vicinity[target_feature_name].mean()
        affectedInstances = len(event_log_filter_worse)
        vicinitySize = len(filtered_data)
        event_log_filter_worse['surprisingnessWorseIndex'] = event_log_filter_worse.apply(lambda row: calculate_surprisingness_index_worse(row=row, target_feature_name=target_feature_name, p_avg= p_avg, vicinitySize=vicinitySize, affectedInstances=affectedInstances), axis=1)
        event_log_filter_worse['RelevanceIndex'] = event_log_filter_worse.apply(lambda row: calculate_relevance_worse(row=row, vicinitySize=vicinitySize), axis=1)
    print("There are " + str(len(event_log_filter_worse)) + " worse performing instances")

    return event_log_filter_better, event_log_filter_worse, surprising_instances_better, surprising_instances_worse

def dt_regression(pd_data_event_log, descriptive_feature_names, target_feature_name, add_conditions, threshold, max_depth, detector_function, directory=None):
    data_by_vicinity_id = {}
    # Filter descriptive data
    descriptive_attributes = pd_data_event_log.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    # Filter target data
    target_feature_name_list = []
    target_feature_name_list.append(target_feature_name)
    target_attributes = pd_data_event_log.filter(items=target_feature_name_list)
    target_feature_name = target_attributes.columns.tolist()
    target_feature_data = target_attributes.values.tolist()

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)
    #print('Target Classes: ' + str(dict(zip(unique, counts))))

    # Create DT Regressor
    clf = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=11, min_samples_leaf=100)
    
    clf.fit(descriptive_feature_data, target_feature_data)
   
    nodes = traverse_tree(clf, descriptive_feature_names)

    surprising_instances = {}

    for node in nodes:
        filtered_data = filter_data_for_conditions(pd_data_event_log, node.conditions)
        data_by_vicinity_id[node.node_id] = filtered_data
        if detector_function == 'boxplot':
            better_performing_instances, worse_performing_instances, surprising_instnaces_better, surprising_instances_worse = find_outliers_for_node(filtered_data, target_feature_name[0], node.conditions, descriptive_feature_names, clf)
        else:
            better_performing_instances, worse_performing_instances, surprising_instnaces_better, surprising_instances_worse = find_outliers_for_node_threshold(filtered_data, target_feature_name[0], node.conditions, descriptive_feature_names, clf, threshold)
        surprising_instances[node.node_id] = (node, better_performing_instances, worse_performing_instances, surprising_instnaces_better, surprising_instances_worse)

    # Visualize DT
    classes = []
    for item in unique:
        classes.append(str(item))
        #print('Classes: ' + str(classes))
    
    # graphviz does not like special characters
    new_feature_names = [] 
    for feature_name in descriptive_feature_names:
        feature_name = feature_name.replace('<', 'l')
        feature_name = feature_name.replace('>', 'g')
        new_feature_names.append(feature_name)
    
    gviz = dectree_visualizer.apply(clf, new_feature_names, classes)
    if directory:
        decision_tree_output_path = str(directory) + '/decision_tree.png'
    else:
        decision_tree_output_path = 'detection/static/detection/figures/decision_tree.png'
    dectree_visualizer.save(gviz, decision_tree_output_path)
    #dectree_visualizer.view(gviz)

    return surprising_instances, data_by_vicinity_id

def find_similar_cases(event_log, case, method):
    if method == 'cluster':
        # clustering
        pd_filtered = event_log.filter(feature_list)
        X = pd_filtered.values.tolist()
        kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        print('Target Classes: ' + str(dict(zip(unique, counts))))
    elif method == 'levenshtein':
        print('Levenshtein')

def nn_regression(event_log, descriptive_feature_names, target_feature_name, add_conditions, threshold):
    # Filter descriptive data
    descriptive_attributes = event_log.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    # Filter target data
    target_feature_name_list = []
    target_feature_name_list.append(target_feature_name)
    target_attributes = event_log.filter(items=target_feature_name_list)
    target_feature_name = target_attributes.columns.tolist()
    target_feature_data = target_attributes.values.tolist()

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)
    #print('Target Classes: ' + str(dict(zip(unique, counts))))

    # Create MLP Regressor
    clf = MLPRegressor()
    
    clf.fit(descriptive_feature_data, target_feature_data)
   
    surprising_instances = []
    all_event_data = event_log.values.tolist()
    all_features = event_log.columns.tolist()

    for case in all_event_data:
        case_feature_data = pd.DataFrame([case], columns=all_features)
        descriptive_attributes = case_feature_data.filter(items=descriptive_feature_names)
        descriptive_attributes_list = descriptive_attributes.values.tolist()
        result = clf.predict(descriptive_attributes_list)
        target_feature_index = all_features.index(target_feature_name[0])
        # Predicted value is very different than actual value
        if result >= case[target_feature_index]+threshold or result <= case[target_feature_index]-threshold:
            #print("Predicted: " +str(result) + " Actual: " + str(case[target_feature_index]))
            conditions = []
            if add_conditions:
                conditions = find_conditions_for_instance(clf, descriptive_attributes_list, descriptive_feature_names)
            #print(conditions)
            instance = SurprisingInstance(case[0], case, target_feature_name[0], case[target_feature_index], None, conditions)
            surprising_instances.append(instance)

    print("There are: " + str(len(surprising_instances)) + " Surprising Instances")

    if add_conditions:
        method = 'cluster'
        for instance in surprising_instances:
            find_similar_cases(event_log, instance, method)

    return surprising_instances

def classify_nn(event_log, descriptive_feature_names, target_feature_name, parameters):
    parameters["add_case_identifier_column"] = True

    data, feature_names = log_to_features.apply(event_log, variant=log_to_features.Variants.TRACE_BASED, parameters = parameters)#, "enable_case_duration" : True})
    print(feature_names)

    #if descriptive_feature_names not in feature_names or target_feature_name not in feature_names:
    #    print('Cannot find feature in event log!')
    #   return []

    # Filter descriptive data
    pd_data = pd.DataFrame(data, columns=feature_names)
    descriptive_attributes = pd_data.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    # Filter target data
    pd_data = pd.DataFrame(data, columns=feature_names)
    target_attributes = pd_data.filter(items=target_feature_name)
    target_feature_name = target_attributes.columns.tolist()
    target_feature_data = target_attributes.values.tolist()
    
    #target, classes = get_class_representation.get_class_representation_by_trace_duration(event_log, 58 * 86400)
    #print(classes)

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)
    print('Target Classes: ' + str(dict(zip(unique, counts))))

    # Create NN Classifier
    clf = MLPClassifier()
    clf.fit(descriptive_feature_data, np_target)

    surprising_instances = []
    surprising_classes = []
    for case in data:
        pd_data = pd.DataFrame([case], columns=feature_names)
        #print(pd_data)
        descriptive_attributes = pd_data.filter(items=descriptive_feature_names)
        #print(descriptive_attributes.values.tolist())
        descriptive_attributes_list = descriptive_attributes.values.tolist()
        result = clf.predict(descriptive_attributes_list)
        target_feature_index = feature_names.index(target_feature_name[0])
        if result != case[target_feature_index]:
            surprising_instances.append(case)
            print(case[0])
            if len(surprising_classes) == 0:
                surprising_classes.append([])
                surprising_classes[0].append(descriptive_attributes_list)
            else:
                added = False
                for classes in surprising_classes:
                    for i in range(len(classes[0])):
                        if classes[0][i] == descriptive_attributes_list[i]:
                            classes.append(descriptive_attributes_list)
                            added = True
                if not added:
                    surprising_classes.append([])
                    surprising_classes[-1].append(descriptive_attributes_list)
            
    print(surprising_classes)
    return surprising_instances

def classify_nb(event_log, descriptive_feature_names, target_feature_name, parameters):
    parameters["add_case_identifier_column"] = True

    data, feature_names = log_to_features.apply(event_log, variant=log_to_features.Variants.TRACE_BASED, parameters = parameters)#, "enable_case_duration" : True})
    print(feature_names)

    #if descriptive_feature_names not in feature_names or target_feature_name not in feature_names:
    #    print('Cannot find feature in event log!')
    #   return []

    # Filter descriptive data
    pd_data = pd.DataFrame(data, columns=feature_names)
    descriptive_attributes = pd_data.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    # Filter target data
    pd_data = pd.DataFrame(data, columns=feature_names)
    target_attributes = pd_data.filter(items=target_feature_name)
    target_feature_name = target_attributes.columns.tolist()
    target_feature_data = target_attributes.values.tolist()
    
    #target, classes = get_class_representation.get_class_representation_by_trace_duration(event_log, 58 * 86400)
    #print(classes)

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)
    print('Target Classes: ' + str(dict(zip(unique, counts))))

    # Create NB Classifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(descriptive_feature_data, np_target)

    surprising_instances = []
    for case in data:
        pd_data = pd.DataFrame([case], columns=feature_names)
        #print(pd_data)
        descriptive_attributes = pd_data.filter(items=descriptive_feature_names)
        #print(descriptive_attributes.values.tolist())
        result = clf.predict(descriptive_attributes.values.tolist())
        target_feature_index = feature_names.index(target_feature_name[0])
        if result != case[target_feature_index]:
            surprising_instances.append(case)

    return surprising_instances

def detect_surprising_instances(pd_data_event_log, descriptive_feature_names, target_feature_name, strategy, add_conditions, threshold, max_depth, detector_function, directory=None):
    if strategy == 'dt_regression':
        return dt_regression(pd_data_event_log, descriptive_feature_names, target_feature_name, add_conditions, threshold, max_depth, detector_function, directory)
    elif strategy == 'dt_classification':
        return classify_dt(pd_data_event_log, descriptive_feature_names, target_feature_name, max_depth, add_conditions)
    else:
        print('Undefined strategy')
        return []
