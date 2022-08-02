from sklearn import tree
from pm4py.visualization.decisiontree import visualizer as dectree_visualizer
import numpy as np
from models import Condition, Node
import matplotlib.pyplot as plt

def supervised_approach(event_log, descriptive_feature_names, target_feature_name, gamma, directory):
    vicinities, clf = vicinity_detecion_regression(event_log, descriptive_feature_names, target_feature_name)
    surprising_instances, surprising_instances_total, better_performing_instances_total, worse_performing_instances_total = surprising_instance_detection(vicinities, event_log, target_feature_name, gamma, directory)
    visualize_dt(clf, descriptive_feature_names, target_feature_name, directory, event_log)

    return surprising_instances, surprising_instances_total, better_performing_instances_total, worse_performing_instances_total

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
    stack = [(0, 0, [])]
    while len(stack) > 0:
        node_id, depth, conditions = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
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
    return pd_data_by_conditions

def vicinity_detecion_regression(event_log, descriptive_feature_names, target_feature_name):
    descriptive_attributes = event_log.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    target_feature_name_list = []
    target_feature_name_list.append(target_feature_name)
    target_attributes = event_log.filter(items=target_feature_name_list)
    target_feature_data = target_attributes.values.tolist()

    print("Start training DT")
    clf = tree.DecisionTreeRegressor(max_depth=5, min_samples_leaf=100)
    
    clf.fit(descriptive_feature_data, target_feature_data)

    vicinities = traverse_tree(clf, descriptive_feature_names)
    return vicinities, clf

def calculate_surprisingness(row, gamma, avg_s, avg_v_without_s, surprisingSize, vicinitySize):
    return gamma * abs(avg_s - avg_v_without_s) + (1-gamma) * (surprisingSize / vicinitySize)

def calculate_effectiveness_better(row, avg_v_without_s, avg_s, otherSize):
    return (avg_v_without_s - avg_s) * otherSize

def calculate_effectiveness_worse(row, avg_v_without_s, avg_s, surprisingSize):
    return (avg_s - avg_v_without_s) * surprisingSize

def find_outliers_for_node(filtered_data, target_feature_name, event_log, gamma):
    print('Target Feature Name: ' + str(target_feature_name))
    Q1 = filtered_data[target_feature_name].quantile(0.25)
    Q3 = filtered_data[target_feature_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 *IQR

    filter_better = (filtered_data[target_feature_name] < lower_bound)
    event_log_filter_better = filtered_data.loc[filter_better]
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_better.index)]
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    avg_s = event_log_filter_better[target_feature_name].mean()
    vicinitySize = len(filtered_data)
    surprisingSize = len(event_log_filter_better)
    if len(event_log_filter_better) > 0:
        event_log_filter_better['Surprisingness'] = event_log_filter_better.apply(lambda row: calculate_surprisingness(row=row, gamma=gamma, avg_s=avg_s, avg_v_without_s=p_avg, surprisingSize=surprisingSize, vicinitySize=vicinitySize), axis=1)
        event_log_filter_better['Effectiveness'] = event_log_filter_better.apply(lambda row: calculate_effectiveness_better(row=row, avg_v_without_s=p_avg, avg_s=avg_s, otherSize=(len(event_log) - len(event_log_filter_worse))), axis=1)
    print("There are " + str(len(event_log_filter_better)) + " better performing instances")

    filter_worse = (filtered_data[target_feature_name] > upper_bound)
    event_log_filter_worse = filtered_data.loc[filter_worse] 
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_worse.index)]
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    avg_s = event_log_filter_worse[target_feature_name].mean()
    vicinitySize = len(filtered_data)
    surprisingSize = len(event_log_filter_worse)
    if len(event_log_filter_worse) > 0:
        event_log_filter_worse['Surprisingness'] = event_log_filter_worse.apply(lambda row: calculate_surprisingness(row=row, gamma=gamma, avg_s=avg_s, avg_v_without_s=p_avg, surprisingSize=surprisingSize, vicinitySize=vicinitySize), axis=1)
        event_log_filter_worse['Effectiveness'] = event_log_filter_worse.apply(lambda row: calculate_effectiveness_worse(row=row, avg_v_without_s=p_avg, avg_s=avg_s, surprisingSize=surprisingSize), axis=1)
    print(event_log_filter_worse.head())
    print("There are " + str(len(event_log_filter_worse)) + " worse performing instances")

    return event_log_filter_better, event_log_filter_worse

def surprising_instance_detection(vicinities, event_log, target_feature_name, gamma, directory):
    surprising_instances = {}

    surprising_instances_total = 0
    better_performing_instances_total = 0
    worse_performing_instances_total = 0

    data_list = {}
    id = 1
    for node in vicinities:
        filtered_data = filter_data_for_conditions(event_log, node.conditions)
        filtered_data.to_csv(str(directory) + "/decisiontree/vicinity_" + str(node.node_id) + ".csv")
        case_id_col_list = filtered_data['@@case_id_column'].tolist()
        data_list[node.node_id] = case_id_col_list
        fig = plt.figure()
        ax = fig.gca()
        filtered_data['Throughput Time'].dt.days.plot.box(ax=ax)
        ax.set_ylabel('Case Duration (Days)')
        plt.savefig(str(directory) + '/decisiontree/boxplot_' + str(node.node_id) + 'case_duration.png')
        better_performing_instances, worse_performing_instances = find_outliers_for_node(filtered_data, target_feature_name, event_log, gamma)

        surprising_instances_total = surprising_instances_total + len(better_performing_instances)
        surprising_instances_total = surprising_instances_total + len(worse_performing_instances)
        better_performing_instances_total = better_performing_instances_total + len(better_performing_instances)
        worse_performing_instances_total = worse_performing_instances_total + len(worse_performing_instances)
        surprising_instances[node.node_id] = (node, better_performing_instances, worse_performing_instances, filtered_data)
        id = id + 1

    def get_cluster_for_row(row, dict):
        for key,value in dict.items():
            if row['@@case_id_column'] in value:
                return key
        return 'NOT FOUND'

    event_log['cluster_id'] = event_log.apply(lambda row: get_cluster_for_row(row, data_list), axis=1)
    print(event_log)

    event_log['days'] = event_log['Throughput Time'].dt.days

    fig = plt.figure()
    ax = fig.gca()
    event_log.boxplot(ax=ax, column=['days'], by=['cluster_id'])
    ax.set_ylabel('Case Duration (Days)')
    plt.savefig(str(directory) + '/decisiontree/boxplot_case_duration_all_clusters.png')
    
    return surprising_instances, surprising_instances_total, better_performing_instances_total, worse_performing_instances_total

def visualize_dt(clf, descriptive_feature_names, target_feature_name, directory, event_log):
    target_feature_name_list = []
    target_feature_name_list.append(target_feature_name)
    target_attributes = event_log.filter(items=target_feature_name_list)
    target_feature_data = target_attributes.values.tolist()

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)

    new_feature_names = [] 
    for feature_name in descriptive_feature_names:
        feature_name = feature_name.replace('<', 'l')
        feature_name = feature_name.replace('>', 'g')
        new_feature_names.append(feature_name)

    # Visualize DT
    classes = []
    for item in unique:
        classes.append(str(item))
    gviz = dectree_visualizer.apply(clf, new_feature_names, classes)
    dectree_visualizer.save(gviz, str(directory) + '/decisiontree/decision_tree.png')
