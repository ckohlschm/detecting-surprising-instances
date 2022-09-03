import numpy as np
import pandas as pd
from sklearn import tree
from pm4py.visualization.decisiontree import visualizer as dectree_visualizer

from .data_reader import transform_log_to_feature_table
from .root_cause_analysis import find_cause_for_instance
from .detection_util import calculate_surprising_instance_statistics, filter_results_by_vicinity_id, import_and_filter_event_log, get_len_surprising_instances, read_session_parameters, transform_event_log_to_situations
from .models import SurprisingInstance, Condition, Node


def apply_supervised_learning(request, context):
    if 'leaf_select' in request.POST:
        selected_leaf_id = request.POST['leaf_select']
        print('Selected leaf: ' + str(selected_leaf_id))
        request.session['selected_leaf_id'] = selected_leaf_id
    
    context, event_log, surprising_instances_len = detect_surprising_instances(request, context)
    context = calculate_surprising_instance_statistics(event_log, surprising_instances_len, context)
    return context

def label_surprising_better(row, surprising_instances):
    for instance in surprising_instances:
        if row['@@case_id_column'] == instance.id:
            if instance.actual_data < instance.target_data:
                return True
    return False

def label_surprising_worse(row, surprising_instances):
    for instance in surprising_instances:
        if row['@@case_id_column'] == instance.id:
            if instance.actual_data > instance.target_data:
                return True
    return False

def root_cause_analysis_dt(surprising_instances, selected_leaf_id, pd_data_event_log, descriptive_feature_names, target_feature_name):
    pd_data_event_log['@@surprising_better'] = pd_data_event_log.apply (lambda row: label_surprising_better(row, surprising_instances), axis=1)
    pd_data_event_log['@@surprising_worse'] = pd_data_event_log.apply (lambda row: label_surprising_worse(row, surprising_instances), axis=1) 
    
    # Root cause analysis
    for instance in surprising_instances:
        if str(instance.leaf_id) == str(selected_leaf_id):
            print("Conditions: " + str(instance.conditions))
            find_cause_for_instance(pd_data_event_log, instance, ['@@surprising_better'], 'dt_rca_better')
            find_cause_for_instance(pd_data_event_log, instance, ['@@surprising_worse'], 'dt_rca_worse')
            break

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
    clf = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=11)
    
    clf.fit(descriptive_feature_data, target_feature_data)
   
    nodes = traverse_tree(clf, descriptive_feature_names)

    surprising_instances = {}
    data_by_vicinity_id = {}

    for node in nodes:
        filtered_data = filter_data_for_conditions(pd_data_event_log, node.conditions)
        data_by_vicinity_id[node.node_id] = filtered_data
        if detector_function == 'boxplot':
            better_performing_instances, worse_performing_instances, surprising_instances_better, surprising_instances_worse = find_outliers_for_node(filtered_data, target_feature_name[0], node.conditions, descriptive_feature_names, clf)
        else:
            better_performing_instances, worse_performing_instances, surprising_instances_better, surprising_instances_worse = find_outliers_for_node_threshold(filtered_data, target_feature_name[0], node.conditions, descriptive_feature_names, clf, threshold)
        surprising_instances[node.node_id] = (node, better_performing_instances, worse_performing_instances, surprising_instances_better, surprising_instances_worse)

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

    return surprising_instances, data_by_vicinity_id

def find_conditions_for_instance(clf, attributes_list, descriptive_feature_names):
    conditions = []
    
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    decision_path = clf.decision_path(attributes_list)
    leaf_id = clf.apply(attributes_list)

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

    return conditions

def find_outliers_for_node_categorical(filtered_data, target_feature_name, conditions, descriptive_feature_names, clf):
    expected_value = filtered_data[target_feature_name].value_counts().index.tolist()[0]
    print('Expected value in vicinity: ' + str(expected_value))

    filter_better = (filtered_data[target_feature_name] != expected_value)
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
            instance = SurprisingInstance(case[0], case, target_feature_name, expected_value, case[target_feature_index], leaf_id, True, conditions)
            surprising_instances_better.append(instance)

        other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_better.index)]
        #print("All: " + str(len(filtered_data))+ " Better: " + str(len(event_log_filter_better)) + " Other: " + str(len(other_instances_in_vicinity)))
    print("There are " + str(len(event_log_filter_better)) + " better performing instances")

    event_log_filter_worse = None
    surprising_instances_worse = []

    return event_log_filter_better, event_log_filter_worse, surprising_instances_better, surprising_instances_worse

def classify_dt(pd_data_event_log, descriptive_feature_names, target_feature_name, maxDepth, add_conditions):
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

    clf = tree.DecisionTreeClassifier(max_depth=maxDepth, random_state=11)
    clf.fit(descriptive_feature_data, target_feature_data)

    nodes = traverse_tree(clf, descriptive_feature_names)

    surprising_instances = {}
    data_by_vicinity_id = {}

    for node in nodes:
        filtered_data = filter_data_for_conditions(pd_data_event_log, node.conditions)
        data_by_vicinity_id[node.node_id] = filtered_data
        better_performing_instances, worse_performing_instances, surprising_instances_better, surprising_instances_worse = find_outliers_for_node_categorical(filtered_data, target_feature_name[0], node.conditions, descriptive_feature_names, clf)
        
        surprising_instances[node.node_id] = (node, better_performing_instances, worse_performing_instances, surprising_instances_better, surprising_instances_worse)

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

    return surprising_instances, data_by_vicinity_id

def detect_surprising_instances_algorithm(pd_data_event_log, descriptive_feature_names, target_feature_name, strategy, add_conditions, threshold, max_depth, detector_function, directory=None):
    if strategy == 'numerical':
        return dt_regression(pd_data_event_log, descriptive_feature_names, target_feature_name, add_conditions, threshold, max_depth, detector_function, directory)
    elif strategy == 'categorical':
        return classify_dt(pd_data_event_log, descriptive_feature_names, target_feature_name, max_depth, add_conditions)
    else:
        print('Undefined strategy')
        return []

def detect_surprising_instances(request, context):
    target_feature, detector_function, model_threshold, model_strategy, situation_type = read_session_parameters(request)

    pd_data_event_log, event_log, feature_list = transform_event_log_to_situations(request=request,situation_type=situation_type)

    decision_tree_max_depth = int(request.session['max_depth_decision_tree'])
    print('Target: ' + str(target_feature))
    
    surprising_instances, data_by_vicinity_id = detect_surprising_instances_algorithm(pd_data_event_log=pd_data_event_log, descriptive_feature_names=feature_list, target_feature_name=target_feature, strategy=model_strategy, add_conditions=True, threshold=model_threshold, max_depth=decision_tree_max_depth, detector_function=detector_function)
    
    context = filter_results_by_vicinity_id(request, surprising_instances, context)

    surprising_instances_len = get_len_surprising_instances(surprising_instances)

    context['decision_tree_path'] = 'detection/figures/decision_tree.png'
    context['target_attribute_name'] = request.session['target_attribute']
    context['decision_tree_path_rca_better'] = 'detection/figures/dt_rca_better.png'
    context['decision_tree_path_rca_worse'] = 'detection/figures/dt_rca_worse.png'
    selected_leaf_id = request.session['selected_leaf_id']

    request.session['instances_in_vicinity'] = data_by_vicinity_id[selected_leaf_id].to_json(orient='split')
    return context, event_log, surprising_instances_len