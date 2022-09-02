import numpy as np
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .data_reader import transform_log_to_feature_table
from .variants_filter import get_variants, apply_variant_filter

from .models import SurprisingInstance, Condition, Node

def apply_clustering(request, context):
    if 'leaf_select' in request.POST:
        selected_leaf_id = request.POST['leaf_select']
        print('Selected leaf: ' + str(selected_leaf_id))
        request.session['selected_leaf_id'] = selected_leaf_id
    
    context, event_log, surprising_instances = detect_surprising_instances(request, context)

    context = calculate_surprising_instance_statistics(event_log, surprising_instances, context)
    return context

def import_and_filter_event_log(request):
    log_path = request.session['log_path']
    variant_filter_strategy = request.session.get('variant_filter_strategy', 'most_common_variant')
    event_log, variants_pd_data, all_variants, variantsdata_piechart = get_variants(log_path, variant_filter_strategy)
    selected_variants = request.session.get('selected_variants', None)
    if not selected_variants:
        default_selection = [key for key in all_variants.keys() if key <= 10]
        if len(all_variants.keys()) > 10:
            default_selection.append('Other')
        request.session['selected_variants'] = default_selection
        selected_variants = default_selection
    filtered_log = apply_variant_filter(event_log, all_variants, selected_variants)
    return filtered_log

def get_len_surprising_instances(surprising_instances):
    all_surprising_instances = []
    for node_id, value in surprising_instances.items():
        node, better_performing_instances, worse_performing_instances, surprising_instances_better, surprising_instances_worse = value
        for instance in surprising_instances_better:
            if instance.id not in all_surprising_instances:
                all_surprising_instances.append(instance.id)
        for instance in surprising_instances_worse:
            if instance.id not in all_surprising_instances:
                all_surprising_instances.append(instance.id)
    return len(all_surprising_instances)

def detect_surprising_instances(request, context):
    log_path = request.session['log_path']
    event_log = import_and_filter_event_log(request)
    pd_data_event_log, feature_names = transform_log_to_feature_table(log_path, event_log)

    # Feature extraction
    target_feature = request.session['target_attribute']
    k_means_number_clusters = int(request.session['k_means_number_clusters'])

    feature_list = request.session['selected_feature_names']
    print('Using ' + str(len(feature_list)) + ' features for vicinity Detection')
    print(feature_list)

    # Detect surprising instances
    detector_function = request.session['detector_function']
    model_threshold = None 
    if detector_function == 'threshold':
        model_threshold = int(request.session['target_attribute_threshold'])
    print('Using Detector function ' + str(detector_function) + ' for surprising instance detection (threshold: ' + str(model_threshold) + ')')
    
    if request.session['target_attribute_type'] == 'categorical':
        model_strategy = 'classification'
    else:
        model_strategy = 'regression'
    
    surprising_instances, data_by_vicinity_id = vicinity_detection(pd_data_event_log, k_means_number_clusters, feature_list, target_feature, detector_function, model_threshold)

    context = filter_results_by_leaf_id(request, surprising_instances, context)

    surprising_instances_len = get_len_surprising_instances(surprising_instances)

    context['target_attribute_name'] = request.session['target_attribute']
    selected_leaf_id = request.session['selected_leaf_id']
    context['selected_leaf_id'] = selected_leaf_id

    request.session['instances_in_vicinity'] = data_by_vicinity_id[selected_leaf_id].to_json(orient='split')
    return context, event_log, surprising_instances_len

def filter_results_by_leaf_id(request, surprising_instances, context):
    leaf_ids = list(surprising_instances.keys())

    if len(leaf_ids) > 0:
        selected_leaf_id = int(request.session.get('selected_leaf_id', leaf_ids[-1]))
    else:
        selected_leaf_id = int(request.session.get('selected_leaf_id', 0))

    list_better_performance = []
    list_worse_performance = []
    list_all_better_performance = []
    list_all_worse_performance = []
    surprising_instances_to_show = []
    barchartleafiddatabetter = []
    barchartleafiddataworse = []
    for node_id, value in surprising_instances.items():
        node, better_performing_instances, worse_performing_instances, surprising_instances_better, surprising_instances_worse = value
        for instance in surprising_instances_better:
            list_all_better_performance.append(instance.actual_data)
        for instance in surprising_instances_worse:
            list_all_worse_performance.append(instance.actual_data)
        barchartleafiddatabetter.append(len(surprising_instances_better))
        barchartleafiddataworse.append(len(surprising_instances_worse))
        if int(node_id) == int(selected_leaf_id):
            surprising_instances_to_show = surprising_instances_better + surprising_instances_worse
            for instance in surprising_instances_better:
                list_better_performance.append(instance.actual_data)
            for instance in surprising_instances_worse:
                list_worse_performance.append(instance.actual_data)
        
    surprising_instances_to_show.sort(key=lambda x: x.calculateDifference(), reverse=True)

    context['barchartleafiddatabetter'] = barchartleafiddatabetter
    context['barchartleafiddataworse'] = barchartleafiddataworse
    context['selected_leaf_id'] = selected_leaf_id
    request.session['selected_leaf_id'] = selected_leaf_id
    context['surprising_instances'] = surprising_instances_to_show
    
    if len(list_better_performance) > 0:
        if request.session['target_attribute'] == '@@caseDuration':
             context['avg_better_leaf_performance'] = datetime.timedelta(seconds=round(sum(list_better_performance) / len(list_better_performance), 0))
        else:
            context['avg_better_leaf_performance'] = round(sum(list_better_performance) / len(list_better_performance), 2)
    else:
        context['avg_better_leaf_performance'] = 0
    if len(list_worse_performance) > 0:
        if request.session['target_attribute'] == '@@caseDuration':
             context['avg_worse_leaf_performance'] = datetime.timedelta(seconds=round(sum(list_worse_performance) / len(list_worse_performance), 0))
        else:
            context['avg_worse_leaf_performance'] = round(sum(list_worse_performance) / len(list_worse_performance), 2)
    else:
        context['avg_worse_leaf_performance'] = 0

    if len(list_all_better_performance) > 0:
        if request.session['target_attribute'] == '@@caseDuration':
             context['avg_all_better_performance'] = datetime.timedelta(seconds=round(sum(list_all_better_performance) / len(list_all_better_performance), 0))
        else:
            context['avg_all_better_performance'] = round(sum(list_all_better_performance) / len(list_all_better_performance), 2)
    else: 
        context['avg_all_better_performance'] = 0
    if len(list_all_worse_performance) > 0:
        if request.session['target_attribute'] == '@@caseDuration':
             context['avg_all_worse_performance'] = datetime.timedelta(seconds=round(sum(list_all_worse_performance) / len(list_all_worse_performance), 0))
        else:
            context['avg_all_worse_performance'] = round(sum(list_all_worse_performance) / len(list_all_worse_performance), 2)
    else: 
        context['avg_all_worse_performance'] = 0

    context['num_better_leaf'] = len(list_better_performance)
    context['num_worse_leaf'] = len(list_worse_performance)
    context['leaf_ids'] = leaf_ids
    #request.session['leaf_ids'] = leaf_ids
    return context

def vicinity_detection(event_log, k_means_number_clusters, feature_names, target_feature, detector_function, threshold):
    print(event_log.head())
    print(feature_names)
    event_log_filtered = event_log.filter(feature_names)
    print(event_log_filtered.head())

    cols_to_normalize = feature_names
    event_log_filtered[cols_to_normalize] = MinMaxScaler().fit_transform(event_log_filtered[cols_to_normalize])
    print(event_log_filtered.head())

    event_log_filtered_no_case_id = event_log_filtered.filter(feature_names)
    situations = event_log_filtered_no_case_id.values.tolist()

    kmeans = KMeans(n_clusters=k_means_number_clusters, random_state=0).fit(situations)

    k_means_labels = kmeans.labels_
    print('Found clusters: ')
    print(k_means_labels)

    event_log['cluster_id'] = k_means_labels

    surprising_instances, data_by_vicinity = surprising_instance_detection(event_log, k_means_number_clusters, target_feature, detector_function, threshold)

    return surprising_instances, data_by_vicinity

def surprising_instance_detection(event_log, k_means_number_clusters, target_feature_name, detector_function, threshold):
    surprising_instances = {}
    data_by_vicinity = {}

    for vicinity_id in range(k_means_number_clusters):
        filtered_data = event_log[event_log['cluster_id'] == vicinity_id]
        data_by_vicinity[vicinity_id] = filtered_data
        features = filtered_data.columns.tolist()
        target_feature_index = features.index(target_feature_name)

        print('Filtered data: ' + str(len(filtered_data)))
        #print(filtered_data.head())
        better_performing_instances_list = []
        worse_performing_instances_list = []
        if detector_function == 'threshold':
            better_performing_instances, worse_performing_instances = find_outliers_for_node_threshold(filtered_data, target_feature_name, threshold)
        else:
            better_performing_instances, worse_performing_instances = find_outliers_for_node(filtered_data, target_feature_name)
        #print(better_performing_instances.head())
        #print(worse_performing_instances.head())
        expected = filtered_data[target_feature_name].mean()
        for instance in better_performing_instances.values.tolist():
            print('Adding better performing instance')
            better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], vicinity_id, False, []))
        for instance in worse_performing_instances.values.tolist():
            print('Adding worse performing instance')
            worse_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], vicinity_id, False, []))

        surprising_instances[vicinity_id] = (vicinity_id, better_performing_instances, worse_performing_instances, better_performing_instances_list, worse_performing_instances_list)
        #case_id_col_list = filtered_data['@@case_id_column'].tolist()

    return surprising_instances, data_by_vicinity

def calculate_surprisingness_index_better(row, target_feature_name, p_avg, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * affectedInstances 

def calculate_surprisingness_index_worse(row, target_feature_name, p_avg, vicinitySize, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * (vicinitySize - affectedInstances)

def calculate_relevance_worse(row, vicinitySize):
    return row['surprisingnessWorseIndex'] * vicinitySize

def calculate_relevance_better(row, vicinitySize):
    return row['surprisingnessBetterIndex'] * vicinitySize

def find_outliers_for_node_threshold(filtered_data, target_feature_name, threshold):
    print('Target Feature Name: ' + str(target_feature_name))
    mean_value = filtered_data[target_feature_name].mean()

    lower_bound = mean_value - threshold
    upper_bound = mean_value + threshold

    filter_better = (filtered_data[target_feature_name] < lower_bound)
    event_log_filter_better = filtered_data.loc[filter_better]
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_better.index)]
    #print("All: " + str(len(filtered_data))+ " Better: " + str(len(event_log_filter_better)) + " Other: " + str(len(other_instances_in_vicinity)))
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    affectedInstances = len(event_log_filter_better)
    vicinitySize = len(filtered_data)
    if len(event_log_filter_better) > 0:
        event_log_filter_better['surprisingnessBetterIndex'] = event_log_filter_better.apply(lambda row: calculate_surprisingness_index_better(row=row, target_feature_name=target_feature_name, p_avg= p_avg, affectedInstances=affectedInstances), axis=1)
        event_log_filter_better['RelevanceIndex'] = event_log_filter_better.apply(lambda row: calculate_relevance_better(row=row, vicinitySize=vicinitySize), axis=1)
    #print(event_log_filter_better.head())
    print("There are " + str(len(event_log_filter_better)) + " better performing instances")

    filter_worse = (filtered_data[target_feature_name] > upper_bound)
    event_log_filter_worse = filtered_data.loc[filter_worse] 
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_worse.index)]
    #print("All: " + str(len(filtered_data))+ " Worse: " + str(len(event_log_filter_worse)) + " Other: " + str(len(other_instances_in_vicinity)))
    #print("All: " + str(filtered_data[target_feature_name].mean())+ " Worse: " + str(event_log_filter_worse[target_feature_name].mean()) + " Other: " + str(other_instances_in_vicinity[target_feature_name].mean()))
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    affectedInstances = len(event_log_filter_worse)
    vicinitySize = len(filtered_data)
    if len(event_log_filter_worse) > 0:
        event_log_filter_worse['surprisingnessWorseIndex'] = event_log_filter_worse.apply(lambda row: calculate_surprisingness_index_worse(row=row, target_feature_name=target_feature_name, p_avg= p_avg, vicinitySize=vicinitySize, affectedInstances=affectedInstances), axis=1)
        event_log_filter_worse['RelevanceIndex'] = event_log_filter_worse.apply(lambda row: calculate_relevance_worse(row=row, vicinitySize=vicinitySize), axis=1)
    #print(event_log_filter_worse.head())
    print("There are " + str(len(event_log_filter_worse)) + " worse performing instances")

    return event_log_filter_better, event_log_filter_worse


def find_outliers_for_node(filtered_data, target_feature_name):
    print('Target Feature Name: ' + str(target_feature_name))
    Q1 = filtered_data[target_feature_name].quantile(0.25)
    Q3 = filtered_data[target_feature_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 *IQR

    filter_better = (filtered_data[target_feature_name] < lower_bound)
    event_log_filter_better = filtered_data.loc[filter_better]
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_better.index)]
    #print("All: " + str(len(filtered_data))+ " Better: " + str(len(event_log_filter_better)) + " Other: " + str(len(other_instances_in_vicinity)))
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    affectedInstances = len(event_log_filter_better)
    vicinitySize = len(filtered_data)
    if len(event_log_filter_better) > 0:
        event_log_filter_better['surprisingnessBetterIndex'] = event_log_filter_better.apply(lambda row: calculate_surprisingness_index_better(row=row, target_feature_name=target_feature_name, p_avg= p_avg, affectedInstances=affectedInstances), axis=1)
        event_log_filter_better['RelevanceIndex'] = event_log_filter_better.apply(lambda row: calculate_relevance_better(row=row, vicinitySize=vicinitySize), axis=1)
    #print(event_log_filter_better.head())
    print("There are " + str(len(event_log_filter_better)) + " better performing instances")

    filter_worse = (filtered_data[target_feature_name] > upper_bound)
    event_log_filter_worse = filtered_data.loc[filter_worse] 
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_worse.index)]
    #print("All: " + str(len(filtered_data))+ " Worse: " + str(len(event_log_filter_worse)) + " Other: " + str(len(other_instances_in_vicinity)))
    #print("All: " + str(filtered_data[target_feature_name].mean())+ " Worse: " + str(event_log_filter_worse[target_feature_name].mean()) + " Other: " + str(other_instances_in_vicinity[target_feature_name].mean()))
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    affectedInstances = len(event_log_filter_worse)
    vicinitySize = len(filtered_data)
    if len(event_log_filter_worse) > 0:
        event_log_filter_worse['surprisingnessWorseIndex'] = event_log_filter_worse.apply(lambda row: calculate_surprisingness_index_worse(row=row, target_feature_name=target_feature_name, p_avg= p_avg, vicinitySize=vicinitySize, affectedInstances=affectedInstances), axis=1)
        event_log_filter_worse['RelevanceIndex'] = event_log_filter_worse.apply(lambda row: calculate_relevance_worse(row=row, vicinitySize=vicinitySize), axis=1)
    #print(event_log_filter_worse.head())
    print("There are " + str(len(event_log_filter_worse)) + " worse performing instances")

    return event_log_filter_better, event_log_filter_worse

def calculate_surprising_instance_statistics(all_cases, surprising_instances_len, context):
    surprising_instance_count = surprising_instances_len
    all_cases_count = len(all_cases)
    surprising_instance_percentage = surprising_instance_count / all_cases_count
    surprising_instance_percentage = round(surprising_instance_percentage * 100, 2)
    context['surprising_instance_count'] = surprising_instance_count
    context['non_surprising_instance_count'] = all_cases_count - surprising_instance_count
    context['all_cases_count'] = all_cases_count
    print('Instance percentage: ' + str(surprising_instance_percentage))
    context['surprising_instance_percentage'] = surprising_instance_percentage
    non_surprising_instance_percentage = 100 - surprising_instance_percentage
    non_surprising_instance_percentage = round(non_surprising_instance_percentage, 2)
    context['non_surprising_instance_percentage'] = non_surprising_instance_percentage
    piechartdata = []
    piechartdata.append(surprising_instance_percentage)
    piechartdata.append(non_surprising_instance_percentage)
    context['surprisinginstancedatapiechart'] = piechartdata
    return context
