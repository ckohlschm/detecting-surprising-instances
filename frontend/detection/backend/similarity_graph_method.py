from turtle import distance
import numpy as np
import networkx as nx
import math
import uuid
import matplotlib.pyplot as plt
# switch backend
plt.switch_backend('agg')
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler

from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.variants.log import get
from pm4py.util import variants_util
from pm4py.stats import get_case_duration
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features

from .variants_filter import get_variants, apply_variant_filter
from .models import SurprisingInstance
from .data_reader import transform_log_to_feature_table

def apply_similarity_graph(request, context):
    if 'variant_select' in request.POST:
        selected_leaf_id = request.POST['variant_select']
        print('Selected leaf: ' + str(selected_leaf_id))
        request.session['selected_leaf_id'] = selected_leaf_id

    context, event_log, surprising_instance_len = detect_surprising_instances(request, context)
    context = calculate_surprising_instance_statistics(event_log, surprising_instance_len, context)
    return context

def calculate_surprising_instance_statistics(all_cases, surprising_instance_len, context):
    surprising_instance_count = surprising_instance_len
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

def levenshtein(seq1, seq2):
    '''
    Edit distance without substitution
    '''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] in [None,"tau"] or seq1[x-1][0]=='n' or "skip" in seq1[x-1] or "tau" in seq1[x-1] :
                matrix [x,y] = min(
                    matrix[x-1, y],
                    matrix[x,y-1] + 1
                )
            elif seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

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

def filter_results_by_community_id(request, surprising_instances, context):
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

    context['barchartvariantsdatabetter'] = barchartleafiddatabetter
    context['barchartvariantsdataworse'] = barchartleafiddataworse
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
    context['variant_ids'] = leaf_ids
    return context

def filter_results_by_variant_id(request, surprising_instances, context):
    variant_ids = []
    for instance in surprising_instances:
        current_variant_id = int(instance.leaf_id)
        if current_variant_id not in variant_ids:
            variant_ids.append(current_variant_id)

    selected_leaf_id = int(request.session.get('selected_leaf_id', variant_ids[0]))
    surprising_instances_to_show = []
    variant_id_dict = {}
    all_better_variant_id_dict = {}
    all_worse_variant_id_dict = {}
    list_all_better_performance = []
    list_all_worse_performance = []
    better_variant_id_dict = {}
    worse_variant_id_dict = {}
    list_better_performance = []
    list_worse_performance = []
    for instance in surprising_instances:
        current_variant_id = int(instance.leaf_id)
        if request.session['target_attribute_type'] == 'categorical':
            if current_variant_id == selected_leaf_id:
                surprising_instances_to_show.append(instance)
        else:
            if instance.actual_data > instance.target_data:
                if current_variant_id in all_worse_variant_id_dict:
                    all_worse_variant_id_dict[current_variant_id] = all_worse_variant_id_dict[current_variant_id] + 1
                else: 
                    all_worse_variant_id_dict[current_variant_id] = 1
                list_all_worse_performance.append(instance.actual_data)
            if instance.actual_data < instance.target_data:
                if current_variant_id in all_better_variant_id_dict:
                    all_better_variant_id_dict[current_variant_id] = all_better_variant_id_dict[current_variant_id] + 1
                else: 
                    all_better_variant_id_dict[current_variant_id] = 1
                list_all_better_performance.append(instance.actual_data)
            if current_variant_id in variant_id_dict:
                variant_id_dict[current_variant_id] = variant_id_dict[current_variant_id] + 1
            else: 
                variant_id_dict[current_variant_id] = 1
            if current_variant_id == selected_leaf_id:
                if instance.actual_data > instance.target_data:
                    if current_variant_id in worse_variant_id_dict:
                        worse_variant_id_dict[current_variant_id] = worse_variant_id_dict[current_variant_id] + 1
                    else: 
                        worse_variant_id_dict[current_variant_id] = 1
                    list_worse_performance.append(instance.actual_data)
                if instance.actual_data < instance.target_data:
                    if current_variant_id in better_variant_id_dict:
                        better_variant_id_dict[current_variant_id] = better_variant_id_dict[current_variant_id] + 1
                    else: 
                        better_variant_id_dict[current_variant_id] = 1
                    list_better_performance.append(instance.actual_data)
                surprising_instances_to_show.append(instance)
    
    #barchartvariantsdata = []
    #for variant_id in variant_ids:
    #    barchartvariantsdata.append(variant_id_dict[variant_id])

    barchartvariantsdatabetter = []
    for variant_id in variant_ids:
        if variant_id in all_better_variant_id_dict:
            barchartvariantsdatabetter.append(all_better_variant_id_dict[variant_id])
        else:
            barchartvariantsdatabetter.append(0)

    barchartvariantsdataworse = []
    for variant_id in variant_ids:
        if variant_id in all_worse_variant_id_dict:
            barchartvariantsdataworse.append(all_worse_variant_id_dict[variant_id])
        else:
            barchartvariantsdataworse.append(0)

    #context['barchartvariantsdata'] = barchartvariantsdata
    context['barchartvariantsdataworse'] = barchartvariantsdataworse
    context['barchartvariantsdatabetter'] = barchartvariantsdatabetter
    context['selected_leaf_id'] = selected_leaf_id
    request.session['selected_leaf_id'] = selected_leaf_id
    surprising_instances_to_show.sort(key=lambda x: x.calculateDifference(), reverse=True)

    context['surprising_instances'] = surprising_instances_to_show
    #request.session['surprising_instances'] = surprising_instances_to_show
    
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
    if selected_leaf_id in better_variant_id_dict:
        context['num_better_leaf'] = better_variant_id_dict[selected_leaf_id]
    else:
        context['num_better_leaf'] = 0
    if selected_leaf_id in worse_variant_id_dict:
        context['num_worse_leaf'] = worse_variant_id_dict[selected_leaf_id]
    else:
        context['num_worse_leaf'] = 0
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
    context['variant_ids'] = variant_ids
    #request.session['variant_ids'] = variant_ids
    return context

def levenshtein_distance_method(event_log, target_feature, distance_threshold, detector_function, threshold):
    variants_count = case_statistics.get_variant_statistics(event_log)

    all_variants = []
    index_variant_map = {}
    var_index = 1
    for variant in variants_count:
        variant_string = variant['variant']
        index_variant_map[variant_string] = var_index
        variant_list = variant_string.split(',')
        all_variants.append(variant_list)
        var_index = var_index + 1

    # Create graph
    G = nx.Graph()
    var_index = 1
    variant_index_map = {}
    for variant in all_variants:
        G.add_node(var_index)
        variant_index_map[var_index] = variant
        var_index = var_index + 1

    for index1, variant1 in variant_index_map.items():
        for index2, variant2 in variant_index_map.items():
            if index1 != index2:
                levenshtein_distance = levenshtein(variant1, variant2)
                if levenshtein_distance < distance_threshold:
                    G.add_edge(index1, index2)


    result = nx.algorithms.community.louvain.louvain_communities(G, seed=123)

    # print('Similar variants dictionary: ' + str(similarity_dict))
    #pos = nx.spring_layout(G, k=5/math.sqrt(G.order()), seed=94)
    #nx.draw(G, pos, with_labels=True)
    #uid = uuid.uuid4()
    #variant_graph_path = 'detection/figures/similarity_graph_' + str(uid) + '.png'
    #context['variant_graph_path'] = variant_graph_path
    #request.session['variant_graph_path'] = variant_graph_path
    #plt.savefig('detection/static/' + str(variant_graph_path))
    #plt.clf()

    surprising_instances = {}
    data_by_vicinity = {}

    id = 1
    for vicinity in result:
        variants_subset = []
        for variant_id in vicinity:
            variant = variant_index_map[variant_id]
            variant_string = ','.join(variant)
            variants_subset.append(variant_string)
        print(variants_subset)

        filtered_log = variants_filter.apply(event_log, variants_subset)

        parameters = {}
        parameters["add_case_identifier_column"] = True
        parameters["enable_case_duration"] = True
        parameters["enable_succ_def_representation"] = False
        data, feature_names = log_to_features.apply(filtered_log, variant=log_to_features.Variants.TRACE_BASED, parameters = parameters)
        filtered_data = pd.DataFrame(data, columns=feature_names)

        data_by_vicinity[id] = filtered_data
        features = filtered_data.columns.tolist()
        target_feature_index = features.index(target_feature)

        better_performing_instances_list = []
        worse_performing_instances_list = []
        if detector_function == 'threshold':
            better_performing_instances, worse_performing_instances = find_outliers_for_node_threshold(filtered_data, target_feature, threshold)
        else:
            better_performing_instances, worse_performing_instances = find_outliers_for_node(filtered_data, target_feature)

        expected = filtered_data[target_feature].mean()
        for instance in better_performing_instances.values.tolist():
            better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature, expected, instance[target_feature_index], id, False, []))
        for instance in worse_performing_instances.values.tolist():
            worse_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature, expected, instance[target_feature_index], id, False, []))

        surprising_instances[id] = (id, better_performing_instances, worse_performing_instances, better_performing_instances_list, worse_performing_instances_list)
        id = id + 1

    return surprising_instances, data_by_vicinity

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
    event_log = import_and_filter_event_log(request)

    # Parameters
    distance_threshold = float(request.session['similarity_graph_distance_max'])

    target_feature = request.session['target_attribute']
    performance_threshold = int(request.session['target_attribute_threshold'])
    if request.session['target_attribute_type'] == 'categorical':
        model_strategy = 'classification'
    else:
        model_strategy = 'regression'

    detector_function = request.session['detector_function']
    
    distance_function = request.session['similarity_graph_distance_function']
    if distance_function == 'similarity_graph_distance_function_euclidean':
        pd_data_event_log, feature_names = transform_log_to_feature_table(request.session['log_path'], event_log)
        selected_feature_names = request.session['selected_feature_names']
        vicinities = vicinity_detecion(pd_data_event_log, selected_feature_names, target_feature, distance_threshold)
        surprising_instances, data_by_vicinity_id = surprising_instance_detection(vicinities, pd_data_event_log, target_feature, detector_function, performance_threshold)
    else:
        surprising_instances, data_by_vicinity_id = levenshtein_distance_method(event_log, target_feature, distance_threshold, detector_function, performance_threshold)

    context = filter_results_by_community_id(request, surprising_instances, context)
    surprising_instances_len = get_len_surprising_instances(surprising_instances)
    selected_leaf_id = int(request.session.get('selected_leaf_id'))
    request.session['instances_in_vicinity'] = data_by_vicinity_id[selected_leaf_id].to_json(orient='split')

    #context['surprising_instances'] = surprising_instances
    context['target_attribute_name'] = request.session['target_attribute']

    return context, event_log, surprising_instances_len

def euclidean_distance(a, b):
    #a = np.asarray(situation_1[1:])
    #b = np.asarray(situation_2[1:])
    dist = np.linalg.norm(a-b)
    # print('Distance ' + str(a) + ':' + str(b) +' - ' + str(dist))
    return dist

def vicinity_detecion(event_log, selected_feature_names, target_feature, distance_threshold):
    feature_names_with_id = []
    for item in selected_feature_names:
        feature_names_with_id.append(item)
    feature_names_with_id.insert(0, '@@case_id_column')
    event_log_filtered = event_log.filter(feature_names_with_id)

    cols_to_normalize = selected_feature_names
    event_log_filtered[cols_to_normalize] = MinMaxScaler().fit_transform(event_log_filtered[cols_to_normalize])

    def find_community_for_log(event_log_filtered, target_feature, distance_threshold):
        situations = event_log_filtered.values.tolist()
        features = event_log_filtered.columns.tolist()
        id_column_index = features.index('@@case_id_column')

        threshold = distance_threshold

        print(features)
        print(id_column_index)

        # Create graph
        G = nx.Graph()
        for situation in situations:
            G.add_node(situation[id_column_index])
       
        print('Finish creating nodes')
        # add edges
        amount_instances = len(situations)
        curr_index = 1 
        for situation in situations:
            vector1 =  np.asarray(situation[1:])
            added_edges = 0
            for situation2 in situations:
                if situation[id_column_index] != situation2[id_column_index]:
                    vector2 = np.asarray(situation2[1:])
                    distance = euclidean_distance(vector1, vector2)
                    if abs(distance) < threshold:
                        G.add_edge(situation[id_column_index], situation2[id_column_index])
                        added_edges = added_edges + 1
            
            print(f'Finish calculation for {str(situation[0])}      ({str(curr_index)}/{str(amount_instances)}):        {str(added_edges)}      edges added')
            curr_index = curr_index + 1
        
        print('Finish creating edges')
        result = nx.algorithms.community.louvain.louvain_communities(G, seed=123)

        print('Louvain: Found ' + str(len(result)) + ' vicinities')
        return result
    
    result = find_community_for_log(event_log_filtered, target_feature, distance_threshold)

    return result

def filter_data_by_ids(event_log, similar_situation_ids):
    df_similar_situations = event_log.loc[event_log['@@case_id_column'].isin(similar_situation_ids)]
    return df_similar_situations

def surprising_instance_detection(vicinities, event_log, target_feature_name, detector_function, threshold):
    surprising_instances = {}
    data_by_vicinity = {}

    id = 1
    for vicinity in vicinities:
        filtered_data = filter_data_by_ids(event_log, vicinity)
        data_by_vicinity[id] = filtered_data
        features = filtered_data.columns.tolist()
        target_feature_index = features.index(target_feature_name)

        better_performing_instances_list = []
        worse_performing_instances_list = []

        expected = filtered_data[target_feature_name].mean()
        if detector_function == 'threshold':
            better_performing_instances, worse_performing_instances = find_outliers_for_node_threshold(filtered_data, target_feature_name, threshold)
        else:
            better_performing_instances, worse_performing_instances = find_outliers_for_node(filtered_data, target_feature_name)
        for instance in better_performing_instances.values.tolist():
            print('Adding better performing instance')
            better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], id, False, []))
        for instance in worse_performing_instances.values.tolist():
            print('Adding worse performing instance')
            worse_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], id, False, []))


        surprising_instances[id] = (id, better_performing_instances, worse_performing_instances, better_performing_instances_list, worse_performing_instances_list)
        id = id + 1
    
    return surprising_instances, data_by_vicinity

def calculate_surprisingness_index_better(row, target_feature_name, p_avg, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * affectedInstances 

def calculate_surprisingness_index_worse(row, target_feature_name, p_avg, vicinitySize, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * (vicinitySize - affectedInstances)

def calculate_relevance_worse(row, vicinitySize):
    return row['surprisingnessWorseIndex'] * vicinitySize

def calculate_relevance_better(row, vicinitySize):
    return row['surprisingnessBetterIndex'] * vicinitySize

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
    print(event_log_filter_worse.head())
    print("There are " + str(len(event_log_filter_worse)) + " worse performing instances")

    return event_log_filter_better, event_log_filter_worse

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

def index_cases(variants_case_dict):
    least_frequent_variant = 'Confirmation of receipt,T06 Determine necessity of stop advice,T02 Check confirmation of receipt,T04 Determine confirmation of receipt,T05 Print and send confirmation of receipt,T10 Determine necessity to stop indication,T16 Report reasons to hold request,T17 Check report Y to stop indication,T19 Determine report Y to stop indication,T20 Print report Y to stop indication'
    
    #least_freq = variants_util.get_variant_from_trace(least_frequent_variant)
    #print(least_freq)

    print('Value: ' + str(variants_case_dict[least_frequent_variant][0]))
    print('Attributes: ' + str(variants_case_dict[least_frequent_variant][0].attributes))
    print('ID: ' + str(variants_case_dict[least_frequent_variant][0].attributes['concept:name']))

def get_neightbors_of_cases(all_variants, edge_threshold, variants_case_dict):
    similar_variants_dict = {}
    var_index = 1
    for variant in all_variants[:1]:
        num_neighbors = 0
        duration_sum = 0
        neighbors = nx.single_source_shortest_path_length(G, var_index, cutoff = edge_threshold)
        variant_string = ','.join(variant)
        #print('Neighbor of ' + str(variant) + ': ' + str(neighbors))
        var_index = var_index + 1

    for key, value in variants_case_dict.items():
        variant_string = key
        variant_list = variant_string.split(',')

    #print('Dict: ' + str(list(variants_case_dict)[-1]))


def apply_alignments(variants_count, event_log):
    first_log_variants = []
    for variant in variants_count[:5]:
        first_log_variants.append(variant['variant'])
    #print('First log: ' + str(first_log_variants))
    second_log_variants = []
    for variant in variants_count[:5]:
        second_log_variants.append(variant['variant'])
    #print('Second log: ' + str(second_log_variants))

    first_filtered_log = variants_filter.apply(event_log, first_log_variants)
    second_filtered_log = variants_filter.apply(event_log, second_log_variants)

    parameters = {logs_alignments.Variants.EDIT_DISTANCE.value.Parameters.PERFORM_ANTI_ALIGNMENT: True}
    alignments = logs_alignments.apply(first_filtered_log, second_filtered_log, parameters=parameters)
    #print('Alignments: ' + str(alignments))