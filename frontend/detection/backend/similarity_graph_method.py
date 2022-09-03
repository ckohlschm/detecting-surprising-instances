from turtle import distance
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features

from .variants_filter import get_variants, apply_variant_filter
from .models import SurprisingInstance
from .data_reader import transform_log_to_feature_table
from .detection_util import calculate_surprising_instance_statistics, import_and_filter_event_log, filter_results_by_vicinity_id, find_outliers_for_node_boxplot, find_outliers_for_node_threshold, find_outliers_for_node_categorical, get_len_surprising_instances, read_session_parameters, transform_event_log_to_situations

def apply_similarity_graph(request, context):
    if 'leaf_select' in request.POST:
        selected_leaf_id = request.POST['leaf_select']
        print('Selected leaf: ' + str(selected_leaf_id))
        request.session['selected_leaf_id'] = selected_leaf_id

    context, event_log, surprising_instance_len = detect_surprising_instances(request, context)
    context = calculate_surprising_instance_statistics(event_log, surprising_instance_len, context)
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

def levenshtein_distance_method(event_log, target_feature, distance_threshold, strategy, detector_function, threshold):
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
        filtered_data, feature_names = transform_log_to_feature_table('', filtered_log)
    
        data_by_vicinity[id] = filtered_data
        features = filtered_data.columns.tolist()
        target_feature_index = features.index(target_feature)

        better_performing_instances_list = []
        worse_performing_instances_list = []
        if strategy == 'categorical':
            better_performing_instances, worse_performing_instances = find_outliers_for_node_categorical(filtered_data, target_feature)
            expected = filtered_data[target_feature].value_counts().index.tolist()[0]
            for instance in better_performing_instances.values.tolist():
                better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature, expected, instance[target_feature_index], id, True, []))
        else:
            if detector_function == 'threshold':
                better_performing_instances, worse_performing_instances = find_outliers_for_node_threshold(filtered_data, target_feature, threshold)
            else:
                better_performing_instances, worse_performing_instances = find_outliers_for_node_boxplot(filtered_data, target_feature)

            expected = filtered_data[target_feature].mean()
            for instance in better_performing_instances.values.tolist():
                better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature, expected, instance[target_feature_index], id, False, []))
            for instance in worse_performing_instances.values.tolist():
                worse_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature, expected, instance[target_feature_index], id, False, []))

        surprising_instances[id] = (id, better_performing_instances, worse_performing_instances, better_performing_instances_list, worse_performing_instances_list)
        id = id + 1

    return surprising_instances, data_by_vicinity

def detect_surprising_instances(request, context):
    target_feature, detector_function, model_threshold, model_strategy, situation_type = read_session_parameters(request)

    pd_data_event_log, event_log, selected_feature_names = transform_event_log_to_situations(request=request,situation_type=situation_type)

    # Read Parameters
    distance_function = request.session['similarity_graph_distance_function']
    distance_threshold = float(request.session['similarity_graph_distance_max'])

    if distance_function == 'similarity_graph_distance_function_euclidean':
        vicinities = vicinity_detecion(pd_data_event_log, selected_feature_names, target_feature, distance_threshold)
        surprising_instances, data_by_vicinity_id = surprising_instance_detection(vicinities, pd_data_event_log, target_feature, model_strategy, detector_function, model_threshold)
    else:
        surprising_instances, data_by_vicinity_id = levenshtein_distance_method(event_log, target_feature, distance_threshold, model_strategy, detector_function, model_threshold)

    context = filter_results_by_vicinity_id(request, surprising_instances, context)
    surprising_instances_len = get_len_surprising_instances(surprising_instances)
    selected_leaf_id = int(request.session.get('selected_leaf_id'))
    request.session['instances_in_vicinity'] = data_by_vicinity_id[selected_leaf_id].to_json(orient='split')

    context['target_attribute_name'] = request.session['target_attribute']

    return context, event_log, surprising_instances_len

def euclidean_distance(a, b):
    dist = np.linalg.norm(a-b)
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
                    if abs(distance) <= threshold:
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

def surprising_instance_detection(vicinities, event_log, target_feature_name, strategy, detector_function, threshold):
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
        if strategy == 'categorical':
            better_performing_instances, worse_performing_instances = find_outliers_for_node_categorical(filtered_data, target_feature_name)
            expected = filtered_data[target_feature_name].value_counts().index.tolist()[0]
            for instance in better_performing_instances.values.tolist():
                better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], id, True, []))
        else:
            expected = filtered_data[target_feature_name].mean()
            if detector_function == 'threshold':
                better_performing_instances, worse_performing_instances = find_outliers_for_node_threshold(filtered_data, target_feature_name, threshold)
            else:
                better_performing_instances, worse_performing_instances = find_outliers_for_node_boxplot(filtered_data, target_feature_name)
            for instance in better_performing_instances.values.tolist():
                better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], id, False, []))
            for instance in worse_performing_instances.values.tolist():
                worse_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], id, False, []))

        surprising_instances[id] = (id, better_performing_instances, worse_performing_instances, better_performing_instances_list, worse_performing_instances_list)
        id = id + 1
    
    return surprising_instances, data_by_vicinity
