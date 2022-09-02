import numpy as np
import networkx as nx
import math
import uuid
import matplotlib.pyplot as plt
import pandas as pd

from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.variants.log import get
from pm4py.util import variants_util
from pm4py.stats import get_case_duration
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features

from .importer import import_file

def process_parameters(request, context):
    read_parameters(request)
    return context


def read_parameters(request):
    if 'submit_parameters_supervised_learning' in request.POST:
        read_parameters_supervised_learning(request)
    if 'submit_parameters_similarity_graph' in request.POST:
        read_parameters_similarity_graph(request)
    if 'submit_parameters_random_walk' in request.POST:
        read_parameters_clustering(request)
    
    print('Parameters submitted! Detecting Surprising Instances')

def read_parameters_similarity_graph(request):
    target_attribute_name = request.POST['target_attribute']
    request.session['target_attribute'] = target_attribute_name
    print('Selected target attribute: ' + str(target_attribute_name))
    
    #target_attribute_type = request.POST['target_attribute_type']
    #request.session['target_attribute_type'] = target_attribute_type
    #print('Selected target attribute type: ' + str(target_attribute_type))
    
    if target_attribute_name == 'delay':
        target_attribute_type = 'categorical' # request.POST['target_attribute_type']
    else:
        target_attribute_type = 'numerical' # request.POST['target_attribute_type']
    request.session['target_attribute_type'] = target_attribute_type
    
    print('Selected target attribute type: ' + str(target_attribute_type))
    target_attribute_threshold = request.POST['target_attribute_threshold']
    request.session['target_attribute_threshold'] = target_attribute_threshold
    print('Selected threshold : ' + str(target_attribute_threshold))

    similarity_graph_distance_function = request.POST['similarity_graph_distance_function']
    request.session['similarity_graph_distance_function'] = similarity_graph_distance_function
    print('Selected distance function: ' + str(similarity_graph_distance_function))
    selected_feature_names =  request.POST.getlist('descriptive_attributes')
    request.session['selected_feature_names'] = selected_feature_names
    print('Selected feature names: ' + str(selected_feature_names))

    similarity_graph_distance_max = request.POST['similarity_graph_distance_max']
    request.session['similarity_graph_distance_max'] = similarity_graph_distance_max
    print('Selected max distance threshold: ' + str(similarity_graph_distance_max))

    detector_funtion = request.POST['detector_function']
    if detector_funtion == 'threshold':
        target_attribute_threshold = request.POST['target_attribute_threshold']
        request.session['target_attribute_threshold'] = target_attribute_threshold
        print('Selected threshold : ' + str(target_attribute_threshold))
    request.session['detector_function'] = detector_funtion
    print('Selected Detector function: ' + str(detector_funtion))

def read_parameters_clustering(request):
    target_attribute_name = request.POST['target_attribute_case']
    request.session['target_attribute'] = target_attribute_name
    print('Selected target attribute: ' + str(target_attribute_name))

    if target_attribute_name == 'delay':
        target_attribute_type = 'categorical' # request.POST['target_attribute_type']
    else:
        target_attribute_type = 'numerical' # request.POST['target_attribute_type']
    request.session['target_attribute_type'] = target_attribute_type
    print('Selected target attribute type: ' + str(target_attribute_type))

    vicinity_detection_method = request.POST['vicinity_detection_method']
    request.session['vicinity_detection_method'] = vicinity_detection_method
    print('Selected vicinity detection method: ' + str(vicinity_detection_method))
    k_means_number_clusters = request.POST['k_means_number_clusters']
    request.session['k_means_number_clusters'] = k_means_number_clusters
    print('Selected number of clusters: ' + str(k_means_number_clusters))
    selected_feature_names =  request.POST.getlist('descriptive_attributes')
    request.session['selected_feature_names'] = selected_feature_names
    print('Selected feature names: ' + str(selected_feature_names))

    detector_funtion = request.POST['detector_function']
    if detector_funtion == 'threshold':
        target_attribute_threshold = request.POST['target_attribute_threshold']
        request.session['target_attribute_threshold'] = target_attribute_threshold
        print('Selected threshold : ' + str(target_attribute_threshold))
    request.session['detector_function'] = detector_funtion
    print('Selected Detector function: ' + str(detector_funtion))

def read_parameters_supervised_learning(request):
    target_attribute_name = request.POST['target_attribute_case']
    request.session['target_attribute'] = target_attribute_name
    print('Selected target attribute: ' + str(target_attribute_name))

    if target_attribute_name == 'delay':
        target_attribute_type = 'categorical' # request.POST['target_attribute_type']
    else:
        target_attribute_type = 'numerical' # request.POST['target_attribute_type']
    request.session['target_attribute_type'] = target_attribute_type
    print('Selected target attribute type: ' + str(target_attribute_type))

    vicinity_detection_method = request.POST['vicinity_detection_method']
    request.session['vicinity_detection_method'] = vicinity_detection_method
    print('Selected vicinity detection method: ' + str(vicinity_detection_method))
    max_depth_decision_tree = request.POST['max_depth_decision_tree']
    request.session['max_depth_decision_tree'] = max_depth_decision_tree
    print('Selected decision tree max depth: ' + str(max_depth_decision_tree))
    selected_feature_names =  request.POST.getlist('descriptive_attributes')
    request.session['selected_feature_names'] = selected_feature_names
    print('Selected feature names: ' + str(selected_feature_names))

    detector_funtion = request.POST['detector_function']
    if detector_funtion == 'threshold':
        target_attribute_threshold = request.POST['target_attribute_threshold']
        request.session['target_attribute_threshold'] = target_attribute_threshold
        print('Selected threshold : ' + str(target_attribute_threshold))
    request.session['detector_function'] = detector_funtion
    print('Selected Detector function: ' + str(detector_funtion))

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



def perform_alignment(request):
    parameters = {}
    log_path = request.session['log_path']
    event_log = import_file(log_path, False)

    variants_count = case_statistics.get_variant_statistics(event_log)

    all_variants = []
    for variant in variants_count:
        variant_string = variant['variant']
        variant_list = variant_string.split(',')
        all_variants.append(variant_list)

    #print('Variant 0: ' + str(variants_count[0]['variant']))
    variants_case_dict = get.get_variants(event_log)

    # Parameters
    distance_threshold = 2
    edge_threshold = 2
    performance_threshold = 100000

    # Create graph
    G = nx.Graph()
    var_index = 1
    variant_index_map = {}
    for variant in all_variants:
        G.add_node(var_index)
        variant_index_map[var_index] = variant
        var_index = var_index + 1

    similarity_dict = {}
    for index1, variant1 in variant_index_map.items():
        similar_variants = []
        for index2, variant2 in variant_index_map.items():
            if index1 != index2:
                levenshtein_distance = levenshtein(variant1, variant2)
                if levenshtein_distance < distance_threshold:
                    G.add_edge(index1, index2)
                    similar_variants.append(variant2)
            variant_string = ','.join(variant1)
        similarity_dict[variant_string] = similar_variants

    print('Similar variants dictionary: ' + str(similarity_dict))

    pos = nx.spring_layout(G, k=5/math.sqrt(G.order()), seed=94)
    nx.draw(G, pos, with_labels=True)
    uid = uuid.uuid4()
    plt.savefig('detection/static/detection/figures/similarity_graph_' + str(uid) + '.png')
    plt.clf()

    surprising_instances = []
    for variant, similar_variants in similarity_dict.items():
        variants_subset = []
        variants_subset.append(variant)
        for similar_variant in similar_variants:
            variant_string = ','.join(similar_variant)
            variants_subset.append(variant_string)

        filtered_log = variants_filter.apply(event_log, variants_subset)

        parameters = {}
        parameters["add_case_identifier_column"] = True
        parameters["enable_case_duration"] = True
        parameters["enable_succ_def_representation"] = False
        data, feature_names = log_to_features.apply(filtered_log, variant=log_to_features.Variants.TRACE_BASED, parameters = parameters)
        pd_data = pd.DataFrame(data, columns=feature_names)
        #print(pd_data.head())
        avg_attribute = pd_data['@@caseDuration'].mean()
        print('Mean case duration: ' + str(avg_attribute))
        
        cases_for_variant = variants_case_dict[variant]
        for case in cases_for_variant:
            # if attribute case duration
            #print(case)
            case_duration = get_case_duration(filtered_log, case.attributes['concept:name'])
            if abs(case_duration - avg_attribute) > performance_threshold:
                surprising_instances.append(case)
        #print('Cases for variant: ' + str(variant) + ': ' + str(cases_for_variant))

    print('There are ' + str(len(surprising_instances)) + ' surprising instances')

    similar_variants_dict = {}
    var_index = 1
    for variant in all_variants[:1]:
        num_neighbors = 0
        duration_sum = 0
        neighbors = nx.single_source_shortest_path_length(G, var_index, cutoff = edge_threshold)
        variant_string = ','.join(variant)
        print('Neighbor of ' + str(variant) + ': ' + str(neighbors))
        var_index = var_index + 1

    for key, value in variants_case_dict.items():
        variant_string = key
        variant_list = variant_string.split(',')

    print('Dict: ' + str(list(variants_case_dict)[-1]))

    least_frequent_variant = 'Confirmation of receipt,T06 Determine necessity of stop advice,T02 Check confirmation of receipt,T04 Determine confirmation of receipt,T05 Print and send confirmation of receipt,T10 Determine necessity to stop indication,T16 Report reasons to hold request,T17 Check report Y to stop indication,T19 Determine report Y to stop indication,T20 Print report Y to stop indication'
    
    #least_freq = variants_util.get_variant_from_trace(least_frequent_variant)
    #print(least_freq)

    print('Value: ' + str(variants_case_dict[least_frequent_variant][0]))
    print('Attributes: ' + str(variants_case_dict[least_frequent_variant][0].attributes))
    print('ID: ' + str(variants_case_dict[least_frequent_variant][0].attributes['concept:name']))

    #for trace in event_log[:3]:
    #    print(trace)
    #    print(trace['events'])

    
    
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
