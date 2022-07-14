import numpy as np
import networkx as nx
import math
import uuid
import matplotlib.pyplot as plt
# switch backend
plt.switch_backend('agg')
import pandas as pd
import datetime

from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.variants.log import get
from pm4py.util import variants_util
from pm4py.stats import get_case_duration
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features

from .variants_filter import get_variants, apply_variant_filter
from .models import SurprisingInstance, Condition

def apply_similarity_graph(request, context):
    if 'variant_select' in request.POST:
        selected_variant_id = request.POST['variant_select']
        print('Selected leaf: ' + str(selected_variant_id))
        request.session['selected_variant_id'] = selected_variant_id

    context, event_log, surprising_instances = detect_surprising_instances(request, context)
    context = calculate_surprising_instance_statistics(event_log, surprising_instances, context)
    return context

def calculate_surprising_instance_statistics(all_cases, surprising_instances, context):
    surprising_instance_count = len(surprising_instances)
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

def filter_results_by_variant_id(request, surprising_instances, context):
    variant_ids = []
    for instance in surprising_instances:
        current_variant_id = int(instance.leaf_id)
        if current_variant_id not in variant_ids:
            variant_ids.append(current_variant_id)

    selected_variant_id = int(request.session.get('selected_variant_id', variant_ids[0]))
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
            if current_variant_id == selected_variant_id:
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
            if current_variant_id == selected_variant_id:
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
    context['selected_variant_id'] = selected_variant_id
    request.session['selected_variant_id'] = selected_variant_id
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
    if selected_variant_id in better_variant_id_dict:
        context['num_better_leaf'] = better_variant_id_dict[selected_variant_id]
    else:
        context['num_better_leaf'] = 0
    if selected_variant_id in worse_variant_id_dict:
        context['num_worse_leaf'] = worse_variant_id_dict[selected_variant_id]
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
    request.session['variant_ids'] = variant_ids
    return context

def detect_surprising_instances(request, context):
    parameters = {}
    event_log = import_and_filter_event_log(request)

    variants_count = case_statistics.get_variant_statistics(event_log)
    variants_case_dict = get.get_variants(event_log)

    all_variants = []
    index_variant_map = {}
    var_index = 1
    for variant in variants_count:
        variant_string = variant['variant']
        index_variant_map[variant_string] = var_index
        variant_list = variant_string.split(',')
        all_variants.append(variant_list)
        var_index = var_index + 1

    # Parameters
    distance_threshold = int(request.session['similarity_graph_distance_max'])
    edge_threshold = 2

    target_feature = request.session['target_attribute']
    performance_threshold = int(request.session['target_attribute_threshold'])
    if request.session['target_attribute_type'] == 'categorical':
        model_strategy = 'classification'
    else:
        model_strategy = 'regression'

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

    # print('Similar variants dictionary: ' + str(similarity_dict))

    pos = nx.spring_layout(G, k=5/math.sqrt(G.order()), seed=94)
    nx.draw(G, pos, with_labels=True)
    uid = uuid.uuid4()
    variant_graph_path = 'detection/figures/similarity_graph_' + str(uid) + '.png'
    context['variant_graph_path'] = variant_graph_path
    request.session['variant_graph_path'] = variant_graph_path
    plt.savefig('detection/static/' + str(variant_graph_path))
    plt.clf()

    surprising_instances = []
    for variant, similar_variants in similarity_dict.items():
        variants_subset = []
        variants_subset.append(variant)
        for similar_variant in similar_variants:
            variant_string = ','.join(similar_variant)
            variants_subset.append(variant_string)

        variant_index = index_variant_map[variant]

        filtered_log = variants_filter.apply(event_log, variants_subset)

        parameters = {}
        parameters["add_case_identifier_column"] = True
        parameters["enable_case_duration"] = True
        parameters["enable_succ_def_representation"] = False
        data, feature_names = log_to_features.apply(filtered_log, variant=log_to_features.Variants.TRACE_BASED, parameters = parameters)
        pd_data = pd.DataFrame(data, columns=feature_names)

        #print('Features: ' + str(feature_names))
        if model_strategy == 'classification':
            # avg_attribute = pd_data[target_feature].mode()
            if target_feature in feature_names:
                avg_attribute = pd_data[target_feature].value_counts().index.tolist()[0]
            else:
                avg_attribute = 0
        else:
            avg_attribute = pd_data[target_feature].mean()
        #print('Mean case duration: ' + str(avg_attribute))
        
        cases_for_variant = variants_case_dict[variant]

        for case in cases_for_variant:
            # if attribute case duration
            #print(case)
            case_duration = get_case_duration(filtered_log, case.attributes['concept:name'])
            if model_strategy == 'classification':
                head, sep, tail = target_feature.partition(':')
                head, sep, tail = tail.partition('@')
                #target_feature_att = target_feature.replace("trace:", "")
                #target_feature_att = target_feature_att.replace("event:", "")
                print('Case attribute: ' + str(case.attributes[head]) + ' avg attribute: ' + str(avg_attribute))
                if avg_attribute != case.attributes[head]:
                    instance = SurprisingInstance(case.attributes['concept:name'], case, target_feature, avg_attribute, case.attributes[head], variant_index, True, [])
                    surprising_instances.append(instance)
            else:
                if abs(case_duration - avg_attribute) > performance_threshold:
                    instance = SurprisingInstance(case.attributes['concept:name'], case, target_feature, avg_attribute, case_duration, variant_index, False, [])
                    surprising_instances.append(instance)
        #print('Cases for variant: ' + str(variant) + ': ' + str(cases_for_variant))

    print('There are ' + str(len(surprising_instances)) + ' surprising instances')

    context = filter_results_by_variant_id(request, surprising_instances, context)

    selected_variant_id = int(request.session.get('selected_variant_id'))
    context['variant_sequence'] = variant_index_map[selected_variant_id]

    #context['surprising_instances'] = surprising_instances
    context['target_attribute_name'] = request.session['target_attribute']

    return context, event_log, surprising_instances

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