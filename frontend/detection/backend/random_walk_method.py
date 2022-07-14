import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .feature_extraction import extract_features, Variants
from .log_to_features import transform_log_to_feature_table
from .variants_filter import get_variants, apply_variant_filter

from .supervised_learning_method import detect_surprising_instances as detect_instances_dt

def apply_random_walk(request, context):
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

def generate_graph(pd_data_event_log, descriptive_feature_names, target_feature_name):

    # descriptive_feature_names.append('trace:department@Experts')

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

    print('Pandas dict: ')
    #print(pd_data_event_log.to_dict())
    pd_data_dict = pd_data_event_log.to_dict('index')
    #for key, value in pd_data_dict.items():
    #    print(value['@@case_id_column'])

    G = nx.Graph()
    #color_map = []
    
    case_ids = []

    k = 0
    num_cases = 1700
    for key, value in pd_data_dict.items():
        if k <= num_cases:
            case_ids.append(value['@@case_id_column'])
            #print('trace:department@Experts' + str(value['trace:department@Experts']))
            G.add_node(value['@@case_id_column'])
            k = k + 1

    for feature_name in descriptive_feature_names:
        G.add_node(feature_name)
        k = 0
        for key, value in pd_data_dict.items():
            if k <= num_cases:
                if value[feature_name] == 1:
                    G.add_edge(value['@@case_id_column'], feature_name, weight=0.5)
                k = k + 1            

    print('Plotting graph:')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
    plt.savefig('detection/static/detection/figures/random_walk.png')
    #plt.show()

    #similar_cases_random_walk = random_walk(G, case_ids, len(descriptive_feature_names), 4)

def random_walk(G, case_ids, num_features, walk_length):
    print('Starting random walk')
    similarity_dict = {}
    # let networkx return the adjacency matrix A
    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype = np.float64)
    # let's evaluate the degree matrix D
    D = np.diag(np.sum(A, axis=0))
    # ...and the transition matrix T
    T = np.dot(np.linalg.pinv(D),A)
    # define the starting node, say the 0-th
    i = 0
    for case_id in case_ids:
        zeros = np.zeros(len(case_ids) + num_features)
        zeros[i] = 1
        p = zeros.reshape(-1,1)
        for k in range(walk_length):
            # evaluate the next state vector
            p = np.dot(T,p)
        #print(p)
        #ind = np.argpartition(p, -3)[-3:]
        #print('Index: ' + str(ind))
        #top4 = p[ind]
        #print('Top 4: ' + str(top4))
        max_index_col = np.argmax(p, axis=0)
        #print('Index: ' + str(max_index_col) + ' value: ' + str(p[max_index_col]))
        result = np.where(p >= p[max_index_col])
        result = [case_ids[x] for x in result[0]]
        similarity_dict[case_id] = result
        #print('Cases similar to case ' + str(case_id) + ': ' + str(result))
        i = i + 1
    return similarity_dict

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

def detect_surprising_instances(request, context):
    log_path = request.session['log_path']
    event_log = import_and_filter_event_log(request)

    pd_data_event_log, feature_names = transform_log_to_feature_table(log_path, event_log)

    print(pd_data_event_log.head())
    print('Rows: ' + str(len(pd_data_event_log.index)))

    # Feature extraction
    target_feature = request.session['target_attribute']
    variant = Variants.PEARSON
    threshold = 0.1
    feature_list = extract_features(pd_data_event_log=pd_data_event_log, variant=variant, target_feature=target_feature, threshold=threshold)
    print('Found ' + str(len(feature_list)) + ' features with a correlation higher than ' + str(threshold))
    print(feature_list)

    # Detect surprising instances
    model_threshold = int(request.session['target_attribute_threshold'])
    
    if request.session['target_attribute_type'] == 'categorical':
        model_strategy = 'dt_classification'
    else:
        model_strategy = 'dt_regression'

    generate_graph(pd_data_event_log, feature_list, '@@caseDuration')

    request.session['supervised_learning_correlation_threshold'] = 0.1
    request.session['supervised_learning_correlation_method'] = 'Pearson'
    request.session['max_depth_decision_tree'] = 5
    request.session['target_attribute_type'] = 'numerical'
    context, event_log, surprising_instances = detect_instances_dt(request, context)

    context['random_walk_graph'] = 'detection/figures/random_walk.png'
    context['surprising_instances'] = surprising_instances

    return context, event_log, surprising_instances