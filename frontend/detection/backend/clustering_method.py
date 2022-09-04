from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .data_reader import transform_log_to_feature_table
from .models import SurprisingInstance
from .detection_util import find_outliers_for_node_boxplot, get_len_surprising_instances, filter_results_by_vicinity_id, find_outliers_for_node_threshold, find_outliers_for_node_categorical, calculate_surprising_instance_statistics, read_session_parameters, transform_event_log_to_situations

def apply_clustering(request, context):
    if 'leaf_select' in request.POST:
        selected_leaf_id = request.POST['leaf_select']
        print('Selected leaf: ' + str(selected_leaf_id))
        request.session['selected_leaf_id'] = selected_leaf_id
    
    context, event_log, surprising_instances = detect_surprising_instances(request, context)

    context = calculate_surprising_instance_statistics(event_log, surprising_instances, context)
    return context

def detect_surprising_instances(request, context):
    target_feature, detector_function, model_threshold, model_strategy, situation_type = read_session_parameters(request)

    pd_data_event_log, event_log, feature_list = transform_event_log_to_situations(request=request,situation_type=situation_type)

    k_means_number_clusters = int(request.session['k_means_number_clusters'])
    print('Target: ' + str(target_feature))
    
    surprising_instances, data_by_vicinity_id = vicinity_detection(pd_data_event_log, k_means_number_clusters, feature_list, target_feature, model_strategy, detector_function, model_threshold)

    context = filter_results_by_vicinity_id(request, surprising_instances, context)
    surprising_instances_len = get_len_surprising_instances(surprising_instances)

    context['target_attribute_name'] = request.session['target_attribute']
    if model_strategy == 'categorical':
        context['categorical'] = True
    else:
        context['categorical'] = False
    selected_leaf_id = request.session['selected_leaf_id']
    context['selected_leaf_id'] = selected_leaf_id

    request.session['instances_in_vicinity'] = data_by_vicinity_id[selected_leaf_id].to_json(orient='split')
    return context, event_log, surprising_instances_len

def vicinity_detection(event_log, k_means_number_clusters, feature_names, target_feature, strategy, detector_function, threshold):
    event_log_filtered = event_log.filter(feature_names)
    
    cols_to_normalize = []
    for feature_name in feature_names:
        if feature_name in event_log_filtered.columns:
            cols_to_normalize.append(feature_name)
    event_log_filtered[cols_to_normalize] = MinMaxScaler().fit_transform(event_log_filtered[cols_to_normalize])
    
    event_log_filtered_no_case_id = event_log_filtered.filter(feature_names)
    situations = event_log_filtered_no_case_id.values.tolist()

    kmeans = KMeans(n_clusters=k_means_number_clusters, random_state=0).fit(situations)

    k_means_labels = kmeans.labels_
    print('Found clusters: ')
    print(k_means_labels)

    event_log['cluster_id'] = k_means_labels

    surprising_instances, data_by_vicinity = surprising_instance_detection(event_log, k_means_number_clusters, target_feature, strategy, detector_function, threshold)

    return surprising_instances, data_by_vicinity

def surprising_instance_detection(event_log, k_means_number_clusters, target_feature_name, strategy, detector_function, threshold):
    surprising_instances = {}
    data_by_vicinity = {}

    for vicinity_id in range(k_means_number_clusters):
        filtered_data = event_log[event_log['cluster_id'] == vicinity_id]
        data_by_vicinity[vicinity_id] = filtered_data
        features = filtered_data.columns.tolist()
        target_feature_index = features.index(target_feature_name)

        print('Filtered data: ' + str(len(filtered_data)))
        better_performing_instances_list = []
        worse_performing_instances_list = []
        if strategy == 'categorical':
            better_performing_instances, worse_performing_instances = find_outliers_for_node_categorical(filtered_data, target_feature_name)
            expected = filtered_data[target_feature_name].value_counts().index.tolist()[0]
            for instance in better_performing_instances.values.tolist():
                better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], vicinity_id, True, []))
        else:
            if detector_function == 'threshold':
                better_performing_instances, worse_performing_instances = find_outliers_for_node_threshold(filtered_data, target_feature_name, threshold)
            else:
                better_performing_instances, worse_performing_instances = find_outliers_for_node_boxplot(filtered_data, target_feature_name)
            expected = filtered_data[target_feature_name].mean()
            for instance in better_performing_instances.values.tolist():
                better_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], vicinity_id, False, []))
            for instance in worse_performing_instances.values.tolist():
                worse_performing_instances_list.append(SurprisingInstance(instance[0], instance, target_feature_name, expected, instance[target_feature_index], vicinity_id, False, []))

        surprising_instances[vicinity_id] = (vicinity_id, better_performing_instances, worse_performing_instances, better_performing_instances_list, worse_performing_instances_list)

    return surprising_instances, data_by_vicinity
