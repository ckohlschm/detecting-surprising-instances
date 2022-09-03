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

def read_common_attributes(request):
    # Situation Type
    situation_type = request.POST['situation_type']
    request.session['situation_type'] = situation_type
    print('Selected situation type: ' + str(situation_type))

    # For Event situations select activity
    if 'selected_activity' in request.POST:
        selected_activity = request.POST['selected_activity']
        request.session['selected_activity'] = selected_activity
        print('Selected event activity: ' + str(selected_activity))

    # Target Feature Name
    if situation_type == 'event':
        target_attribute_name = request.POST['target_attribute_event']
        request.session['target_attribute'] = target_attribute_name
        print('Selected target attribute: ' + str(target_attribute_name))
    else: 
        target_attribute_name = request.POST['target_attribute_case']
        request.session['target_attribute'] = target_attribute_name
        print('Selected target attribute: ' + str(target_attribute_name))

    # Target Feature Type
    if target_attribute_name in ['next activity', 'previous activity', '@@delay']:
        target_attribute_type = 'categorical'
    else:
        target_attribute_type = 'numerical'
    request.session['target_attribute_type'] = target_attribute_type
    print('Selected target attribute type: ' + str(target_attribute_type))

    # Selected features
    if situation_type == 'event':
        selected_feature_names = request.POST.getlist('event_features')
        request.session['event_features'] = selected_feature_names
        print('Selected Event feature names: ' + str(selected_feature_names))
        #situation_activities = request.POST.getlist('situation_activities')
        #request.session['situation_activities'] = situation_activities
        #print('Selected situation activity names: ' + str(situation_activities))
    else: 
        # Descriptive features
        selected_feature_names =  request.POST.getlist('descriptive_attributes')
        request.session['selected_feature_names'] = selected_feature_names
        print('Selected feature names: ' + str(selected_feature_names))

    # Detector Function + Threshold
    detector_funtion = request.POST['detector_function']
    if detector_funtion == 'threshold':
        target_attribute_threshold = request.POST['target_attribute_threshold']
        request.session['target_attribute_threshold'] = target_attribute_threshold
        print('Selected threshold : ' + str(target_attribute_threshold))
    request.session['detector_function'] = detector_funtion
    print('Selected Detector function: ' + str(detector_funtion))

def read_parameters_similarity_graph(request):
    read_common_attributes(request)
   
    similarity_graph_distance_function = request.POST['similarity_graph_distance_function']
    request.session['similarity_graph_distance_function'] = similarity_graph_distance_function
    print('Selected distance function: ' + str(similarity_graph_distance_function))

    similarity_graph_distance_max = request.POST['similarity_graph_distance_max']
    request.session['similarity_graph_distance_max'] = similarity_graph_distance_max
    print('Selected max distance threshold: ' + str(similarity_graph_distance_max))

def read_parameters_clustering(request):
    read_common_attributes(request)

    vicinity_detection_method = request.POST['vicinity_detection_method']
    request.session['vicinity_detection_method'] = vicinity_detection_method
    print('Selected vicinity detection method: ' + str(vicinity_detection_method))

    k_means_number_clusters = request.POST['k_means_number_clusters']
    request.session['k_means_number_clusters'] = k_means_number_clusters
    print('Selected number of clusters: ' + str(k_means_number_clusters))

def read_parameters_supervised_learning(request):
    read_common_attributes(request)

    vicinity_detection_method = request.POST['vicinity_detection_method']
    request.session['vicinity_detection_method'] = vicinity_detection_method
    print('Selected vicinity detection method: ' + str(vicinity_detection_method))

    max_depth_decision_tree = request.POST['max_depth_decision_tree']
    request.session['max_depth_decision_tree'] = max_depth_decision_tree
    print('Selected decision tree max depth: ' + str(max_depth_decision_tree))
    