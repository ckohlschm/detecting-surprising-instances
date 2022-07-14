import pandas as pd
import datetime
from statistics import mean
import uuid

from pm4py.statistics.traces.generic.log import case_statistics

from .model_detection import discover_pn_model, discover_bpmn_model
from .log_to_features import transform_log_to_features
from .variants_filter import get_variants, apply_variant_filter

def import_log_discover_features_and_model(request, context):
    # Reset selected leaf id in case we want to discover another decision tree
    if 'selected_leaf_id' in request.session:
        del request.session['selected_leaf_id']
    if 'selected_variant_id' in request.session:
        del request.session['selected_variant_id']
    # First, check for POST requests. Depending on request, filter data or just update views
    context = check_for_post_requests(request, context)
    
    return context
    
def check_for_post_requests(request, context):
    if request.method == 'POST':
        print('POST request detected')
        context = check_for_variant_type_change(request, context)
        context = check_for_variant_filter_request(request, context)
        context = check_for_model_change_request(request, context)
        context = check_for_table_page_request(request, context)
    else:
        if request.session.get('initial_run', True):
            # Initial load
            print('Initial load')
            log_path = request.session['log_path']
            request.session['variant_filter_strategy'] = 'most_common_variant'
            event_log, all_variants, context = discover_variants(request, log_path, context)
            default_selection = [key for key in all_variants.keys() if key <= 10]
            if len(all_variants.keys()) > 10:
                default_selection.append('Other')
            request.session['selected_variants'] = default_selection
            context['selected_variants'] = default_selection
            # Extract features
            event_log_features, feature_names = transform_log_to_features(log_path, event_log)
            request.session['event_log_features'] = event_log_features
            request.session['feature_names'] = feature_names
            feature_names.remove('@@case_id_column')
            context['feature_names'] = feature_names
            # Case duration
            context, avg_case_duration = calculate_log_statistics(context, event_log)
            request.session['avg_case_duration'] = avg_case_duration
            # Discover Process Model
            context = discover_model(request, context, log_path, event_log)
            # Parse previously extracted features
            context = parse_situation_feature_data(request, context)
            request.session['initial_run'] = False
        else:
            print('Using session data')
            context['variant_filter_strategy'] = request.session['variant_filter_strategy']
            variants_json_data = request.session['variants_pd_data']
            variants_pd_data = pd.read_json(variants_json_data, orient='split')
            context['variants_pd_data'] = variants_pd_data
            context['variantsdatapiechart'] = request.session['variantsdatapiechart']
            context['selected_variants'] = request.session['selected_variants']
            context['feature_names'] = request.session['feature_names']
            context['model_name'] = request.session['model_name']
            context['model_path'] = request.session['model_path']
            context['avg_case_duration'] = request.session['avg_case_duration']
            context = parse_situation_feature_data(request, context)

    context = select_parameters_for_method(request, context)

    return context

def select_parameters_for_method(request, context):
    context['target_attribute'] = request.session.get('target_attribute', '@@caseDuration')
    context['target_attribute_type'] = request.session.get('target_attribute_type', 'numerical')
    context['target_attribute_threshold'] = request.session.get('target_attribute_threshold', 1000000)
    context['supervised_learning_correlation_method'] = request.session.get('supervised_learning_correlation_method', 'supervised_learning_correlation_method_pearson')
    context['max_depth_decision_tree'] = request.session.get('max_depth_decision_tree', 3)
    context['similarity_graph_distance_function'] = request.session.get('similarity_graph_distance_function', 'similarity_graph_distance_function_levenshtein')
    context['similarity_graph_distance_function'] = request.session.get('similarity_graph_distance_max', 1)
    context['supervised_learning_correlation_threshold'] = request.session.get('supervised_learning_correlation_threshold', 0.1)
    context['similarity_graph_distance_max'] = request.session.get('similarity_graph_distance_max', 1)

    return context

def filter_event_data_and_discover_model(request, context):
    log_path = request.session['log_path']
    event_log, all_variants, context = discover_variants(request, log_path, context)
    # Filter event log
    selected_variants = request.session.get('selected_variants', None)
    if not selected_variants:
        default_selection = [key for key in all_variants.keys() if key <= 10]
        if len(all_variants.keys()) > 10:
            default_selection.append('Other')
        request.session['selected_variants'] = default_selection
        selected_variants = default_selection
    print('Request Selected variants: ' + str(selected_variants))
    filtered_log = apply_variant_filter(event_log, all_variants, selected_variants)
    # Extract features
    event_log_features, feature_names = transform_log_to_features(log_path, filtered_log)
    request.session['event_log_features'] = event_log_features
    feature_names.remove('@@case_id_column')
    context['feature_names'] = feature_names
    # Case duration
    context, avg_case_duration = calculate_log_statistics(context, filtered_log)
    request.session['avg_case_duration'] = avg_case_duration
    # Discover Process Model
    context = discover_model(request, context, log_path, filtered_log)
    # Parse previously extracted features into table
    context = parse_situation_feature_data(request, context)
  
    return context

def check_for_variant_type_change(request, context):
    if 'variant_type' in request.POST:
        if 'most_common_variant' in request.POST['variant_type']:
            request.session['variant_filter_strategy'] = 'most_common_variant'
        elif 'least_common_variant' in request.POST['variant_type']:
            request.session['variant_filter_strategy'] = 'least_common_variant'
        elif 'longest_throughput_time' in request.POST['variant_type']:
            request.session['variant_filter_strategy'] = 'longest_throughput_time'
        elif 'shortest_throughput_time' in request.POST['variant_type']:
            request.session['variant_filter_strategy'] = 'shortest_throughput_time'
        if request.session.get('selected_variants', None):
            del request.session['selected_variants']
        context = filter_event_data_and_discover_model(request, context)
    variant_filter_strategy = request.session.get('variant_filter_strategy', 'most_common_variant')
    context['variant_filter_strategy'] = variant_filter_strategy
    return context

def check_for_variant_filter_request(request, context):
    if 'filter_variants' in request.POST:
        selected_variant_result = []
        for item in request.POST:
            if item.startswith('variantcheck-'):
                item = item.replace('variantcheck-','',1)
                variant_id = item
                if variant_id != 'Other':
                    variant_id = int(item)
                selected_variant_result.append(variant_id)
        request.session['selected_variants'] = selected_variant_result
        print('Selected Variants: ' + str(selected_variant_result))
        context = filter_event_data_and_discover_model(request, context)
    selected_variants = request.session.get('selected_variants', None)
    context['selected_variants'] = selected_variants
    return context

def check_for_model_change_request(request, context):
    if 'pn' in request.POST or 'bpmn' in request.POST:
        context = filter_event_data_and_discover_model(request, context)

    model_name = request.session.get('model_name', 'Petri-Net')
    context['model_name'] = model_name
    model_path = request.session.get('model_path', None)
    context['model_path'] = model_path
    return context

def check_for_table_page_request(request, context):
    for item in request.POST:
        if 'Previous' or 'Next' in item:
            print('Using session data')
            context['variant_filter_strategy'] = request.session['variant_filter_strategy']
            variants_json_data = request.session['variants_pd_data']
            variants_pd_data = pd.read_json(variants_json_data, orient='split')
            context['variants_pd_data'] = variants_pd_data
            context['variantsdatapiechart'] = request.session['variantsdatapiechart']
            context['selected_variants'] = request.session['selected_variants']
            context['feature_names'] = request.session['feature_names']
            context['model_name'] = request.session['model_name']
            context['model_path'] = request.session['model_path']
            context['avg_case_duration'] = request.session['avg_case_duration']
            context = parse_situation_feature_data(request, context)
        else:
            page_list = request.session['features_table_page_list']
            for page in page_list:
                if str(page) == item:
                    context['variant_filter_strategy'] = request.session['variant_filter_strategy']
                    variants_json_data = request.session['variants_pd_data']
                    variants_pd_data = pd.read_json(variants_json_data, orient='split')
                    context['variants_pd_data'] = variants_pd_data
                    context['variantsdatapiechart'] = request.session['variantsdatapiechart']
                    context['selected_variants'] = request.session['selected_variants']
                    context['feature_names'] = request.session['feature_names']
                    context['model_name'] = request.session['model_name']
                    context['model_path'] = request.session['model_path']
                    context['avg_case_duration'] = request.session['avg_case_duration']
                    context = parse_situation_feature_data(request, context)

    return context

def discover_variants(request, log_path, context):
    # Discover variants in event data
    variant_filter_strategy = request.session.get('variant_filter_strategy', 'most_common_variant')
    event_log, variants_pd_data, all_variants, variantsdata_piechart = get_variants(log_path, variant_filter_strategy)
    
    variants_json = variants_pd_data.to_json(orient="split")
    request.session['variants_pd_data'] = variants_json
    context['variants_pd_data'] = variants_pd_data
    
    request.session['variantsdatapiechart'] = variantsdata_piechart
    context['variantsdatapiechart'] = variantsdata_piechart

    return event_log, all_variants, context

def discover_model(request, context, log_path, event_log=None):
    model_name = request.session.get('model_name', 'Petri-Net')
    model_path = request.session.get('model_path', None)
    uid = uuid.uuid4()
    if model_path:
        if 'pn' in request.POST:
            model_name = 'Petri-Net'
            model_path = discover_pn_model(log_path, 'pn' + str(uid), event_log)
            request.session['model_path'] = model_path
        elif 'bpmn' in request.POST:
            model_name = 'BPMN'
            model_path = discover_bpmn_model(log_path, 'bpmn' + str(uid), event_log)
            request.session['model_path'] = model_path
        elif model_name == 'Petri-Net':
            model_path = discover_pn_model(log_path, 'pn' + str(uid), event_log)
            request.session['model_path'] = model_path
        elif model_name == 'BPMN':
            model_path = discover_bpmn_model(log_path, 'bpmn' + str(uid), event_log)
            request.session['model_path'] = model_path
    else:
        # In first run discover petri net
        model_name = 'Petri-Net'
        print('Initial run discovery')
        log_path = request.session['log_path']
        model_path = discover_pn_model(log_path, 'pn' + str(uid), event_log)
        
    request.session['model_name'] = model_name
    context['model_name'] = model_name
    request.session['model_path'] = model_path
    context['model_path'] = model_path
    return context

def calculate_log_statistics(context, event_log):
    all_case_durations = case_statistics.get_all_casedurations(event_log)
    avg_variant_case_duration = mean(all_case_durations)
    avg_variant_case_duration = round(avg_variant_case_duration, 0)
    duration = datetime.timedelta(seconds=avg_variant_case_duration)
    avg_case_duration = str(duration)
    context['avg_case_duration'] = avg_case_duration
    return context, avg_case_duration

def parse_situation_feature_data(request, context):
    # convert log to features
    if request.session.get('event_log_features', None):
        json_data = request.session['event_log_features']
        pd_data = pd.read_json(json_data, orient='split')
        print("Successfully parsed json data: ")
        print(pd_data.head())
        # Parse feature table
        context = create_features_table(request, context, pd_data)
    else:
        context['features_table_error_msg'] = 'No event log features found'
    return context

def create_features_table(request, context, pd_data):
    active_page = int(request.session.get('features_table_active_page', 1))

    offset = 0
    offset_length = 10
    page_list = []
    data_length = len(pd_data.index)
    print('There are ' + str(data_length) + ' rows in the dataset')
    page_length = round(data_length/offset_length)
    if page_length == 0:
        page_length = 1
    page_list = list(range(1, page_length + 1))
    print(page_list)

    # Active page does not exist anymore after filter
    if active_page > page_length:
        active_page = 1

    # Update active page
    if request.method == 'POST':
        for item in request.POST:
            if 'Previous' in item:
                if active_page > 1:
                    active_page = active_page - 1
            elif 'Next' in item:
                if active_page < len(page_list):
                    active_page = active_page + 1
            else:
                for page in page_list:
                    if str(page) == item:
                        active_page = int(item)
        request.session['features_table_active_page'] = active_page
        print('Updating active page to: ' + str(active_page))

    # Numbers below table
    if len(page_list) > 10:
        new_list = []
        for page in page_list[:2]:
            new_list.append(page)
        if active_page >= 5:
            new_list.append('...')
        for page in page_list[active_page-2:active_page]:
            if page not in new_list:
                new_list.append(page)
        for page in page_list[active_page:active_page+1]:
            if page not in new_list:
                new_list.append(page)
        if active_page <= len(page_list) - 4:
            new_list.append('...')
        for page in page_list[-2:]:
            if page not in new_list:
                new_list.append(page)
        page_list = new_list
    offset = (active_page - 1) * offset_length
    print('Offset: ' + str(offset))
    print(pd_data.head())

    # Calculate current data entries
    selected_rows = pd_data.iloc[offset:offset+offset_length]
    #pd_data = df.to_html(classes='table table-hover table-centered table-nowrap mb-0 rounded border-0', table_id='example')

    context['features_table_pd_data'] = selected_rows
    context['features_table_data_length'] = data_length
    if data_length < offset_length :
        context['features_table_offset_length'] = data_length
    else:
        context['features_table_offset_length'] = offset_length
    context['features_table_active_page'] = active_page
    context['features_table_page_list'] = page_list
    request.session['features_table_page_list'] = page_list
    return context