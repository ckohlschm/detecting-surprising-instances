import datetime
import pandas as pd

from .variants_filter import get_variants, apply_variant_filter
from .data_reader import transform_log_to_feature_table
from .situation_features import get_situation_feature_table

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

def read_session_parameters(request):
    # Detect surprising instances
    detector_function = request.session['detector_function']
    model_threshold = None 
    if detector_function == 'threshold':
        model_threshold = int(request.session['target_attribute_threshold'])
    print('Using Detector function ' + str(detector_function) + ' for surprising instance detection (threshold: ' + str(model_threshold) + ')')
    
    if request.session['target_attribute_type'] == 'categorical':
        model_strategy = 'categorical'
    else:
        model_strategy = 'numerical'
    
    situation_type = request.session['situation_type']

    if situation_type == 'event':
        target_feature = str(request.session['selected_activity']) + '_' + str(request.session['target_attribute'])
    else:
        target_feature = str(request.session['target_attribute'])

    return target_feature, detector_function, model_threshold, model_strategy, situation_type

def transform_event_log_to_situations(request, situation_type):
    log_path = request.session['log_path']
    event_log = import_and_filter_event_log(request)
    if situation_type == 'event':
        # situation_activities = request.session['situation_activities']
        situation_target_activity = request.session['selected_activity']
        target_feature = request.session['target_attribute']
        
        feature_list = []
        event_features = request.session['event_features']
        for feature in event_features:
            # for activity in situation_activities:
            feature_list.append((feature, situation_target_activity))
        feature_list.append(('@@case_id_column', situation_target_activity))
        print('Using features: ' + str(feature_list))

        pd_data_event_log = get_situation_feature_table(log=event_log, situation_type='event', situation_activities=[situation_target_activity], target_feature=(target_feature, situation_target_activity), features=feature_list, sensitive_feature=(None, None))
        #pd_data_event_log.to_csv('situations.csv')
        columns_to_encode = []
        for col in pd_data_event_log.columns:
            # convert datetime objects to the right format
            if pd_data_event_log[col].dtype == 'object' and len(set([type(obj) for obj in pd_data_event_log[col]])) == 1 and type(pd_data_event_log[col][0]) == datetime.datetime:
                pd_data_event_log[col] = pd.to_datetime(pd_data_event_log[col], utc=True)
            if 'elapsed time' in col:
                pd_data_event_log[col] = pd_data_event_log[col].dt.total_seconds()
            if pd_data_event_log[col].dtype in ["object", "category"]:
                columns_to_encode.append(col)
        
        target_feature_name = str(request.session['selected_activity']) + '_' + str(request.session['target_attribute'])
        id_column_name = str(request.session['selected_activity']) + '_' + str('@@case_id_column')
        if id_column_name in columns_to_encode:
            columns_to_encode.remove(id_column_name)
        if target_feature_name in columns_to_encode:
            columns_to_encode.remove(target_feature_name)

        # One-Hot-Encode data
        pd_data_event_log = pd.get_dummies(pd_data_event_log, columns=columns_to_encode)
        #pd_data_event_log.to_csv('situations_one_hot_encoded.csv')
        feature_list = pd_data_event_log.columns.to_list()
        if target_feature_name in feature_list:
            feature_list.remove(target_feature_name)
        if id_column_name in feature_list:
            feature_list.remove(id_column_name)

        # shift column '@@case_id_column' to first position and update name
        first_column = pd_data_event_log.pop(id_column_name)
        pd_data_event_log.insert(0, id_column_name, first_column)
        pd_data_event_log.rename(columns={id_column_name: '@@case_id_column'}, inplace=True)
    else:
        pd_data_event_log, feature_names = transform_log_to_feature_table(log_path, event_log)
        # Feature extraction
        feature_list = request.session['selected_feature_names']
        print('Using ' + str(len(feature_list)) + ' features for vicinity Detection')
        print(feature_list)

    return pd_data_event_log, event_log, feature_list


def filter_results_by_vicinity_id(request, surprising_instances, context):
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
        if request.session['target_attribute'] == '@@caseDuration' or request.session['target_attribute'] == 'elapsed time':
             context['avg_better_leaf_performance'] = datetime.timedelta(seconds=round(sum(list_better_performance) / len(list_better_performance), 0))
        else:
            context['avg_better_leaf_performance'] = 0 # round(sum(list_better_performance) / len(list_better_performance), 2)
    else:
        context['avg_better_leaf_performance'] = 0
    if len(list_worse_performance) > 0:
        if request.session['target_attribute'] == '@@caseDuration' or request.session['target_attribute'] == 'elapsed time':
             context['avg_worse_leaf_performance'] = datetime.timedelta(seconds=round(sum(list_worse_performance) / len(list_worse_performance), 0))
        else:
            context['avg_worse_leaf_performance'] = 0 # round(sum(list_worse_performance) / len(list_worse_performance), 2)
    else:
        context['avg_worse_leaf_performance'] = 0

    if len(list_all_better_performance) > 0:
        if request.session['target_attribute'] == '@@caseDuration' or request.session['target_attribute'] == 'elapsed time':
             context['avg_all_better_performance'] = datetime.timedelta(seconds=round(sum(list_all_better_performance) / len(list_all_better_performance), 0))
        else:
            context['avg_all_better_performance'] = 0 # round(sum(list_all_better_performance) / len(list_all_better_performance), 2)
    else: 
        context['avg_all_better_performance'] = 0
    if len(list_all_worse_performance) > 0:
        if request.session['target_attribute'] == '@@caseDuration' or request.session['target_attribute'] == 'elapsed time':
             context['avg_all_worse_performance'] = datetime.timedelta(seconds=round(sum(list_all_worse_performance) / len(list_all_worse_performance), 0))
        else:
            context['avg_all_worse_performance'] = 0 # round(sum(list_all_worse_performance) / len(list_all_worse_performance), 2)
    else: 
        context['avg_all_worse_performance'] = 0

    context['num_better_leaf'] = len(list_better_performance)
    context['num_worse_leaf'] = len(list_worse_performance)
    context['leaf_ids'] = leaf_ids
    return context

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

def calculate_surprisingness_index_better(row, target_feature_name, p_avg, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * affectedInstances 

def calculate_surprisingness_index_worse(row, target_feature_name, p_avg, vicinitySize, affectedInstances):
    return (abs(row[target_feature_name] - p_avg) ) * (vicinitySize - affectedInstances)

def calculate_relevance_worse(row, vicinitySize):
    return row['surprisingnessWorseIndex'] * vicinitySize

def calculate_relevance_better(row, vicinitySize):
    return row['surprisingnessBetterIndex'] * vicinitySize

def find_outliers_for_node(filtered_data, target_feature_name, upper_bound, lower_bound):
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


def find_outliers_for_node_boxplot(filtered_data, target_feature_name):
    print('Target Feature Name: ' + str(target_feature_name))
    Q1 = filtered_data[target_feature_name].quantile(0.25)
    Q3 = filtered_data[target_feature_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 *IQR

    return find_outliers_for_node(filtered_data, target_feature_name, upper_bound, lower_bound)

def find_outliers_for_node_threshold(filtered_data, target_feature_name, threshold):
    print('Target Feature Name: ' + str(target_feature_name))
    mean_value = filtered_data[target_feature_name].mean()

    lower_bound = mean_value - threshold
    upper_bound = mean_value + threshold

    return find_outliers_for_node(filtered_data, target_feature_name, upper_bound, lower_bound)
    
def find_outliers_for_node_categorical(filtered_data, target_feature_name):
    expected_value = filtered_data[target_feature_name].value_counts().index.tolist()[0]
    print('Expected value in vicinity: ' + str(expected_value))

    filter_better = (filtered_data[target_feature_name] != expected_value)
    event_log_filter_better = filtered_data.loc[filter_better]
    event_log_filter_worse = None

    return event_log_filter_better, event_log_filter_worse
