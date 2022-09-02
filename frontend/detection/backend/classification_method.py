import datetime

from pm4py.statistics.attributes.log.get import get_all_trace_attributes_from_log, get_all_event_attributes_from_log

from .feature_extraction import extract_features, Variants
from .data_reader import transform_log_to_feature_table
from .variants_filter import get_variants, apply_variant_filter
from .detection import detect_surprising_instances as detect_surprising_instances_algorithm
from .root_cause_analysis import find_cause_for_instance

def apply_supervised_learning(request, context):
    if 'leaf_select' in request.POST:
        selected_leaf_id = request.POST['leaf_select']
        print('Selected leaf: ' + str(selected_leaf_id))
        request.session['selected_leaf_id'] = selected_leaf_id
    
    context, event_log, surprising_instances_len = detect_surprising_instances(request, context)
    context = calculate_surprising_instance_statistics(event_log, surprising_instances_len, context)
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

def label_surprising_better(row, surprising_instances):
    for instance in surprising_instances:
        if row['@@case_id_column'] == instance.id:
            if instance.actual_data < instance.target_data:
                return True
    return False

def label_surprising_worse(row, surprising_instances):
    for instance in surprising_instances:
        if row['@@case_id_column'] == instance.id:
            if instance.actual_data > instance.target_data:
                return True
    return False

def root_cause_analysis_dt(surprising_instances, selected_leaf_id, pd_data_event_log, descriptive_feature_names, target_feature_name):
    pd_data_event_log['@@surprising_better'] = pd_data_event_log.apply (lambda row: label_surprising_better(row, surprising_instances), axis=1)
    pd_data_event_log['@@surprising_worse'] = pd_data_event_log.apply (lambda row: label_surprising_worse(row, surprising_instances), axis=1) 
    
    # Root cause analysis
    for instance in surprising_instances:
        if str(instance.leaf_id) == str(selected_leaf_id):
            print("Conditions: " + str(instance.conditions))
            find_cause_for_instance(pd_data_event_log, instance, ['@@surprising_better'], 'dt_rca_better')
            find_cause_for_instance(pd_data_event_log, instance, ['@@surprising_worse'], 'dt_rca_worse')
            break

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
    decision_tree_max_depth = int(request.session['max_depth_decision_tree'])
    feature_list = request.session['selected_feature_names'] # ['case:LoanGoal@Boat', 'case:LoanGoal@Business goal', 'case:LoanGoal@Car', 'case:LoanGoal@Caravan / Camper', 'case:LoanGoal@Debt restructuring', 'case:LoanGoal@Existing loan takeover', 'case:LoanGoal@Extra spending limit', 'case:LoanGoal@Home improvement', 'case:LoanGoal@Motorcycle', 'case:LoanGoal@Not speficied', 'case:LoanGoal@Other, see explanation', 'case:LoanGoal@Remaining debt home', 'case:LoanGoal@Tax payments', 'case:LoanGoal@Unknown', 'case:ApplicationType@New credit', 'case:RequestedAmount', '@@num_occurrence_O_Create Offer'] # extract_features(pd_data_event_log=pd_data_event_log, variant=variant, target_feature=target_feature, threshold=threshold)
    print('Using ' + str(len(feature_list)) + ' features for vicinity Detection')
    print(feature_list)

    # Detect surprising instances
    detector_function = request.session['detector_function']
    model_threshold = None 
    if detector_function == 'threshold':
        model_threshold = int(request.session['target_attribute_threshold'])
    print('Using Detector function ' + str(detector_function) + ' for surprising instance detection (threshold: ' + str(model_threshold) + ')')
    
    if request.session['target_attribute_type'] == 'categorical':
        model_strategy = 'dt_classification'
    else:
        model_strategy = 'dt_regression'
    
    surprising_instances, data_by_vicinity_id = detect_surprising_instances_algorithm(pd_data_event_log=pd_data_event_log, descriptive_feature_names=feature_list, target_feature_name=target_feature, strategy=model_strategy, add_conditions=True, threshold=model_threshold, max_depth=decision_tree_max_depth, detector_function=detector_function)
    
    context = filter_results_by_leaf_id(request, surprising_instances, context)

    surprising_instances_len = get_len_surprising_instances(surprising_instances)
    #root_cause_analysis_dt(surprising_instances, request.session['selected_leaf_id'], pd_data_event_log, feature_names, target_feature)

    context['decision_tree_path'] = 'detection/figures/decision_tree.png'
    context['target_attribute_name'] = request.session['target_attribute']
    context['decision_tree_path_rca_better'] = 'detection/figures/dt_rca_better.png'
    context['decision_tree_path_rca_worse'] = 'detection/figures/dt_rca_worse.png'
    selected_leaf_id = request.session['selected_leaf_id']

    request.session['instances_in_vicinity'] = data_by_vicinity_id[selected_leaf_id].to_json(orient='split')
    return context, event_log, surprising_instances_len