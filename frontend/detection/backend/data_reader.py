from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
import pandas as pd
from .importer import import_file
from .log_to_features import convert_log_to_features

def transform_to_dataframe(event_log, parameters):
    data, feature_names = convert_log_to_features(event_log, parameters = parameters)
    #data, feature_names = log_to_features.apply(event_log, variant=log_to_features.Variants.TRACE_BASED, parameters = parameters)
    #print(feature_names)

    pd_data = pd.DataFrame(data, columns=feature_names)
    return pd_data, feature_names

def transform_log_to_features(log_path, event_log=None):
    parameters = {}
    parameters["add_case_identifier_column"] = True
    parameters["enable_case_duration"] = True
    parameters["enable_succ_def_representation"] = False
    #parameters["enable_resource_workload"] = True
    parameters["enable_max_concurrent_events"] = True
    parameters["enable_count_activities"] = True
    
    if not event_log:
        event_log = import_file(path=log_path, filter=False)
    pd_data, feature_names = transform_to_dataframe(event_log=event_log, parameters=parameters)
    print('Extracted features: ' + str(pd_data.head()))
    result = pd_data.to_json(orient="split")
    return result, feature_names

def transform_log_to_feature_table(log_path, event_log=None):
    parameters = {}
    parameters["add_case_identifier_column"] = True
    parameters["enable_case_duration"] = True
    parameters["enable_succ_def_representation"] = False
    #parameters["enable_resource_workload"] = True
    parameters["enable_max_concurrent_events"] = True
    parameters["enable_count_activities"] = True
    
    if not event_log:
        event_log = import_file(path=log_path, filter=False)
    pd_data, feature_names = transform_to_dataframe(event_log=event_log, parameters=parameters)
    return pd_data, feature_names