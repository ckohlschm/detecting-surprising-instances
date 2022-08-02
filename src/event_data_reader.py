from pm4py.objects.log.importer.xes import importer as xes_importer
from log_to_features import convert_log_to_features
import pandas as pd

def read_BPI17():
    # BPI Challenge 2017
    event_log = xes_importer.apply('event_logs/BPI Challenge 2017.xes.gz')
    parameters = {}
    parameters["str_ev_attr"] = []
    parameters["num_ev_attr"] = []
    parameters["str_tr_attr"] = ["LoanGoal", "ApplicationType"]
    parameters["num_tr_attr"] = ["RequestedAmount"]
    parameters["activity_names_to_count"] = ["O_Create Offer"]
    return event_log, parameters

def transform_to_dataframe(event_log, parameters):
    data, feature_names = convert_log_to_features(event_log, parameters = parameters)

    pd_data = pd.DataFrame(data, columns=feature_names)
    return pd_data

def read_event_log(event_log_name = 'BPI17'):
    if event_log_name == 'BPI17':
        event_log, parameters = read_BPI17()
    else:
        print('Undefined event log name')
        return None
    parameters["add_case_identifier_column"] = True
    parameters["enable_case_duration"] = True
    parameters["enable_succ_def_representation"] = False
    parameters["enable_max_concurrent_events"] = False
    parameters["enable_count_activities"] = True

    pd_data = transform_to_dataframe(event_log=event_log, parameters=parameters)

    return pd_data, event_log
