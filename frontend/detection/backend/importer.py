""" Use import_file(path, filter=True) to import xes or csv with pmy4py functions.
    Returns error if not a csv or xes file.
    activated filter = True (default) will remove traces without most frequent start and end activity
    For CSV: Expected Column names are
    case:concept:name -> case ID, concept:name -> activity, time:timestamp -> timestamp
"""
import os

import pandas as pd
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.util import sorting


# import xes and sort the event log by timestamps and remove incomplete traces
def import_xes(path, filter=True):
    log_xes = xes_importer.apply(path)
    log_xes = sorting.sort_timestamp(log_xes)
    if filter:
        log_xes = remove_uncomplete_traces(log_xes)
    return log_xes


# import csv and sort the event log by timestamps, complete incomplete traces.
# Expected Column names are case:concept:name -> case ID,concept:name -> activity ,time:timestamp -> timestamp
def import_csv(path, filter=True):
    log_csv = pd.read_csv(path, sep=',')
    log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
    # parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'Case ID'}
    log_csv = log_csv.sort_values('time:timestamp')
    event_log = log_converter.apply(
        log_csv)
    if filter:
        event_log = remove_uncomplete_traces(event_log)
    return event_log


# Remove incomplete traces / noises :
# If trace does not contain (most frequent) start and end activity, we remove the trace.
# Thus we only allow one start and one end activity
def remove_uncomplete_traces(event_log):
    start_activity = list(start_activities_filter.get_start_activities(event_log).keys())[0]
    end_activities = list(end_activities_filter.get_end_activities(event_log).keys())

    filtered_log = end_activities_filter.apply(event_log, end_activities)
    filtered_log = start_activities_filter.apply(filtered_log, start_activity)

    cnt_removed_traces = len(event_log) - len(filtered_log)
    print('Most frequent start activity is: ' + start_activity + ' Removing all traces without that start activity..')
    print(
        'Most frequent end activity is: ' + str(end_activities) + ' Removing all traces without that end activities..')
    print('Number of removed traces: ' + str(cnt_removed_traces))
    return filtered_log


def import_file(path, filter=True):
    filename, file_extension = os.path.splitext(path)
    # file_extension = os.path.splitext(filename)[1]
    if file_extension == ".csv":
        print('Importing CSV File ' + filename + '...')
        print('Filter is set: ' + str(filter))
        return import_csv(path, filter)
    elif file_extension == ".xes":
        print('Importing CSV File ' + filename + '...')
        print('Filter is set: ' + str(filter))
        return import_xes(path, filter)
    else:
        print('Error: Please choose XES or CSV file.')
        raise ValueError('File not valid')
