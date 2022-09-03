"""
A tool to extract a situation feature table from an event log.
"""

import numpy as np
import pandas as pd
from pm4py.objects.log.obj import Trace, EventLog

situation_feature_list = ['trace delay',
                          'deviation',
                          'trace duration',
                          'elapsed time',
                          'next activity',
                          'previous activity',
                          '@@case_id_column']


def get_situations_from_trace(trace: Trace,
                              activities: list = None,
                              trace_situation: bool = False,
                              activity_att: str = 'concept:name'):
    """Generate situations from a trace.

    :param trace: the trace to generate situations from
    :param activities: a list of all activities to be considered for situations
    :param trace_situation: a boolean value indicating that only trace situations are considered
    :param activity_att: the attribute carrying the activity name in the trace"""
    if trace_situation:
        yield trace
    else:
        situation = []
        for event in trace:
            situation.append(event)
            if event[activity_att] in activities:
                yield Trace(situation.copy(), attributes=trace.attributes)


def enrich_event_log(log: EventLog, features: list):
    """Enrich an event log by a given set of features

    :param log: The event log to enrich
    :param features: A list of features as (attribute, activity)-tuples"""
    # add some useful features
    for trace in log:
        trace.attributes['time:start'] = trace[0]['time:timestamp']
        trace.attributes['time:finish'] = trace[-1]['time:timestamp']
        trace.attributes['time:trace-duration'] = trace.attributes['time:finish'] - trace.attributes['time:start']
    # add the desired features
    for feature, _ in features:
        if feature == 'trace delay':
            values = [trace.attributes['time:trace-duration'].total_seconds() for trace in log]
            sorted_values = sorted(values)
            iqr = sorted_values[int(0.75 * len(sorted_values))] - sorted_values[int(0.25 * len(sorted_values))]
            threshold = sorted_values[int(0.75 * len(sorted_values))] + 1.5 * iqr
            for trace in log:
                trace.attributes['trace delay'] = trace.attributes['time:trace-duration'].total_seconds() > threshold
        elif feature == 'deviation':
            for trace in log:
                trace.attributes['deviation'] = not trace.attributes['replay']['trace_is_fit']
        elif feature == '@@case_id_column':
            for trace in log:
                trace.attributes['@@case_id_column'] = trace.attributes['concept:name']
        elif feature == 'next activity':
            for trace in log:
                next_ = 'END'
                for event in reversed(trace):
                    event['next activity'] = next_
                    next_ = event['concept:name']
        elif feature == 'previous activity':
            for trace in log:
                previous = 'START'
                for event in trace:
                    event['previous activity'] = previous
                    previous = event['concept:name']


def compute_feature(situation: Trace, feature: str, activity: str = None):
    """Compute a desired feature of a situation trace

    :param situation: the situation for which to compute a feature
    :param feature: the desired feature
    :param activity: in case of an event level feature the corresponding activity is required"""
    if feature == 'deviation':
        return situation.attributes['deviation']
    elif feature == 'trace delay':
        return situation.attributes['trace delay']
    elif feature == 'trace duration':
        return situation.attributes['time:trace-duration']
    elif feature == '@@case_id_column':
        return situation.attributes['@@case_id_column']
    elif feature == 'elapsed time':
        return situation[-1]['time:timestamp'] - situation.attributes['time:start']
    elif feature == 'next activity':
        for event in reversed(situation):
            if event['concept:name'] == activity:
                return event['next activity']
        return 'NONE'
    elif feature == 'previous activity':
        for event in reversed(situation):
            if event['concept:name'] == activity:
                return event['previous activity']
        return 'NONE'


def get_situation_features(situations: list,
                           features: list,
                           sensitive_feature: tuple,
                           target_feature: tuple):
    """Create a data frame with the desired features for the given situations.

    :param situations: a list of situations
    :param features: a list of features, i.e. tuples of strings of the form (attribute, activity)
    :param sensitive_feature: a situation feature that marks the sensitive attribute
    :param target_feature: the target feature for any classification task"""
    # determine feature names and create empty feature dictionary
    feature_names = [activity + '_' + attribute if activity is not None else attribute
                     for attribute, activity in features]
    data = {feature: [] for feature in feature_names}

    for trace in situations:
        for feature in features:
            attribute, activity = feature
            if activity is None:  # trace level situation feature
                if attribute in situation_feature_list:  # feature has to be computed
                    data[attribute].append(compute_feature(trace, attribute))
                else:  # feature is explicit in the situation
                    try:
                        data[attribute].append(trace.attributes[attribute])
                    except KeyError:
                        data[attribute].append('NONE')
            else:  # event level situation feature
                feature_name = activity + '_' + attribute
                event = None
                for e in reversed(trace):  # get last event with the activity from the situation
                    if e['concept:name'] == activity:
                        event = e
                        break
                if event is not None:
                    if attribute in situation_feature_list:
                        data[feature_name].append(compute_feature(trace, attribute, activity))
                    else:
                        try:
                            data[feature_name].append(event[attribute])
                        except KeyError:
                            data[feature_name].append('NONE')
                else:  # no matching event in the situation
                    data[feature_name].append('NONE')

    table = pd.DataFrame(data)

    # mark the columns for sensitive and target feature
    #sensitive_feature_name = sensitive_feature[0] if sensitive_feature[1] is None else sensitive_feature[0] + \
    #                                                                                   sensitive_feature[1]
    target_feature_name = target_feature[0] if target_feature[1] is None else target_feature[1] + '_' + target_feature[0]
    #table.rename(columns={target_feature_name: 'TARGET_' + target_feature_name},
    #             inplace=True)

    return table


def get_situation_feature_table(log: EventLog,
                                situation_type: str,
                                situation_activities: list,
                                features: list,
                                sensitive_feature: tuple,
                                target_feature: tuple):
    """Extract a set of situation features from an event log and return them as a data table.

    :param log: An event log
    :param situation_type: The type of situation to extract, one of "trace" and "event"
    :param situation_activities: A list of activity names, only relevant in case of event situations
    :param features: A list of features specified as (attribute, activity)-tuples, for trace level features (attribute, None)
    :param sensitive_feature: A feature that is marked as sensitive
    :param target_feature: A target feature for classification tasks"""
    # get situations
    situations = [sit for trace in log for sit in get_situations_from_trace(trace, activities=situation_activities,
                                                                            trace_situation=(
                                                                                    situation_type == 'trace'))]
    # remove duplicates from features and ensure sensitive and target feature are in features
    features = list(dict.fromkeys(features))
    #if sensitive_feature not in features:
    #    features.append(sensitive_feature)
    if target_feature not in features:
        features.append(target_feature)

    enrich_event_log(log, features)

    # create situation feature table
    table = get_situation_features(situations, features, sensitive_feature, target_feature)
    return table


def add_discrimination(table: pd.DataFrame, target: str, new: str, probabilities: tuple):
    """Add an artificial binary discriminatory feature to the data, such that the new feature is linked to another
    feature in the data.

    :param table: A table containing the original data
    :param target: The binary target feature to which the new feature should be linked
    :param new: The name of the new feature
    :param probabilities: A tuple with two numbers between 0.0 and 1.0, where the first number is the probability of the
                          new feature to be True given that the target feature is True and the second number is the
                          probability that the new feature is True given that the target feature is False"""
    # get the binary condition
    cond = table[target]
    length = len(cond)
    if cond.dtype == 'bool':
        pass
    elif cond.dtype in ['int32', 'int64', 'float32', 'float64']:
        cond = cond > 0.5 * max(cond)

    # generate new feature
    p_true, p_false = probabilities
    sensitive = np.where(cond, np.random.binomial(1, p_true, (length,)), np.random.binomial(1, p_false, (length,)))
    table[new] = np.array(sensitive, dtype=bool)

    return table
