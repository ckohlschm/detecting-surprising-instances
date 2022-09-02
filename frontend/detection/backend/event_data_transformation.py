import pandas as pd
import datetime
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features

def extract_event_log_features(event_log, parameters):
    data, feature_names = log_to_features.apply(event_log, variant=log_to_features.Variants.TRACE_BASED, parameters = parameters)
    print(feature_names)
    # Dataframe for event log
    pd_data = pd.DataFrame(data, columns=feature_names)
    return pd_data, feature_names

def label_delay(row, lower_threshold, upper_threshold):
    if row['@@caseDuration'] > upper_threshold:
        return 'Long'
    elif row['@@caseDuration'] < lower_threshold:
        return 'Short'
    return 'Medium'

def add_delay(pd_data, lower_threshold=None, upper_threshold=None):
    pd_data_copy = pd_data.copy()
    if upper_threshold != None and lower_threshold != None:
        pd_data_copy['@@delay'] = pd_data_copy.apply (lambda row: label_delay(row=row, lower_threshold=lower_threshold, upper_threshold=upper_threshold), axis=1)
    elif upper_threshold != None:
        pd_data_copy['@@delay'] = pd_data_copy.apply (lambda row: row['@@caseDuration'] > upper_threshold, axis=1)
    else:
        iqr = pd_data_copy['@@caseDuration'].quantile(.75) - pd_data_copy['@@caseDuration'].quantile(.25)
        threshold = pd_data_copy['@@caseDuration'].quantile(.75) + 1.5 * iqr
        pd_data_copy['@@delay'] = pd_data_copy.apply (lambda row: row['@@caseDuration'] > threshold, axis=1)
    return pd_data_copy

def _make_df_binary(table: pd.DataFrame, n_bins: int, prefix_sep: str):
    """Convert the columns in a dataframe to binary values. """
    table = table.copy()
    for col in table.columns:
        if table[col].dtype == 'bool':
            continue
        # convert datetime objects to the right format
        if table[col].dtype == 'object' \
                and len(set([type(obj) for obj in table[col]])) == 1 \
                and type(table[col][0]) == datetime.datetime:
            table[col] = pd.to_datetime(table[col], utc=True)
        # Boolean conversion for binary data
        if table[col].dtype == 'int64' and len(table[col].unique()) == 2:
            table[col] = (table[col] == 1)
        # bin numerical and time data
        if table[col].dtype != 'object' and table[col].dtype != 'bool':
            table[col] = pd.cut(table[col], n_bins)

    # one hot encoding of all columns
    table = pd.get_dummies(table, prefix_sep=prefix_sep)
    # columns with the type bool need to be converted separately
    table = pd.get_dummies(table, prefix_sep=prefix_sep, columns=[col for col in table.columns
                                                                  if table[col].dtype == 'bool'])

    # remove columns with missing data or only one value
    no_data_cols = [col for col in table.columns if col.endswith("NONE")]
    one_value_cols = [col for col in table.columns if len(set(table[col])) == 1]
    table = table.drop(columns=[col for col in no_data_cols + one_value_cols if not col.startswith('TARGET_')])
    return table