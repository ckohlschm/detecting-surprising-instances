from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def apply_kmeans(event_log, feature_names, features, k, gamma, directory):
    event_log = vicinity_detection(event_log, feature_names, features, k)
    surprising_instances, surprising_instances_total, better_performing_instances_total, worse_performing_instances_total = surprising_instance_detection(event_log, '@@caseDuration', k, gamma, directory)

    return surprising_instances, surprising_instances_total, better_performing_instances_total, worse_performing_instances_total

def vicinity_detection(event_log, feature_names, features, k):
    event_log_filtered = event_log.filter(features)

    cols_to_normalize = feature_names
    event_log_filtered[cols_to_normalize] = MinMaxScaler().fit_transform(event_log_filtered[cols_to_normalize])

    event_log_filtered_no_case_id = event_log_filtered.filter(feature_names)
    situations = event_log_filtered_no_case_id.values.tolist()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(situations)

    k_means_labels = kmeans.labels_
    print('Found clusters: ' + str(k_means_labels))
    event_log['cluster_id'] = k_means_labels
    return event_log

def surprising_instance_detection(event_log, target_feature_name, k, gamma, directory):
    surprising_instances = {}

    surprising_instances_total = 0
    better_performing_instances_total = 0
    worse_performing_instances_total = 0
    id = 1
    data_list = {}
    for vicinity_id in range(k):
        filtered_data = event_log[event_log['cluster_id'] == vicinity_id]
        print('Filtered data: ')
        print(filtered_data.head())
        filtered_data.to_csv(str(directory) + "/kmeans/vicinity_" + str(id) + ".csv")
        better_performing_instances, worse_performing_instances = find_outliers_for_node(filtered_data, target_feature_name, gamma)
        surprising_instances_total = surprising_instances_total + len(better_performing_instances)
        surprising_instances_total = surprising_instances_total + len(worse_performing_instances)
        better_performing_instances_total = better_performing_instances_total + len(better_performing_instances)
        worse_performing_instances_total = worse_performing_instances_total + len(worse_performing_instances)
        surprising_instances[id] = (better_performing_instances, worse_performing_instances)

        case_id_col_list = filtered_data['@@case_id_column'].tolist()
        data_list[id] = case_id_col_list
        id = id + 1

    def get_cluster_for_row(row, dict):
        for key,value in dict.items():
            if row['@@case_id_column'] in value:
                return key
        return 'NOT FOUND'

    event_log['cluster_id'] = event_log.apply(lambda row: get_cluster_for_row(row, data_list), axis=1)
    print(event_log)

    event_log['days'] = event_log['Throughput Time'].dt.days

    fig = plt.figure()
    ax = fig.gca()
    event_log.boxplot(ax=ax, column=['days'], by=['cluster_id'])
    ax.set_ylabel('Case Duration (Days)')
    plt.savefig(str(directory) + '/kmeans/boxplot_case_duration_all_clusters.png')
    
    return surprising_instances, surprising_instances_total, better_performing_instances_total, worse_performing_instances_total

def calculate_surprisingness(row, gamma, avg_s, avg_v_without_s, surprisingSize, vicinitySize):
    return gamma * abs(avg_s - avg_v_without_s) + (1-gamma) * (surprisingSize / vicinitySize)

def calculate_effectiveness_better(row, avg_v_without_s, avg_s, otherSize):
    return (avg_v_without_s - avg_s) * otherSize

def calculate_effectiveness_worse(row, avg_v_without_s, avg_s, surprisingSize):
    return (avg_s - avg_v_without_s) * surprisingSize

def find_outliers_for_node(filtered_data, target_feature_name, gamma):
    print('Target Feature Name: ' + str(target_feature_name))
    Q1 = filtered_data[target_feature_name].quantile(0.25)
    Q3 = filtered_data[target_feature_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 *IQR

    filter_better = (filtered_data[target_feature_name] < lower_bound)
    event_log_filter_better = filtered_data.loc[filter_better]
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_better.index)]
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    avg_s = event_log_filter_better[target_feature_name].mean()
    surprisingSize = len(event_log_filter_better)
    vicinitySize = len(filtered_data)
    if len(event_log_filter_better) > 0:
        event_log_filter_better['Surprisingness'] = event_log_filter_better.apply(lambda row: calculate_surprisingness(row=row, gamma=gamma, avg_s=avg_s, avg_v_without_s=p_avg, surprisingSize=surprisingSize, vicinitySize=vicinitySize), axis=1)
        event_log_filter_better['Effectiveness'] = event_log_filter_better.apply(lambda row: calculate_effectiveness_better(row=row, avg_v_without_s=p_avg, avg_s=avg_s, otherSize=(len(filtered_data) - len(event_log_filter_worse))), axis=1)
    print("There are " + str(len(event_log_filter_better)) + " better performing instances")

    filter_worse = (filtered_data[target_feature_name] > upper_bound)
    event_log_filter_worse = filtered_data.loc[filter_worse] 
    other_instances_in_vicinity = filtered_data[~filtered_data.index.isin(event_log_filter_worse.index)]
    p_avg = other_instances_in_vicinity[target_feature_name].mean()
    avg_s = event_log_filter_worse[target_feature_name].mean()
    surprisingSize = len(event_log_filter_worse)
    vicinitySize = len(filtered_data)
    if len(event_log_filter_worse) > 0:
        event_log_filter_worse['Surprisingness'] = event_log_filter_worse.apply(lambda row: calculate_surprisingness(row=row, gamma=gamma, avg_s=avg_s, avg_v_without_s=p_avg, surprisingSize=surprisingSize, vicinitySize=vicinitySize), axis=1)
        event_log_filter_worse['Effectiveness'] = event_log_filter_worse.apply(lambda row: calculate_effectiveness_worse(row=row, avg_v_without_s=p_avg, avg_s=avg_s, surprisingSize=surprisingSize), axis=1)
    print("There are " + str(len(event_log_filter_worse)) + " worse performing instances")

    return event_log_filter_better, event_log_filter_worse