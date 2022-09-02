from event_data_reader import read_event_log
from feature_extraction import extract_features, Variants
from detection import detect_surprising_instances
from root_cause_analysis import root_cause_analysis
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

directory = 'Assignment2'

intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
)

def display_time(seconds, granularity=5):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

if not os.path.exists(directory):
    os.makedirs(directory)

total_runtime_start = time.perf_counter()

read_runtime_start = time.perf_counter()
event_log = read_event_log(directory)
print(event_log)
read_runtime_end = time.perf_counter()

boxplot = event_log.boxplot(column=['@@caseDuration']) 
boxplot.plot()
boxplot = event_log.boxplot(column=['@@caseDuration']) 
boxplot.plot()
plt.savefig(str(directory) + '/boxplot_case_duration.png')

Q1 = event_log['@@caseDuration'].quantile(0.25)
Q3 = event_log['@@caseDuration'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 *IQR

filter_better = (event_log['@@caseDuration'] < lower_bound)
event_log_filter_better = event_log.loc[filter_better] 
print(event_log_filter_better.head())
print("There are " + str(len(event_log_filter_better)) + " better outliers")

filter_better = (event_log['@@caseDuration'] > upper_bound)
event_log_filter_worse = event_log.loc[filter_better] 
print(event_log_filter_worse.head())
print("There are " + str(len(event_log_filter_worse)) + " worse outliers")

target_feature = '@@caseDuration'
threshold = 0.1
variant = Variants.PEARSON
feature_extraction_start = time.perf_counter()
feature_list = extract_features(pd_data_event_log=event_log, variant=variant, target_feature=target_feature, threshold=threshold)
feature_extraction_end = time.perf_counter()
print('Found ' + str(len(feature_list)) + ' features with a correlation higher than ' + str(threshold))
print(feature_list)

model_threshold = 1000000
model_strategy = 'dt_regression'
detect_instances_start = time.perf_counter()
surprising_instances_dt = []
surprising_instances_dt = detect_surprising_instances(pd_data_event_log=event_log, descriptive_feature_names=feature_list, target_feature_name=target_feature, strategy=model_strategy, add_conditions=True, threshold=model_threshold, max_depth=5, directory=directory)
detect_instances_end = time.perf_counter()

#root_cause_analysis(event_log=event_log, surprising_instances=surprising_instances,model_strategy=model_strategy)

total_runtime_end = time.perf_counter()

f = open(str(directory) + "/report.txt", "w")
f.write(f"Read event data:                  {read_runtime_end - read_runtime_start:0.4f} seconds\n")
f.write(f"Extract features:                 {feature_extraction_end - feature_extraction_start:0.4f} seconds\n")
f.write(f"Detect surprising instances:      {detect_instances_end - detect_instances_start:0.4f} seconds\n")
f.write(f"Total runtime:                    {total_runtime_end - total_runtime_start:0.4f} seconds\n")
f.write(f"Event log size:                   {len(event_log)} cases\n\n")

f.write(f"=" + 20*"=" + "Boxplots" + 20*"=" + "=\n")
f.write(f"IQR: {IQR} - lb: {display_time(lower_bound)} - ub: {display_time(upper_bound)} - Q1: {display_time(Q1)} - Q3: {display_time(Q3)}\n")
f.write(f"Better performing instances: {len(event_log_filter_better)}\n")
f.write(f"{event_log_filter_better.head()}\n")
f.write(f"Worse performing instances: {len(event_log_filter_worse)}\n")
f.write(f"{event_log_filter_worse.head()}\n\n")

f.write(f"=" + 20*"=" + "Decision Tree" + 20*"=" + "=\n")
for node_id, value in surprising_instances_dt.items():
    node, better_performing_instances, worse_performing_instances, surprising_instnaces_better, surprising_instances_worse = value
    f.write(f"Node: {node}\n")
    f.write(f"Better performing instances: {len(better_performing_instances)}\n")
    if 'surprisingnessBetterIndex' in better_performing_instances:
        f.write(f"Average surprisingness measure: {better_performing_instances['surprisingnessBetterIndex'].mean()}\n")
        f.write(f"Average relevance index: {better_performing_instances['RelevanceIndex'].mean()}\n")
    f.write(f"Worse performing instances: {len(worse_performing_instances)}\n")
    if 'surprisingnessWorseIndex' in worse_performing_instances:
        f.write(f"Average surprisingness measure: {worse_performing_instances['surprisingnessWorseIndex'].mean()}\n")
        f.write(f"Average relevance index: {worse_performing_instances['RelevanceIndex'].mean()}\n")

f.write(f"=" + 20*"=" + "Similarity Graph" + 20*"=" + "=\n")

f.close()
