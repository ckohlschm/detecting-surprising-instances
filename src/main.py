from event_data_reader import read_event_log
from boxplot_method import apply_boxplots
from classification_method import supervised_approach
from similarity_graph_method import apply_similarity_graph
from clustering_method import apply_kmeans
import matplotlib.pyplot as plt
import os
import datetime

directory = 'BPI17'
boxplots = True
decision_tree = True
similarity_graph = True
kmeans = True
target_feature = '@@caseDuration'
gamma = 0.01

feature_names = ['case:LoanGoal@Boat', 'case:LoanGoal@Business goal', 'case:LoanGoal@Car', 'case:LoanGoal@Caravan / Camper', 'case:LoanGoal@Debt restructuring', 'case:LoanGoal@Existing loan takeover', 'case:LoanGoal@Extra spending limit', 'case:LoanGoal@Home improvement', 'case:LoanGoal@Motorcycle', 'case:LoanGoal@Not speficied', 'case:LoanGoal@Other, see explanation', 'case:LoanGoal@Remaining debt home', 'case:LoanGoal@Tax payments', 'case:LoanGoal@Unknown', 'case:ApplicationType@New credit', 'case:RequestedAmount', '@@num_occurrence_O_Create Offer'] # ['case:LoanGoal@Boat', 'case:LoanGoal@Business goal', 'case:LoanGoal@Car', 'case:LoanGoal@Caravan / Camper', 'case:LoanGoal@Debt restructuring', 'case:LoanGoal@Existing loan takeover', 'case:LoanGoal@Extra spending limit', 'case:LoanGoal@Home improvement', 'case:LoanGoal@Motorcycle', 'case:LoanGoal@Not speficied', 'case:LoanGoal@Other, see explanation', 'case:LoanGoal@Remaining debt home', 'case:LoanGoal@Tax payments', 'case:LoanGoal@Unknown', 'case:ApplicationType@Limit raise', 'case:ApplicationType@New credit', 'case:RequestedAmount', '@@num_occurrence_O_Create Offer']
features = ['@@case_id_column', 'case:LoanGoal@Boat', 'case:LoanGoal@Business goal', 'case:LoanGoal@Car', 'case:LoanGoal@Caravan / Camper', 'case:LoanGoal@Debt restructuring', 'case:LoanGoal@Existing loan takeover', 'case:LoanGoal@Extra spending limit', 'case:LoanGoal@Home improvement', 'case:LoanGoal@Motorcycle', 'case:LoanGoal@Not speficied', 'case:LoanGoal@Other, see explanation', 'case:LoanGoal@Remaining debt home', 'case:LoanGoal@Tax payments', 'case:LoanGoal@Unknown', 'case:ApplicationType@New credit', 'case:RequestedAmount', '@@num_occurrence_O_Create Offer'] # ['case:LoanGoal@Boat', 'case:LoanGoal@Business goal', 'case:LoanGoal@Car', 'case:LoanGoal@Caravan / Camper', 'case:LoanGoal@Debt restructuring', 'case:LoanGoal@Existing loan takeover', 'case:LoanGoal@Extra spending limit', 'case:LoanGoal@Home improvement', 'case:LoanGoal@Motorcycle', 'case:LoanGoal@Not speficied', 'case:LoanGoal@Other, see explanation', 'case:LoanGoal@Remaining debt home', 'case:LoanGoal@Tax payments', 'case:LoanGoal@Unknown', 'case:ApplicationType@Limit raise', 'case:ApplicationType@New credit', 'case:RequestedAmount', '@@num_occurrence_O_Create Offer']

def display_time(seconds, granularity=5):
    result = []
    intervals = (('weeks', 604800), ('days', 86400), ('hours', 3600), ('minutes', 60), ('seconds', 1))

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

def convert_to_datetime(row):
    return datetime.timedelta(seconds=round(row['@@caseDuration'], 0))

event_log, pm4py_event_log = read_event_log(directory)

if not os.path.exists(str(directory)):
    os.makedirs(str(directory))

# add time information
event_log['Throughput Time'] = event_log.apply(lambda row: convert_to_datetime(row), axis=1)
event_log.to_csv(str(directory) + "/event_log_modified.csv")

if boxplots:
    fig = plt.figure()
    ax = fig.gca()
    event_log['Throughput Time'].dt.days.plot.box(ax=ax)
    ax.set_ylabel('Case Duration (Days)')
    plt.savefig(str(directory) + '/boxplot.png')
    event_log_better, event_log_worse, upper_bound, lower_bound, IQR, Q1, Q3 = apply_boxplots(event_log, target_feature)

if decision_tree:
    if not os.path.exists(str(directory) + "/decisiontree"):
        os.makedirs(str(directory) + "/decisiontree")
    surprising_instances_dt, surprising_instances_total_dt, better_performing_instances_total_dt, worse_performing_instances_total_dt = supervised_approach(event_log, feature_names, target_feature, gamma, directory)

if similarity_graph:
    if not os.path.exists(str(directory) + "/similaritygraph"):
        os.makedirs(str(directory) + "/similaritygraph")
    
    surprising_instances_sg, surprising_instances_total_sg, better_performing_instances_total_sg, worse_performing_instances_total_sg = apply_similarity_graph(event_log, feature_names, features, target_feature, gamma, directory)

if kmeans:
    if not os.path.exists(str(directory) + "/kmeans"):
        os.makedirs(str(directory) + "/kmeans")
    k = 25
    surprising_instances_km, surprising_instances_total_km, better_performing_instances_total_km, worse_performing_instances_total_km = apply_kmeans(event_log, feature_names, features, k, gamma, directory)

f = open(str(directory) + "/report.txt", "w")
f.write(f"Number of situations:                   {len(event_log)} cases\n")
f.write(f"Features in event log:            {len(event_log.columns)}\n\n")

if boxplots:
    f.write(f"=" + 20*"=" + "Boxplots" + 20*"=" + "=\n")
    f.write(f"Found a total of {len(event_log_better) + len(event_log_worse)} surprising instances\n")
    f.write(f"IQR: {IQR} - lb: {display_time(lower_bound)} - ub: {display_time(upper_bound)} - Q1: {display_time(Q1)} - Q3: {display_time(Q3)}\n")
    f.write(f"Better performing instances: {len(event_log_better)}\n")
    f.write(f"Worse performing instances: {len(event_log_worse)}\n")

if decision_tree:
    f.write(f"=" + 20*"=" + "Decision Tree" + 20*"=" + "=\n")
    f.write(f"Found a total of {surprising_instances_total_dt} surprising instances\n")
    f.write(f"Better performing instances: {better_performing_instances_total_dt}\n")
    f.write(f"Worse performing instances: {worse_performing_instances_total_dt}\n")
    for node_id, value in surprising_instances_dt.items():
        node, better_performing_instances, worse_performing_instances, vicinity = value
        f.write(f"Node: {node}\n")
        f.write(f"Better performing instances: {len(better_performing_instances)}\n")
        if len(better_performing_instances) > 0:
            f.write(f"Effectiveness: {better_performing_instances['Effectiveness'].mean()}\n")
            f.write(f"Surprisingness: {better_performing_instances['Surprisingness'].mean()}\n")
            better_performing_instances.to_csv(str(directory) + "/decisiontree/better_performing_instances_" + str(node_id) + ".csv")
        f.write(f"Worse performing instances: {len(worse_performing_instances)}\n")
        if len(worse_performing_instances) > 0:
            f.write(f"Effectiveness: {worse_performing_instances['Effectiveness'].mean()}\n")
            f.write(f"Surprisingness: {worse_performing_instances['Surprisingness'].mean()}\n")
            worse_performing_instances.to_csv(str(directory) + "/decisiontree/worse_performing_instances_" + str(node_id) + ".csv")

if kmeans:
    f.write(f"=" + 20*"=" + "K-Means" + 20*"=" + "=\n")
    f.write(f"Found a total of {surprising_instances_total_km} surprising instances\n")
    f.write(f"Better performing instances:      {better_performing_instances_total_km}\n")
    f.write(f"Worse performing instances:       {worse_performing_instances_total_km}\n")
    for vicinity_id, value in surprising_instances_km.items():
        better_performing_instances, worse_performing_instances = value
        f.write(f"ID: {vicinity_id}\n")
        f.write(f"Better performing instances: {len(better_performing_instances)}\n")
        if len(better_performing_instances) > 0:
            f.write(f"Effectiveness: {better_performing_instances['Effectiveness'].mean()}\n")
            f.write(f"Surprisingness: {better_performing_instances['Surprisingness'].mean()}\n")
            better_performing_instances.to_csv(str(directory) + "/kmeans/better_performing_instances_" + str(vicinity_id) + ".csv")
        if len(worse_performing_instances) > 0:
            f.write(f"Effectiveness: {worse_performing_instances['Effectiveness'].mean()}\n")
            f.write(f"Surprisingness: {worse_performing_instances['Surprisingness'].mean()}\n")
            worse_performing_instances.to_csv(str(directory) + "/kmeans/worse_performing_instances_" + str(vicinity_id) + ".csv")

if similarity_graph:
    f.write(f"=" + 20*"=" + "Similarity Graph" + 20*"=" + "=\n")
    f.write(f"Found a total of {surprising_instances_total_sg} surprising instances\n")
    f.write(f"Better performing instances:      {better_performing_instances_total_sg}\n")
    f.write(f"Worse performing instances:       {worse_performing_instances_total_sg}\n")
    for vicinity_id, value in surprising_instances_sg.items():
        better_performing_instances, worse_performing_instances = value
        f.write(f"ID: {vicinity_id}\n")
        f.write(f"Better performing instances: {len(better_performing_instances)}\n")
        if len(better_performing_instances) > 0:
            f.write(f"Effectiveness: {better_performing_instances['Effectiveness'].mean()}\n")
            f.write(f"Surprisingness: {better_performing_instances['Surprisingness'].mean()}\n")
            better_performing_instances.to_csv(str(directory) + "/similaritygraph/better_performing_instances_" + str(vicinity_id) + ".csv")
        f.write(f"Worse performing instances: {len(worse_performing_instances)}\n")
        if len(worse_performing_instances) > 0:
            f.write(f"Effectiveness: {worse_performing_instances['Effectiveness'].mean()}\n")
            f.write(f"Surprisingness: {worse_performing_instances['Surprisingness'].mean()}\n")
            worse_performing_instances.to_csv(str(directory) + "/similaritygraph/worse_performing_instances_" + str(vicinity_id) + ".csv")

f.close()
