from pm4py.visualization.decisiontree import visualizer as dectree_visualizer
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.Endpoint import Endpoint
from sklearn import tree
import numpy as np
import pandas as pd
import json
import dowhy
from dowhy import CausalModel
from cdt.causality.graph import LiNGAM, PC, GES
from rpy2.robjects import r as R
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import preprocessing
import graphviz
import networkx as nx 


def train_dt_classifier(pd_data, descriptive_feature_names, target_feature_name, maxDepth):
    # Filter descriptive data
    descriptive_attributes = pd_data.filter(items=descriptive_feature_names)
    descriptive_feature_names = descriptive_attributes.columns.tolist()
    descriptive_feature_data = descriptive_attributes.values.tolist()

    # Filter target data
    target_attributes = pd_data.filter(items=target_feature_name)
    target_feature_name = target_attributes.columns.tolist()
    target_feature_data = target_attributes.values.tolist()

    np_target = np.array(target_feature_data)
    unique, counts = np.unique(np_target, return_counts=True)
    print('Target Classes: ' + str(dict(zip(unique, counts))) + '|MaxDepth: ' + str(maxDepth))

    # Create DT Classifier
    #clf = RandomForestClassifier(max_depth=7)
    clf = tree.DecisionTreeClassifier(max_depth=maxDepth, random_state=11)
    
    clf.fit(descriptive_feature_data, target_feature_data)
    return clf, unique

def generate_dt_for_query_string(query_string, event_log, new_descriptive_feature_names, target_feature_name, outputfile_name):
    print("Query data for query string " + str(query_string))
    pd_data_by_conditions = event_log.query(query_string)
    print(pd_data_by_conditions.head())

    clf_new, new_unique = train_dt_classifier(pd_data=pd_data_by_conditions, descriptive_feature_names=new_descriptive_feature_names, target_feature_name=target_feature_name, maxDepth=5)

    # Visualize DT
    new_classes = []
    for item in new_unique:
        new_classes.append(str(item))

    # graphviz does not like special characters
    new_feature_names = [] 
    for feature_name in new_descriptive_feature_names:
        feature_name = feature_name.replace('<', 'l')
        feature_name = feature_name.replace('>', 'g')
        new_feature_names.append(feature_name)

    gviz = dectree_visualizer.apply(clf_new, new_feature_names, new_classes)
    decision_tree_output_path = 'detection/static/detection/figures/' + str(outputfile_name) + '.png'
    dectree_visualizer.save(gviz, decision_tree_output_path)
    return decision_tree_output_path

def find_cause_for_instance(event_log, instance_to_investigate, target_feature_name, output_file_name):
    query_string = ""
    for condition in instance_to_investigate.conditions:
        query_string += "`" + str(condition.attribute_name) + "`"
        if condition.greater:
            query_string += " > "
        else:
            query_string += " <= "
        query_string += str(condition.threshold)
        query_string += " & "

    #query_string_better = query_string + "`" + str(instance_to_investigate.target_attribute) + "`" + " < " + str(instance_to_investigate.target_data)

    #query_string_worse = query_string + "`" + str(instance_to_investigate.target_attribute) + "`" + " > " + str(instance_to_investigate.target_data)

    #print("Query String: " + query_string)

    # Instances with same attributes
    #query_string_for_same_instances = query_string
    #query_string_for_same_instances += "`" + str(instance_to_investigate.target_attribute) + "`"
    #query_string_for_same_instances += " == "
    #query_string_for_same_instances += str(instance_to_investigate.target_data)

    #df_with_same_attributes = pd_data.query(query_string_for_same_instances)
    #print(df_with_same_attributes.head())
    #print(df_with_same_attributes.info())

    # Instances with different target attribute
    #query_string_for_different_instances = query_string
    #query_string_for_different_instances += "`" + str(instance_to_investigate.target_attribute) + "`"
    #query_string_for_different_instances += " != "
    #query_string_for_different_instances += str(instance_to_investigate.target_data)

    #df_with_different_attributes = pd_data.query(query_string_for_different_instances)
    #print(df_with_different_attributes.head())
    #print(df_with_different_attributes.info())

    new_descriptive_feature_names = event_log.columns.tolist()
    
    if '@@caseDuration' in new_descriptive_feature_names:
        new_descriptive_feature_names.remove('@@caseDuration')
    if '@@delay' in new_descriptive_feature_names:
        new_descriptive_feature_names.remove('@@delay')
    if '@@case_id_column' in new_descriptive_feature_names:
        new_descriptive_feature_names.remove('@@case_id_column')
    if '@@surprising' in new_descriptive_feature_names:
        new_descriptive_feature_names.remove('@@surprising')
    if '@@surprising_better' in new_descriptive_feature_names:
        new_descriptive_feature_names.remove('@@surprising_better')
    if '@@surprising_worse' in new_descriptive_feature_names:
        new_descriptive_feature_names.remove('@@surprising_worse')
    
    print('New features: ' + str(new_descriptive_feature_names))
    #for condition in instance_to_investigate.conditions:
    #    if condition.attribute_name in new_descriptive_feature_names:
    #        new_descriptive_feature_names.remove(condition.attribute_name)

    query_string = query_string.rsplit('&', 1)[0]
    print("New query string: " + query_string)
    pd_data_by_conditions = event_log.query(query_string)
    print(pd_data_by_conditions.head())

    #print("New query string better: " + query_string_better)
    #pd_data_by_conditions_better = event_log.query(query_string_better)
    #print(pd_data_by_conditions_better.head())

    #print("New query string worse: " + query_string_worse)
    #pd_data_by_conditions_worse = event_log.query(query_string_worse)
    #print(pd_data_by_conditions_worse.head())


    clf_new, new_unique = train_dt_classifier(pd_data=pd_data_by_conditions, descriptive_feature_names=new_descriptive_feature_names, target_feature_name=target_feature_name, maxDepth=5)

    # Visualize DT
    new_classes = []
    for item in new_unique:
        new_classes.append(str(item))

    # graphviz does not like special characters
    new_feature_names = [] 
    for feature_name in new_descriptive_feature_names:
        feature_name = feature_name.replace('<', 'l')
        feature_name = feature_name.replace('>', 'g')
        new_feature_names.append(feature_name)

    gviz = dectree_visualizer.apply(clf_new, new_feature_names, new_classes)
    decision_tree_output_path = 'detection/static/detection/figures/' + str(output_file_name) + '.png'
    dectree_visualizer.save(gviz, decision_tree_output_path)
    #dt_file_name_better = generate_dt_for_query_string(query_string_better, event_log, new_descriptive_feature_names, target_feature_name, 'dt_rca_better')
    #dt_file_name_worse = generate_dt_for_query_string(query_string_worse, event_log, new_descriptive_feature_names, target_feature_name, 'dt_rca_worse')

def root_cause_analysis(event_log, surprising_instances, model_strategy):
    if 'regression' in model_strategy:
        surprising_label = '@@surprising'
        def isSurprising(row, surprising_instances):
            for instance in surprising_instances:
                if row['@@case_id_column'] == instance.id:
                    return True
            return False

        event_log[surprising_label] = event_log.apply(lambda row: isSurprising(row, surprising_instances), axis=1)

    # Root cause analysis
    for instance in surprising_instances:
        target_surprising_name = [surprising_label]
        find_cause_for_instance(event_log, instance, target_surprising_name)

def apply_root_cause_analysis_old(request, context):
    data_in_vicinity_json = request.session['instances_in_vicinity']
    data_in_vicinity = pd.read_json(data_in_vicinity_json, orient='split')
    
    print(data_in_vicinity.head())
    print(data_in_vicinity.columns.to_list())
    data_in_vicinity = data_in_vicinity.drop(['@@case_id_column', 'case:ApplicationType@Limit raise', 'event:EventOrigin@Application', 'event:EventOrigin@Offer', 'event:EventOrigin@Workflow', 'event:Action@Created', 'event:Action@Deleted', 'event:Action@Obtained', 'event:Action@Released', 'event:Action@statechange', 'event:concept:name@A_Accepted', 'event:concept:name@A_Cancelled', 'event:concept:name@A_Complete', 'event:concept:name@A_Concept', 'event:concept:name@A_Create Application', 'event:concept:name@A_Denied', 'event:concept:name@A_Incomplete', 'event:concept:name@A_Pending', 'event:concept:name@A_Submitted', 'event:concept:name@A_Validating', 'event:concept:name@O_Accepted', 'event:concept:name@O_Cancelled', 'event:concept:name@O_Created', 'event:concept:name@O_Refused', 'event:concept:name@O_Returned', 'event:concept:name@O_Sent (mail and online)', 'event:concept:name@O_Sent (online only)', 'event:concept:name@W_Assess potential fraud', 'event:concept:name@W_Call after offers', 'event:concept:name@W_Call incomplete files', 'event:concept:name@W_Complete application', 'event:concept:name@W_Handle leads', 'event:concept:name@W_Personal Loan collection', 'event:concept:name@W_Shortened completion ', 'event:concept:name@W_Validate application', 'event:NumberOfTerms', 'event:OfferedAmount',  '@@max_concurrent_activities_general', '@@num_occurrence_O_Cancelled', '@@num_occurrence_A_Pending', '@@num_occurrence_W_Handle leads', '@@num_occurrence_O_Returned', '@@num_occurrence_A_Cancelled', '@@num_occurrence_A_Validating', '@@num_occurrence_A_Incomplete', '@@num_occurrence_A_Accepted', '@@num_occurrence_A_Concept', '@@num_occurrence_W_Personal Loan collection', '@@num_occurrence_A_Complete', '@@num_occurrence_W_Call after offers', '@@num_occurrence_A_Create Application', '@@num_occurrence_A_Submitted', '@@num_occurrence_W_Shortened completion ', '@@num_occurrence_O_Created', '@@num_occurrence_O_Accepted', '@@num_occurrence_O_Refused', '@@num_occurrence_W_Complete application', '@@num_occurrence_O_Sent (mail and online)', '@@num_occurrence_W_Assess potential fraud', '@@num_occurrence_W_Validate application', '@@num_occurrence_A_Denied', '@@num_occurrence_W_Call incomplete files', '@@num_occurrence_O_Sent (online only)'], axis=1)

# 'trace:creator@Fluxicon Disco', 'trace:variant-index', 'event:dismissal@#', 'event:dismissal@3', 'event:dismissal@5', 'event:dismissal@B', 'event:dismissal@C', 'event:dismissal@D', 'event:dismissal@G', 'event:dismissal@I', 'event:dismissal@K', 'event:dismissal@N', 'event:dismissal@NIL', 'event:concept:name@Add penalty', 'event:concept:name@Appeal to Judge', 'event:concept:name@Create Fine', 'event:concept:name@Insert Date Appeal to Prefecture', 'event:concept:name@Insert Fine Notification', 'event:concept:name@Notify Result Appeal to Offender', 'event:concept:name@Payment', 'event:concept:name@Receive Result Appeal from Prefecture', 'event:concept:name@Send Appeal to Prefecture', 'event:concept:name@Send Fine', 'event:concept:name@Send for Credit Collection'], axis=1) # , 'event:concept:name@Confirmation of receipt', 'event:concept:name@T02 Check confirmation of receipt', 'event:concept:name@T03 Adjust confirmation of receipt', 'event:concept:name@T04 Determine confirmation of receipt', 'event:concept:name@T05 Print and send confirmation of receipt', 'event:concept:name@T06 Determine necessity of stop advice', 'event:concept:name@T07-1 Draft intern advice aspect 1', 'event:concept:name@T07-2 Draft intern advice aspect 2', 'event:concept:name@T07-3 Draft intern advice hold for aspect 3', 'event:concept:name@T07-4 Draft internal advice to hold for type 4', 'event:concept:name@T07-5 Draft intern advice aspect 5', 'event:concept:name@T08 Draft and send request for advice', 'event:concept:name@T09-1 Process or receive external advice from party 1', 'event:concept:name@T09-2 Process or receive external advice from party 2', 'event:concept:name@T09-3 Process or receive external advice from party 3', 'event:concept:name@T09-4 Process or receive external advice from party 4', 'event:concept:name@T10 Determine necessity to stop indication', 'event:concept:name@T11 Create document X request unlicensed', 'event:concept:name@T12 Check document X request unlicensed', 'event:concept:name@T13 Adjust document X request unlicensed', 'event:concept:name@T14 Determine document X request unlicensed', 'event:concept:name@T15 Print document X request unlicensed', 'event:concept:name@T16 Report reasons to hold request', 'event:concept:name@T17 Check report Y to stop indication', 'event:concept:name@T18 Adjust report Y to stop indicition', 'event:concept:name@T19 Determine report Y to stop indication', 'event:concept:name@T20 Print report Y to stop indication', '@@caseDuration'], axis=1)
    print(data_in_vicinity.columns.to_list())
    try:
        G, edges = fci(data_in_vicinity, depth=5)
        graph_nodes = G.get_nodes()
        graph_edges = G.get_graph_edges()

        nodes = []
        node_name_dict = {}
        for i in range(len(graph_nodes)):
            node_name_dict[graph_nodes[i].get_name()] = data_in_vicinity.columns.to_list()[i]
            nodes.append(data_in_vicinity.columns.to_list()[i].replace('event:', '').replace('trace:', ''))

        print(node_name_dict)
        context['nodes'] = nodes
        edges = []
        for edge in graph_edges:
            endpoint1 = edge.get_endpoint1()
            endpoint2 = edge.get_endpoint2()

            if endpoint1 is Endpoint.TAIL:
                left = "F"
            else:
                if endpoint1 is Endpoint.ARROW:
                    left = "T"
                else:
                    left = "F"

            if endpoint2 is Endpoint.TAIL:
                right = "F"
            else:
                if endpoint2 is Endpoint.ARROW:
                    right = "T"
                else:
                    right = "F"

            edge_list = []
            edge_list.append(node_name_dict[edge.get_node1().get_name()].replace('event:', '').replace('trace:', ''))
            edge_list.append(node_name_dict[edge.get_node2().get_name()].replace('event:', '').replace('trace:', ''))
            edge_list.append(left)
            edge_list.append(right)
            edges.append(edge_list)
        context['edges'] = edges

        for edge in graph_edges:
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            endpoint1 = edge.get_endpoint1()
            endpoint2 = edge.get_endpoint2()

        print(G)
        print(edges)
    except:
        print('Singular Matrix')
        context['Exception'] = 'Singular Matrix'

    context['data_in_vicinity'] = data_in_vicinity
    return context

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph

def apply_root_cause_analysis(request, context):
    if 'submit_causal_structure' in request.POST:
        return estimate_treatment_effect(request, context)
    data_in_vicinity_json = request.session['instances_in_vicinity']
    data_in_vicinity = pd.read_json(data_in_vicinity_json, orient='split')
    target_attribute = request.session['target_attribute']
    
    print(data_in_vicinity.head())
    print(data_in_vicinity.columns.to_list())
    selected_feature_names = request.session['selected_feature_names']
    print(selected_feature_names)
    selected_feature_names.append(request.session['target_attribute'])
    #features_to_delete = ['@@case_id_column', 'case:ApplicationType@Limit raise', 'event:EventOrigin@Application', 'event:EventOrigin@Offer', 'event:EventOrigin@Workflow', 'event:Action@Created', 'event:Action@Deleted', 'event:Action@Obtained', 'event:Action@Released', 'event:Action@statechange', 'event:concept:name@A_Accepted', 'event:concept:name@A_Cancelled', 'event:concept:name@A_Complete', 'event:concept:name@A_Concept', 'event:concept:name@A_Create Application', 'event:concept:name@A_Denied', 'event:concept:name@A_Incomplete', 'event:concept:name@A_Pending', 'event:concept:name@A_Submitted', 'event:concept:name@A_Validating', 'event:concept:name@O_Accepted', 'event:concept:name@O_Cancelled', 'event:concept:name@O_Created', 'event:concept:name@O_Refused', 'event:concept:name@O_Returned', 'event:concept:name@O_Sent (mail and online)', 'event:concept:name@O_Sent (online only)', 'event:concept:name@W_Assess potential fraud', 'event:concept:name@W_Call after offers', 'event:concept:name@W_Call incomplete files', 'event:concept:name@W_Complete application', 'event:concept:name@W_Handle leads', 'event:concept:name@W_Personal Loan collection', 'event:concept:name@W_Shortened completion ', 'event:concept:name@W_Validate application',  '@@max_concurrent_activities_general', '@@num_occurrence_O_Cancelled', '@@num_occurrence_A_Pending', '@@num_occurrence_W_Handle leads', '@@num_occurrence_O_Returned', '@@num_occurrence_A_Cancelled', '@@num_occurrence_A_Validating', '@@num_occurrence_A_Incomplete', '@@num_occurrence_A_Accepted', '@@num_occurrence_A_Concept', '@@num_occurrence_W_Personal Loan collection', '@@num_occurrence_A_Complete', '@@num_occurrence_W_Call after offers', '@@num_occurrence_A_Create Application', '@@num_occurrence_A_Submitted', '@@num_occurrence_W_Shortened completion ', '@@num_occurrence_O_Created', '@@num_occurrence_O_Accepted', '@@num_occurrence_O_Refused', '@@num_occurrence_W_Complete application', '@@num_occurrence_O_Sent (mail and online)', '@@num_occurrence_W_Assess potential fraud', '@@num_occurrence_W_Validate application', '@@num_occurrence_A_Denied', '@@num_occurrence_W_Call incomplete files', '@@num_occurrence_O_Sent (online only)', '@@case_id_column', 'case:ApplicationType@Limit raise', 'event:EventOrigin@Application', 'event:EventOrigin@Offer', 'event:EventOrigin@Workflow', 'event:Action@Created', 'event:Action@Deleted', 'event:Action@Obtained', 'event:Action@Released', 'event:Action@statechange', 'event:concept:name@A_Accepted', 'event:concept:name@A_Cancelled', 'event:concept:name@A_Complete', 'event:concept:name@A_Concept', 'event:concept:name@A_Create Application', 'event:concept:name@A_Denied', 'event:concept:name@A_Incomplete', 'event:concept:name@A_Pending', 'event:concept:name@A_Submitted', 'event:concept:name@A_Validating', 'event:concept:name@O_Accepted', 'event:concept:name@O_Cancelled', 'event:concept:name@O_Created', 'event:concept:name@O_Refused', 'event:concept:name@O_Returned', 'event:concept:name@O_Sent (mail and online)', 'event:concept:name@O_Sent (online only)', 'event:concept:name@W_Assess potential fraud', 'event:concept:name@W_Call after offers', 'event:concept:name@W_Call incomplete files', 'event:concept:name@W_Complete application', 'event:concept:name@W_Handle leads', 'event:concept:name@W_Personal Loan collection', 'event:concept:name@W_Shortened completion ', 'event:concept:name@W_Validate application',  '@@max_concurrent_activities_general', '@@num_occurrence_O_Cancelled', '@@num_occurrence_A_Pending', '@@num_occurrence_W_Handle leads', '@@num_occurrence_O_Returned', '@@num_occurrence_A_Cancelled', '@@num_occurrence_A_Validating', '@@num_occurrence_A_Incomplete', '@@num_occurrence_A_Accepted', '@@num_occurrence_A_Concept', '@@num_occurrence_W_Personal Loan collection', '@@num_occurrence_A_Complete', '@@num_occurrence_W_Call after offers', '@@num_occurrence_A_Create Application', '@@num_occurrence_A_Submitted', '@@num_occurrence_W_Shortened completion ', '@@num_occurrence_O_Created', '@@num_occurrence_O_Accepted', '@@num_occurrence_O_Refused', '@@num_occurrence_W_Complete application', '@@num_occurrence_O_Sent (mail and online)', '@@num_occurrence_W_Assess potential fraud', '@@num_occurrence_W_Validate application', '@@num_occurrence_A_Denied', '@@num_occurrence_W_Call incomplete files', '@@num_occurrence_O_Sent (online only)']
    #for item in features_to_delete:
    #    if item in data_in_vicinity.columns.to_list():
    #        data_in_vicinity = data_in_vicinity.drop(item, axis=1) 

    data_in_vicinity = data_in_vicinity.filter(selected_feature_names)
    print('Unique values: ' + str(data_in_vicinity.nunique()))
    unique_values = data_in_vicinity.nunique()
    for key,value in unique_values.items():
        if value == 1:
            data_in_vicinity = data_in_vicinity.drop(key, axis=1)
            print('Dropping attribute ' + str(key) + ' with only one value')

    print(data_in_vicinity.head())
    #print('Unique values: ' + str(data_in_vicinity.nunique()))
    #x = data_in_vicinity.values
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(x)
    #data_in_vicinity = pd.DataFrame(x_scaled, columns=data_in_vicinity.columns.tolist())
    print(data_in_vicinity.columns.to_list())

    graphs = {}
    nxgraph = None
    labels = [f'{col}' for i, col in enumerate(data_in_vicinity.columns)]
    new_labels = []
    for label in labels:
        label = label.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
        new_labels.append(label)
    data_in_vicinity.columns = new_labels
    functions = {
#        'LiNGAM' : LiNGAM,
#        'PC' : PC,
        'GES' : GES,
    }

    for method, lib in functions.items():
        obj = lib()
        output = obj.predict(data_in_vicinity)
        nxgraph = output
        adj_matrix = nx.to_numpy_matrix(output)
        adj_matrix = np.asarray(adj_matrix)
        graph_dot = make_graph(adj_matrix, new_labels)
        graphs[method] = graph_dot

    # Visualize graphs
    for method, graph in graphs.items():
        print("Method : %s"%(method))
        print("Graph : %s"%(graph))

    for method, graph in graphs.items():
        #if method != "LiNGAM":
        #    continue
        print('\n*****************************************************************************\n')
        print("Causal Discovery Method : %s"%(method))
        
        # Obtain valid dot format
        #graph.edge('O_Create Offer_MonthlyCost', 'TARGET_O_Create Offer_Selected', label='0.4')
        print(graph.source)
        graph_dot = str_to_dot(graph.source)
        print(graph_dot)
        causal_estimates = []
        def estimate_effect(treatment, outcome):
            try:
                # Define Causal Model
                model=CausalModel(
                        data = data_in_vicinity,
                        treatment=treatment,
                        outcome=outcome,
                        graph=graph_dot)

                # Identification
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                print(identified_estimand)
                
                # Estimation
                estimate = model.estimate_effect(identified_estimand,
                                                method_name="backdoor.linear_regression",
                                                control_value=0,
                                                treatment_value=1,
                                                confidence_intervals=True,
                                                test_significance=True)
                print("Causal Estimate " + str() + " is " + str(estimate.value))

                context['estimate'] = estimate.value
                return estimate.value
            except nx.exception.NetworkXError as e:
                context['Exception'] = 'Could not identify causal effects: "' + str(e) + '"'
                return None
            

        for edge in nxgraph.edges:
            node_1, node_2 = edge
            treatment = node_2.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '').replace(':', '')
            outcome = node_1.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '').replace(':', '')
            effect = estimate_effect(treatment, outcome)
            if effect:
                effect_estimate = round(effect, 3)
                causal_estimates.append((node_1, node_2, effect_estimate))
            outcome = node_2.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
            treatment = node_1.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
            effect = estimate_effect(treatment, outcome)
            if effect:
                effect_estimate = round(effect, 3)
                causal_estimates.append((node_1, node_2, effect_estimate))
            

        print('Causal estimates: ' + str(causal_estimates))
        
        if nxgraph:
            graph_nodes = nxgraph.nodes
            graph_edges = nxgraph.edges

            nodes = []
            for i in range(len(graph_nodes)):
                nodes.append(data_in_vicinity.columns.to_list()[i])

            print(nodes)
            context['nodes'] = nodes
            edges = []
            print(graph_edges)
            for edge in graph_edges:
                node_1, node_2 = edge
                if [node_2, node_1, "T", "F"] in edges:
                    print('Already in list')
                    edges.remove([node_2, node_1, "T", "F"])
                    edge_list = []
                    edge_list.append(node_1)
                    edge_list.append(node_2)
                    edge_list.append("T")
                    edge_list.append("T")
                else:
                    edge_list = []
                    edge_list.append(node_1)
                    edge_list.append(node_2)
                    edge_list.append("T")
                    edge_list.append("F")
                edges.append(edge_list)
            context['edges'] = edges

    context['data_in_vicinity'] = data_in_vicinity
    context['target_attribute'] = target_attribute
    context['causal_estimates'] = causal_estimates
    return context

def estimate_treatment_effect(request, context):
    print('Estimating new Effect!')
    edges = []
    nodes = []
    edge_list = None
    node_list = None
    if request.POST['GraphLinks']:
        edge_list = json.loads(request.POST['GraphLinks'])

    if request.POST['GraphNodes']:
        node_list = json.loads(request.POST['GraphNodes'])

    for node in node_list:
        nodes.append(node['id'])

    for edge in edge_list:
        edge_llist = []
        edge_llist.append(edge['source']['id'])
        edge_llist.append(edge['target']['id'])
        if edge['left']:
            edge_llist.append("T")
        else:
            edge_llist.append("F")
        if edge['right']:
            edge_llist.append("T")
        else:
            edge_llist.append("F")
        edges.append(edge_llist)

    data_in_vicinity_json = request.session['instances_in_vicinity']
    data_in_vicinity = pd.read_json(data_in_vicinity_json, orient='split')
    target_attribute = request.session['target_attribute']
    
    selected_feature_names = request.session['selected_feature_names']
    print(selected_feature_names)
    selected_feature_names.append(request.session['target_attribute'])
    #print(data_in_vicinity.head())
    #print(data_in_vicinity.columns.to_list())
    #features_to_delete = ['@@case_id_column', 'case:ApplicationType@Limit raise', 'event:EventOrigin@Application', 'event:EventOrigin@Offer', 'event:EventOrigin@Workflow', 'event:Action@Created', 'event:Action@Deleted', 'event:Action@Obtained', 'event:Action@Released', 'event:Action@statechange', 'event:concept:name@A_Accepted', 'event:concept:name@A_Cancelled', 'event:concept:name@A_Complete', 'event:concept:name@A_Concept', 'event:concept:name@A_Create Application', 'event:concept:name@A_Denied', 'event:concept:name@A_Incomplete', 'event:concept:name@A_Pending', 'event:concept:name@A_Submitted', 'event:concept:name@A_Validating', 'event:concept:name@O_Accepted', 'event:concept:name@O_Cancelled', 'event:concept:name@O_Created', 'event:concept:name@O_Refused', 'event:concept:name@O_Returned', 'event:concept:name@O_Sent (mail and online)', 'event:concept:name@O_Sent (online only)', 'event:concept:name@W_Assess potential fraud', 'event:concept:name@W_Call after offers', 'event:concept:name@W_Call incomplete files', 'event:concept:name@W_Complete application', 'event:concept:name@W_Handle leads', 'event:concept:name@W_Personal Loan collection', 'event:concept:name@W_Shortened completion ', 'event:concept:name@W_Validate application',  '@@max_concurrent_activities_general', '@@num_occurrence_O_Cancelled', '@@num_occurrence_A_Pending', '@@num_occurrence_W_Handle leads', '@@num_occurrence_O_Returned', '@@num_occurrence_A_Cancelled', '@@num_occurrence_A_Validating', '@@num_occurrence_A_Incomplete', '@@num_occurrence_A_Accepted', '@@num_occurrence_A_Concept', '@@num_occurrence_W_Personal Loan collection', '@@num_occurrence_A_Complete', '@@num_occurrence_W_Call after offers', '@@num_occurrence_A_Create Application', '@@num_occurrence_A_Submitted', '@@num_occurrence_W_Shortened completion ', '@@num_occurrence_O_Created', '@@num_occurrence_O_Accepted', '@@num_occurrence_O_Refused', '@@num_occurrence_W_Complete application', '@@num_occurrence_O_Sent (mail and online)', '@@num_occurrence_W_Assess potential fraud', '@@num_occurrence_W_Validate application', '@@num_occurrence_A_Denied', '@@num_occurrence_W_Call incomplete files', '@@num_occurrence_O_Sent (online only)', '@@case_id_column', 'case:ApplicationType@Limit raise', 'event:EventOrigin@Application', 'event:EventOrigin@Offer', 'event:EventOrigin@Workflow', 'event:Action@Created', 'event:Action@Deleted', 'event:Action@Obtained', 'event:Action@Released', 'event:Action@statechange', 'event:concept:name@A_Accepted', 'event:concept:name@A_Cancelled', 'event:concept:name@A_Complete', 'event:concept:name@A_Concept', 'event:concept:name@A_Create Application', 'event:concept:name@A_Denied', 'event:concept:name@A_Incomplete', 'event:concept:name@A_Pending', 'event:concept:name@A_Submitted', 'event:concept:name@A_Validating', 'event:concept:name@O_Accepted', 'event:concept:name@O_Cancelled', 'event:concept:name@O_Created', 'event:concept:name@O_Refused', 'event:concept:name@O_Returned', 'event:concept:name@O_Sent (mail and online)', 'event:concept:name@O_Sent (online only)', 'event:concept:name@W_Assess potential fraud', 'event:concept:name@W_Call after offers', 'event:concept:name@W_Call incomplete files', 'event:concept:name@W_Complete application', 'event:concept:name@W_Handle leads', 'event:concept:name@W_Personal Loan collection', 'event:concept:name@W_Shortened completion ', 'event:concept:name@W_Validate application',  '@@max_concurrent_activities_general', '@@num_occurrence_O_Cancelled', '@@num_occurrence_A_Pending', '@@num_occurrence_W_Handle leads', '@@num_occurrence_O_Returned', '@@num_occurrence_A_Cancelled', '@@num_occurrence_A_Validating', '@@num_occurrence_A_Incomplete', '@@num_occurrence_A_Accepted', '@@num_occurrence_A_Concept', '@@num_occurrence_W_Personal Loan collection', '@@num_occurrence_A_Complete', '@@num_occurrence_W_Call after offers', '@@num_occurrence_A_Create Application', '@@num_occurrence_A_Submitted', '@@num_occurrence_W_Shortened completion ', '@@num_occurrence_O_Created', '@@num_occurrence_O_Accepted', '@@num_occurrence_O_Refused', '@@num_occurrence_W_Complete application', '@@num_occurrence_O_Sent (mail and online)', '@@num_occurrence_W_Assess potential fraud', '@@num_occurrence_W_Validate application', '@@num_occurrence_A_Denied', '@@num_occurrence_W_Call incomplete files', '@@num_occurrence_O_Sent (online only)']
    #for item in features_to_delete:
    #    if item in data_in_vicinity.columns.to_list():
    #        data_in_vicinity = data_in_vicinity.drop(item, axis=1)

    data_in_vicinity = data_in_vicinity.filter(selected_feature_names)
    print('Unique values: ' + str(data_in_vicinity.nunique()))
    unique_values = data_in_vicinity.nunique()
    for key,value in unique_values.items():
        if value == 1:
            data_in_vicinity = data_in_vicinity.drop(key, axis=1)
            print('Dropping attribute ' + str(key) + ' with only one value')

    print(data_in_vicinity.head())

    nxgraph = nx.DiGraph()
    for node in nodes:
        nxgraph.add_node(node)
    
    for edge in edges:
        print('Adding edge ' + str(edge[0]) + ' - ' + str(edge[1]))
        if edge[2] == "T":
            nxgraph.add_edge(edge[0], edge[1])
        if edge[3] == "T":
            nxgraph.add_edge(edge[1], edge[0])

    attribute_list = []
    for item in data_in_vicinity.columns.to_list():
        attribute_list.append(item)

    adj_matrix = nx.to_numpy_matrix(nxgraph)
    adj_matrix = np.asarray(adj_matrix)
    labels = [f'{col}' for i, col in enumerate(attribute_list)]
    new_labels = []
    for label in labels:
        label = label.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
        new_labels.append(label)
    graph_dot = make_graph(adj_matrix, new_labels)
    graph_dot = str_to_dot(graph_dot.source)
    data_in_vicinity.columns = new_labels

    causal_estimates = []
    def estimate_effect(treatment, outcome):
        try:
            # Define Causal Model
            model=CausalModel(
                    data = data_in_vicinity,
                    treatment=treatment,
                    outcome=outcome,
                    graph=graph_dot)

            # Identification
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            print(identified_estimand)
            
            # Estimation
            estimate = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=0,
                                            treatment_value=1,
                                            confidence_intervals=True,
                                            test_significance=True)
            print("Causal Estimate is " + str(estimate.value))

            context['estimate'] = estimate.value
            return estimate.value
        except nx.exception.NetworkXError as e:
            print('Error: ' + str(e))
            context['Exception'] = 'Could not identify causal effects: "' + str(e) + '"'
            return None
        
    print('Edges: ' + str(nxgraph.edges))
    for edge in nxgraph.edges:
        node_1, node_2 = edge
        treatment = node_2.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
        outcome = node_1.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
        effect = estimate_effect(treatment, outcome)
        if effect:
            effect_estimate = round(effect, 3)
            causal_estimates.append((node_1, node_2, effect_estimate))
        outcome = node_2.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
        treatment = node_1.replace('<', '').replace('>', '').replace('@', '').replace('-', '').replace(':', '')
        effect = estimate_effect(treatment, outcome)
        if effect:
            effect_estimate = round(effect, 3)
            causal_estimates.append((node_1, node_2, effect_estimate))

    print('Causal estimates: ' + str(causal_estimates))

    context['data_in_vicinity'] = data_in_vicinity
    context['edges'] = edges
    context['nodes'] = nodes
    context['causal_estimates'] = causal_estimates
    context['target_attribute'] = target_attribute
    return context
