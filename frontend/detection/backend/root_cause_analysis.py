import numpy as np
import pandas as pd
import json
from dowhy import CausalModel
from cdt.causality.graph import GES

import graphviz
import networkx as nx 

from .detection_util import transform_event_log_to_situations

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

    situation_type = request.session['situation_type']
    if situation_type == 'event':
        selected_feature_names = data_in_vicinity.columns.to_list()
        
        if '@@case_id_column' in selected_feature_names:
            selected_feature_names.remove('@@case_id_column')
    else:
        selected_feature_names = request.session['selected_feature_names']
        print(selected_feature_names)
        selected_feature_names.append(request.session['target_attribute'])

    data_in_vicinity = data_in_vicinity.filter(selected_feature_names)
    print('Unique values: ' + str(data_in_vicinity.nunique()))
    unique_values = data_in_vicinity.nunique()
    for key,value in unique_values.items():
        if value == 1:
            data_in_vicinity = data_in_vicinity.drop(key, axis=1)
            print('Dropping attribute ' + str(key) + ' with only one value')

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
            else:
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
    
    situation_type = request.session['situation_type']
    if situation_type == 'event':
        selected_feature_names = data_in_vicinity.columns.to_list()
        
        if '@@case_id_column' in selected_feature_names:
            selected_feature_names.remove('@@case_id_column')
    else:
        selected_feature_names = request.session['selected_feature_names']
        print(selected_feature_names)
        selected_feature_names.append(request.session['target_attribute'])

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
        else:
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
