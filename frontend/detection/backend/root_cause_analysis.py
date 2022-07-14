from pm4py.visualization.decisiontree import visualizer as dectree_visualizer
from sklearn import tree
import numpy as np

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