from enum import Enum
from scipy.spatial import distance

class Variants(Enum):
    PEARSON = 'pearson',
    ASSOCIATION_RULE = 'association_rule'

def extract_features_pearson(pd_data_event_log, target_feature, threshold):
    feature_list = []
    correlation = pd_data_event_log.corr(method='pearson')
    #print(correlation[target_feature])

    features_below_threshold = correlation[correlation[target_feature]<=-1*threshold]
    #print(features_below_threshold)
    feature_list.extend(features_below_threshold.index.tolist())
    features_above_threshold = correlation[correlation[target_feature]>=threshold]
    #print(features_above_threshold)
    feature_list.extend(features_above_threshold.index.tolist())
    feature_list.remove(target_feature)

    return feature_list

def extract_features_distance(pd_data_event_log, target_feature, threshold):
    def distance_correlation(a,b):
        return distance.correlation(a,b)

    feature_list = []
    correlation = pd_data_event_log.corr(method=distance_correlation)
    print(correlation[target_feature])

    features_above_threshold = correlation[correlation[target_feature]>=threshold]
    print(features_above_threshold)
    feature_list.extend(features_above_threshold.index.tolist())
    if target_feature in feature_list: feature_list.remove(target_feature)

    return feature_list

def extract_features(pd_data_event_log, variant, target_feature, threshold):
    if variant == Variants.PEARSON:
        return extract_features_pearson(pd_data_event_log=pd_data_event_log, target_feature=target_feature,threshold=threshold)
    elif variant == Variants.DISTANCE:
        return extract_features_distance(pd_data_event_log=pd_data_event_log, target_feature=target_feature,threshold=threshold)
    elif variant == Variants.ASSOCIATION_RULE:
        print('Association Rules')
    else:
        print('Undefined feature extraction method')