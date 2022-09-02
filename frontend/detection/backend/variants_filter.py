import pandas as pd
import datetime
from statistics import mean

from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics

from .importer import import_file

def get_variants(log_path, variant_filter_strategy):
    event_log = import_file(log_path, False)
    variants_count = case_statistics.get_variant_statistics(event_log)
    all_variants = {}
    #for variant in variants_count:
        # Filter for current variant
        #variants = []
        #variants.append(variant['variant'])
        #filtered_log = variants_filter.apply(event_log, variants)
        # Case duration for this variant
        #all_case_durations = case_statistics.get_all_casedurations(filtered_log)
        #variant['durations'] = all_case_durations
        #variant['avg_duration'] = mean(all_case_durations)

    #print(variants_count)
    if variant_filter_strategy == 'least_common_variant':
        variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=False)
    elif variant_filter_strategy == 'longest_throughput_time':
        variants_count = sorted(variants_count, key=lambda x: x['avg_duration'], reverse=True)
    elif variant_filter_strategy == 'shortest_throughput_time':
        variants_count = sorted(variants_count, key=lambda x: x['avg_duration'], reverse=False)
    else:
        variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)

    variantsdata_piechart = []
    column_names = ['Index', 'Percentage', 'Throughput time', 'Percentage Number', 'Variant']
    variants_pd_data = pd.DataFrame(columns = column_names)
    index_count = 1
    for variant in variants_count[:10]:
        variant_dict = {}
        variant_dict['Index'] = index_count
        decimal_percentage = variant['count'] / len(event_log)
        percentage = "{:.2%}".format(decimal_percentage)
        variant_dict['Percentage'] = percentage
        rounded_decimal_percentage = round(decimal_percentage, 2)
        rounded_decimal_percentage = rounded_decimal_percentage * 100
        variant_dict['Percentage Number'] = rounded_decimal_percentage
        rounded_decimal_percentage = round(rounded_decimal_percentage, 0)
        variantsdata_piechart.append(rounded_decimal_percentage)
        avg_variant_case_duration = 0 # mean(variant['durations'])
        avg_variant_case_duration = round(avg_variant_case_duration, 0)
        duration = datetime.timedelta(seconds=avg_variant_case_duration)
        variant_dict['Throughput time'] = str(duration)
        variant_dict['Variant'] = variant['variant']
        all_variants[index_count] = variant['variant']
        variants_pd_data = variants_pd_data.append(variant_dict, ignore_index = True)
        index_count = index_count + 1

    # For more than 10 variants, aggregate as 'Other'
    if len(variants_count) > 10:
        variant_dict = {}
        variant_dict['Index'] = 'Other'
        variants = []
        accumulate_count = 0
        durations_list = []
        for variant in variants_count[10:]:
            variants.append(variant['variant'])
            accumulate_count = accumulate_count + variant['count']
            #durations_list.extend(variant['durations'])
            all_variants[index_count] = variant['variant']
            index_count = index_count + 1
        filtered_log = variants_filter.apply(event_log, variants)
        decimal_percentage = accumulate_count / len(event_log)
        percentage = "{:.2%}".format(decimal_percentage)
        avg_variant_case_duration = 0 # mean(durations_list)
        avg_variant_case_duration = round(avg_variant_case_duration, 0)
        duration = datetime.timedelta(seconds=avg_variant_case_duration)
        rounded_decimal_percentage = round(decimal_percentage, 2)
        rounded_decimal_percentage = rounded_decimal_percentage * 100
        variant_dict['Percentage Number'] = rounded_decimal_percentage
        rounded_decimal_percentage = round(rounded_decimal_percentage, 0)
        variantsdata_piechart.append(rounded_decimal_percentage)
        variant_dict['Percentage'] = percentage
        variant_dict['Throughput time'] = str(duration)
        variant_dict['Variant'] = variant['variant']
        variants_pd_data = variants_pd_data.append(variant_dict, ignore_index = True)
    print(variants_pd_data.head())
    return event_log, variants_pd_data, all_variants, variantsdata_piechart 

def apply_auto_filter(log_path):
    event_log = import_file(log_path, False)
    auto_filtered_log = variants_filter.apply_auto_filter(event_log)
    return auto_filtered_log

def apply_percentage_filter(log_path, percentage):
    event_log = import_file(log_path, False)
    filtered_log = variants_filter.filter_log_variants_percentage(event_log, percentage=percentage)
    return filtered_log

def apply_variant_filter(event_log, all_variants, selected_variants):
    selected_variants_sequences = []
    for item in selected_variants:
        if item != 'Other':
            selected_variants_sequences.append(all_variants[item])
        else:
            for key, value in all_variants.items():
                if key > 10:
                    selected_variants_sequences.append(value)
    print(str(len(selected_variants_sequences)) +  ' selected Variants')
    print(selected_variants_sequences[:2])
    filtered_log = variants_filter.apply(event_log, selected_variants_sequences)
    return filtered_log