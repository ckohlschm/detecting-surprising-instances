def apply_boxplots(event_log, target_feature_name):
    Q1 = event_log[target_feature_name].quantile(0.25)
    Q3 = event_log[target_feature_name].quantile(0.75)
    IQR = Q3 - Q1

    mean = event_log[target_feature_name].mean()
    print('Mean: ' + str(mean))

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 *IQR

    filter_better = (event_log[target_feature_name] < lower_bound)
    event_log_filter_better = event_log.loc[filter_better]
    print(event_log_filter_better.head())
    print("There are " + str(len(event_log_filter_better)) + " better performing situations")

    filter_better = (event_log[target_feature_name] > upper_bound)
    event_log_filter_worse = event_log.loc[filter_better]
    print(event_log_filter_worse.head())
    print("There are " + str(len(event_log_filter_worse)) + " worse performing situations")
    return event_log_filter_better, event_log_filter_worse, upper_bound, lower_bound, IQR, Q1, Q3