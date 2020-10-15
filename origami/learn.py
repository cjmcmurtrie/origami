import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve


def get_scores(probas, true_labels, class_to_score='positive', class_ix=1, precision_threshold=0.7):
    '''

    :param probas: probabilities to score
    :param true_labels: the multiclass labels
    :param class_to_score: the actual class label, e.g. 'positive'
    :param class_ix: the column index of the class (in the probabilities matrix)
    :param precision_threshold: the minimum acceptable precision on thresholding
    :return: Tuple(float or None)
    '''
    try:
        auc = roc_auc_score(
            true_labels,
            probas,
            average='macro',
            multi_class='ovo'
        )
    except ValueError:
        auc = 'AUC undefined'
    prec, reca, thre = precision_recall_curve(
        true_labels == class_to_score,
        probas[:, class_ix]
    )
    ix_list = [i for i, p in enumerate(prec) if p >= precision_threshold]
    if ix_list:
        p = round(prec[ix_list[0]], 3)
        r = round(reca[ix_list[0]], 3)
        t = round(thre[ix_list[0] - 1], 3)
        return auc, p, r, t
    else:
        return auc, (None, None, None)


def get_model():
    '''
    Just a simple model factory function, returning a balanced random forest classifier
    :return: RandomForestClassifier
    '''
    return RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced'
    )


def data_split(train_data, test_data, feature_columns, target_column):
    '''
    Split out the datasets into features and labels
    :param train_data: the training dataframe
    :param test_data: the testing dataframe
    :param feature_columns: feature column names to input for training
    :param target_column: target label column name
    :return: Tuple[pd.DataFrame]
    '''
    return (
        train_data[feature_columns],
        train_data[target_column],
        test_data[feature_columns],
        test_data[target_column]
    )


def run_evaluation(
        dataset, feature_columns, target_column,
        time_column, evaluation_time_window='M', eval_start_time='2018-02-01'
):
    '''

    :param dataset:
    :param feature_columns:
    :param target_column:
    :param time_column:
    :param evaluation_time_window:
    :param eval_start_time:
    :return:
    '''
    results = []
    time_grouped = dataset \
        .set_index(time_column) \
        .groupby(pd.Grouper(freq=evaluation_time_window))
    eval_start_time = pd.to_datetime(eval_start_time)
    for test_period, test_data in time_grouped:
        min_test_date = test_data.index.min()
        if min_test_date >= eval_start_time:
            print('evaluating test period', test_period)
            train_data = dataset[dataset[time_column] < min_test_date]
            train_x, train_y, test_x, test_y = data_split(
                train_data,
                test_data,
                feature_columns=feature_columns,
                target_column=target_column
            )
            model = get_model().fit(train_x, train_y)
            probas = model.predict_proba(test_x)
            auc, pos_precision, pos_recall, pos_threshold = get_scores(
                probas,
                test_y,
                class_to_score='positive',
                class_ix=1,
                precision_threshold=0.7
            )
            results.append({
                'eval_period': test_period,
                'auc': auc,
                'precision': pos_precision,
                'recall': pos_recall,
                'threshold': pos_threshold
            })
    return pd.DataFrame(results)
