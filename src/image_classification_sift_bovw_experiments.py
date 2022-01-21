import os
import numpy as np
import pandas as pd

from logger_utils import CSVWriter

from sklearn.model_selection import cross_validate
from learning_utils import get_standard_scaler, get_tfidf_transformer, get_accuracy_scoring_fn
from learning_utils import get_learning_pipeline, compute_classification_metrics_test_data, print_classification_metrics
from learning_utils import get_knn_classifier, get_adaboost_classifier, get_gaussian_nb_classifier, get_grid_search_classifier, get_svc_classifier

def test_knn_classifier_sift(dir_sift_data="sift_features/", file_csv_gs_cv="image_classification_gs_cv_knn.csv"):
    file_csv_gs_cv = os.path.join(dir_sift_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(file_csv_gs_cv)

    train_y = np.load(os.path.join(dir_sift_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_sift_data, "test_labels.npy"))

    file_log_results = "image_classification_test_knn.csv"
    test_col_names = ["features", "num_visual_words", "preprocess", "classifier", "n_neighbors", "metric", "weights", "test_acc", "test_f1", "test_roc_auc"]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_sift_data, file_log_results), test_col_names)

    for idx_clf in range(len(df_gs_cv)):
        num_visual_words = df_gs_cv.num_visual_words[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        metric = df_gs_cv.metric[idx_clf]
        n_neighbors = df_gs_cv.n_neighbors[idx_clf]
        weights = df_gs_cv.weights[idx_clf]

        train_x = np.load(os.path.join(dir_sift_data, f"train_sift_{num_visual_words}.npy"))
        test_x = np.load(os.path.join(dir_sift_data, f"test_sift_{num_visual_words}.npy"))

        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
        elif preprocess == "tf_idf":
            preprocessor = get_tfidf_transformer()
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        classifier = get_knn_classifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
        pipeline = [(preprocess, preprocessor), ("knn", classifier)]
        learning_pipeline = get_learning_pipeline(pipeline)
        print(f"num visual words : {num_visual_words}")
        print(f"learning pipeline : {pipeline}")

        learning_pipeline.fit(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
        row_ = ["sift", num_visual_words, preprocess, "knn", n_neighbors, metric, weights, round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)]
        csv_writer.write_row(row_)
    csv_writer.close()
    return

def test_adaboost_classifier_sift(dir_sift_data="sift_features/", file_csv_gs_cv="image_classification_gs_cv_adaboost.csv"):
    file_csv_gs_cv = os.path.join(dir_sift_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(file_csv_gs_cv)

    train_y = np.load(os.path.join(dir_sift_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_sift_data, "test_labels.npy"))

    file_log_results = "image_classification_test_adaboost.csv"
    test_col_names = [
        "features", "num_visual_words", "preprocess", "classifier",
        "base_estimator__criterion", "base_estimator__max_depth", "base_estimator__min_samples_split", "base_estimator__splitter",
        "learning_rate", "n_estimators", "test_acc", "test_f1", "test_roc_auc"
    ]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_sift_data, file_log_results), test_col_names)

    for idx_clf in range(len(df_gs_cv)):
        num_visual_words = df_gs_cv.num_visual_words[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        criterion = df_gs_cv.base_estimator__criterion[idx_clf]
        max_depth = df_gs_cv.base_estimator__max_depth[idx_clf]
        min_samples_split = df_gs_cv.base_estimator__min_samples_split[idx_clf]
        splitter = df_gs_cv.base_estimator__splitter[idx_clf]
        learning_rate = df_gs_cv.learning_rate[idx_clf]
        n_estimators = df_gs_cv.n_estimators[idx_clf]

        train_x = np.load(os.path.join(dir_sift_data, f"train_sift_{num_visual_words}.npy"))
        test_x = np.load(os.path.join(dir_sift_data, f"test_sift_{num_visual_words}.npy"))

        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
        elif preprocess == "tf_idf":
            preprocessor = get_tfidf_transformer()
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        classifier = get_adaboost_classifier(
            n_estimators=n_estimators, learning_rate=learning_rate, criterion=criterion,
            max_depth=max_depth, min_samples_split=min_samples_split, splitter=splitter
        )
        pipeline = [(preprocess, preprocessor), ("adaboost", classifier)]
        learning_pipeline = get_learning_pipeline(pipeline)
        print("learning pipeline")
        print(learning_pipeline)

        learning_pipeline.fit(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
        row_ = [
            "sift", num_visual_words, preprocess, "adaboost", criterion, max_depth, min_samples_split, splitter,
            learning_rate, n_estimators, round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)
        ]
        csv_writer.write_row(row_)
    csv_writer.close()
    return

def test_svc_classifier_sift(dir_sift_data="sift_features/", file_csv_gs_cv="image_classification_gs_cv_svc.csv"):
    file_csv_gs_cv = os.path.join(dir_sift_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(file_csv_gs_cv)

    train_y = np.load(os.path.join(dir_sift_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_sift_data, "test_labels.npy"))

    file_log_results = "image_classification_test_svc.csv"
    test_col_names = ["features", "num_visual_words", "preprocess", "classifier", "C", "kernel", "degree", "gamma", "test_acc", "test_f1", "test_roc_auc"]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_sift_data, file_log_results), test_col_names)

    for idx_clf in range(len(df_gs_cv)):
        num_visual_words = df_gs_cv.num_visual_words[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        C = df_gs_cv.C[idx_clf]
        kernel = df_gs_cv.kernel[idx_clf]
        degree = df_gs_cv.degree[idx_clf]
        gamma = df_gs_cv.gamma[idx_clf]

        train_x = np.load(os.path.join(dir_sift_data, f"train_sift_{num_visual_words}.npy"))
        test_x = np.load(os.path.join(dir_sift_data, f"test_sift_{num_visual_words}.npy"))

        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
        elif preprocess == "tf_idf":
            preprocessor = get_tfidf_transformer()
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        classifier = get_svc_classifier(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True)
        pipeline = [(preprocess, preprocessor), ("svc", classifier)]
        learning_pipeline = get_learning_pipeline(pipeline)
        print(f"num visual words : {num_visual_words}")
        print(f"learning pipeline : {pipeline}")

        learning_pipeline.fit(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
        row_ = ["sift", num_visual_words, preprocess, "svc", C, kernel, degree, gamma, round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)]
        csv_writer.write_row(row_)
    csv_writer.close()
    return

def do_gs_cv_image_classification_sift(dir_sift_data="sift_features/", which_classifier="knn", start_num_visual_words=None, end_num_visual_words=None):
    if start_num_visual_words is None:
        start_num_visual_words = 5
    if end_num_visual_words is None:
        end_num_visual_words = 205

    train_y = np.load(os.path.join(dir_sift_data, "train_labels.npy"))

    if which_classifier == "knn":
        param_grid = {
            "n_neighbors" : np.arange(6, 25),
            "weights": ["uniform", "distance"],
            "metric" : ["euclidean", "chebyshev", "manhattan"],
        }
    elif which_classifier == "adaboost":
        param_grid = {
            "base_estimator__criterion" : ["gini", "entropy"],
            "base_estimator__splitter" :   ["best", "random"],
            "base_estimator__max_depth" : np.arange(5, 11),
            "base_estimator__min_samples_split" : np.arange(2, 5),
            "n_estimators" : np.arange(10, 160, 10),
            "learning_rate" : [0.01, 0.1, 1],
        }
    elif which_classifier == "svc":
        param_grid = {
            "C" : list(np.around(np.arange(0.1, 1, 0.1), decimals=1)) + list(np.arange(1, 11, 1)),
            "kernel" : ["linear", "poly", "rbf", "sigmoid"],
            "degree" : np.arange(2, 6),
            "gamma" : ["scale", "auto"],
        }
    else:
        print(f"wrong option : {which_classifier}")
        return

    list_all_params = sorted(list(param_grid.keys()))

    file_log_results=f"image_classification_gs_cv_{which_classifier}.csv"
    cv_col_names = ["features", "num_visual_words", "preprocess", "classifier"] + list_all_params + ["best_cv_score"]
    cv_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_sift_data, file_log_results), cv_col_names)

    for num_visual_words in range(start_num_visual_words, end_num_visual_words, 5):
        train_x = np.load(os.path.join(dir_sift_data, f"train_sift_{num_visual_words}.npy"))

        if which_classifier == "knn":
            classifier = get_knn_classifier()
        elif which_classifier == "adaboost":
            classifier = get_adaboost_classifier()
        elif which_classifier == "svc":
            classifier = get_svc_classifier()

        list_pipelines = [
            [("std_scaler", get_standard_scaler()), ("grid_search", get_grid_search_classifier(classifier, param_grid))],
            [("tf_idf", get_tfidf_transformer()), ("grid_search", get_grid_search_classifier(classifier, param_grid))],
        ]

        for idx_pipeline in range(len(list_pipelines)):
            learning_pipeline = get_learning_pipeline(list_pipelines[idx_pipeline])
            print(f"num visual words : {num_visual_words}")
            print("learning pipeline")
            print(learning_pipeline)
            learning_pipeline.fit(train_x, train_y)
            print(f"best params : {learning_pipeline['grid_search'].best_params_}")
            print(f"best score : {learning_pipeline['grid_search'].best_score_}\n\n")

            if idx_pipeline == 0:
                preprocess_method = "std_scaler"
            else:
                preprocess_method = "tf_idf"

            param_values = []
            for param_key in list_all_params:
                param_values.append(learning_pipeline["grid_search"].best_params_[param_key])

            row_ = ["sift", num_visual_words, preprocess_method, which_classifier] \
                + param_values \
                + [round(learning_pipeline["grid_search"].best_score_, 4)]
            csv_writer.write_row(row_)
    csv_writer.close()
    return

"""
def do_cv_nb_image_classification_sift(dir_sift_data, start_num_visual_words=None, end_num_visual_words=None):
    if start_num_visual_words is None:
        start_num_visual_words = 5
    if end_num_visual_words is None:
        end_num_visual_words = 205

    scoring_fn = get_accuracy_scoring_fn()
    train_y = np.load(os.path.join(dir_sift_data, "train_labels.npy"))

    file_log_results = "image_classification_cv_gaussian_nb.csv"
    cv_col_names = ["features", "num_visual_words", "preprocess", "classifier", "best_cv_score"]
    cv_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_sift_data, file_log_results), cv_col_names)

    for num_visual_words in range(start_num_visual_words, end_num_visual_words, 5):
        train_x = np.load(os.path.join(dir_sift_data, f"train_sift_{num_visual_words}.npy"))
        preprocess_method = "std_scaler"
        pipeline = [(preprocess_method, get_standard_scaler()), ("gaussian_nb", get_gaussian_nb_classifier())]
        learning_pipeline = get_learning_pipeline(pipeline)
        clf_cv = cross_validate(learning_pipeline, train_x, train_y, cv=5, scoring=scoring_fn)

        row_ = ["sift", num_visual_words, preprocess_method, "gaussian_nb"] \
            + [round(np.mean(clf_cv["test_score"]), 4)]
        csv_writer.write_row(row_)
    csv_writer.close()
    return

def test_nb_image_classifier_sift(dir_sift_data, start_num_visual_words=None, end_num_visual_words=None):
    if start_num_visual_words is None:
        start_num_visual_words = 5
    if end_num_visual_words is None:
        end_num_visual_words = 205

    scoring_fn = get_accuracy_scoring_fn()
    train_y = np.load(os.path.join(dir_sift_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_sift_data, "test_labels.npy"))

    file_log_results = "image_classification_test_gaussian_nb.csv"
    test_col_names = ["features", "num_visual_words", "preprocess", "classifier", "test_acc", "test_f1", "test_roc_auc"]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_sift_data, file_log_results), test_col_names)

    for num_visual_words in range(start_num_visual_words, end_num_visual_words, 5):
        print(f"num visual words : {num_visual_words}")
        train_x = np.load(os.path.join(dir_sift_data, f"train_sift_{num_visual_words}.npy"))
        test_x = np.load(os.path.join(dir_sift_data, f"test_sift_{num_visual_words}.npy"))

        preprocess_method = "std_scaler"
        pipeline = [(preprocess_method, get_standard_scaler()), ("gaussian_nb", get_gaussian_nb_classifier())]
        learning_pipeline = get_learning_pipeline(pipeline)
        learning_pipeline.fit(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)

        row_ = ["sift", num_visual_words, preprocess_method, "gaussian_nb", round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)]
        print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
        csv_writer.write_row(row_)
    csv_writer.close()
    return
"""
