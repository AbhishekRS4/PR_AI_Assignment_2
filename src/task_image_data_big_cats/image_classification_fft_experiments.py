import os
import numpy as np
import pandas as pd

from logger_utils import CSVWriter

from sklearn.model_selection import cross_validate
from learning_utils import get_standard_scaler, get_tfidf_transformer, get_accuracy_scoring_fn
from learning_utils import get_learning_pipeline, compute_classification_metrics_test_data, print_classification_metrics
from learning_utils import get_knn_classifier, get_adaboost_classifier, get_gaussian_nb_classifier, get_grid_search_classifier, get_svc_classifier

def test_knn_classifier_fft(dir_fft_data="fft_features/", file_csv_gs_cv="image_classification_gs_cv_knn.csv"):
    file_csv_gs_cv = os.path.join(dir_fft_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(file_csv_gs_cv)

    train_y = np.load(os.path.join(dir_fft_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_fft_data, "test_labels.npy"))

    file_log_results = "image_classification_test_knn.csv"
    test_col_names = ["features", "num_dims", "preprocess", "classifier", "n_neighbors", "metric", "weights", "test_acc", "test_f1", "test_roc_auc"]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_fft_data, file_log_results), test_col_names)

    for idx_clf in range(len(df_gs_cv)):
        num_dims = df_gs_cv.num_dims[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        metric = df_gs_cv.metric[idx_clf]
        n_neighbors = df_gs_cv.n_neighbors[idx_clf]
        weights = df_gs_cv.weights[idx_clf]

        train_x = np.load(os.path.join(dir_fft_data, f"train_fft_{num_dims}.npy"))
        test_x = np.load(os.path.join(dir_fft_data, f"test_fft_{num_dims}.npy"))

        classifier = get_knn_classifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
            pipeline = [(preprocess, preprocessor), ("knn", classifier)]
        elif preprocess == "none":
            pipeline = [("knn", classifier)]
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        learning_pipeline = get_learning_pipeline(pipeline)
        print(f"num dims : {num_dims}")
        print(f"learning pipeline : {pipeline}")

        learning_pipeline.fit(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
        row_ = ["fft", num_dims, preprocess, "knn", n_neighbors, metric, weights, round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)]
        csv_writer.write_row(row_)
    csv_writer.close()
    return

def test_adaboost_classifier_fft(dir_fft_data="fft_features/", file_csv_gs_cv="image_classification_gs_cv_adaboost.csv"):
    file_csv_gs_cv = os.path.join(dir_fft_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(file_csv_gs_cv)

    train_y = np.load(os.path.join(dir_fft_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_fft_data, "test_labels.npy"))

    file_log_results = "image_classification_test_adaboost.csv"
    test_col_names = [
        "features", "num_dims", "preprocess", "classifier",
        "base_estimator__criterion", "base_estimator__max_depth", "base_estimator__min_samples_split", "base_estimator__splitter",
        "learning_rate", "n_estimators", "test_acc", "test_f1", "test_roc_auc"
    ]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_fft_data, file_log_results), test_col_names)

    for idx_clf in range(len(df_gs_cv)):
        num_dims = df_gs_cv.num_dims[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        criterion = df_gs_cv.base_estimator__criterion[idx_clf]
        max_depth = df_gs_cv.base_estimator__max_depth[idx_clf]
        min_samples_split = df_gs_cv.base_estimator__min_samples_split[idx_clf]
        splitter = df_gs_cv.base_estimator__splitter[idx_clf]
        learning_rate = df_gs_cv.learning_rate[idx_clf]
        n_estimators = df_gs_cv.n_estimators[idx_clf]

        train_x = np.load(os.path.join(dir_fft_data, f"train_fft_{num_dims}.npy"))
        test_x = np.load(os.path.join(dir_fft_data, f"test_fft_{num_dims}.npy"))

        classifier = get_adaboost_classifier(
            n_estimators=n_estimators, learning_rate=learning_rate, criterion=criterion,
            max_depth=max_depth, min_samples_split=min_samples_split, splitter=splitter
        )
        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
            pipeline = [(preprocess, preprocessor), ("adaboost", classifier)]
        elif preprocess == "none":
            pipeline = [("adaboost", classifier)]
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        learning_pipeline = get_learning_pipeline(pipeline)
        print("learning pipeline")
        print(learning_pipeline)

        learning_pipeline.fit(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
        row_ = [
            "fft", num_dims, preprocess, "adaboost", criterion, max_depth, min_samples_split, splitter,
            learning_rate, n_estimators, round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)
        ]
        csv_writer.write_row(row_)
    csv_writer.close()
    return

def test_svc_classifier_fft(dir_fft_data="fft_features/", file_csv_gs_cv="image_classification_gs_cv_svc.csv"):
    file_csv_gs_cv = os.path.join(dir_fft_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(file_csv_gs_cv)

    train_y = np.load(os.path.join(dir_fft_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_fft_data, "test_labels.npy"))

    file_log_results = "image_classification_test_svc.csv"
    test_col_names = [
        "features", "num_dims", "preprocess", "classifier",
        "C", "kernel", "degree", "gamma",
        "test_acc", "test_f1", "test_roc_auc"
    ]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_fft_data, file_log_results), test_col_names)

    for idx_clf in range(len(df_gs_cv)):
        num_dims = df_gs_cv.num_dims[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        C = df_gs_cv.C[idx_clf]
        kernel = df_gs_cv.kernel[idx_clf]
        degree = df_gs_cv.degree[idx_clf]
        gamma = df_gs_cv.gamma[idx_clf]

        train_x = np.load(os.path.join(dir_fft_data, f"train_fft_{num_dims}.npy"))
        test_x = np.load(os.path.join(dir_fft_data, f"test_fft_{num_dims}.npy"))

        classifier = get_svc_classifier(
            C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True
        )
        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
            pipeline = [(preprocess, preprocessor), ("svc", classifier)]
        elif preprocess == "none":
            pipeline = [("svc", classifier)]
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        learning_pipeline = get_learning_pipeline(pipeline)
        print("learning pipeline")
        print(learning_pipeline)

        learning_pipeline.fit(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
        row_ = [
            "fft", num_dims, preprocess, "svc", C, kernel, degree, gamma,
            round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)
        ]
        csv_writer.write_row(row_)
    csv_writer.close()
    return

def do_gs_cv_image_classification_fft(dir_fft_data="fft_features/", which_classifier="knn", start_num_dims=None, end_num_dims=None):
    if start_num_dims is None:
        start_num_dims = 50
    if end_num_dims is None:
        end_num_dims = 130

    train_y = np.load(os.path.join(dir_fft_data, "train_labels.npy"))

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
            "base_estimator__max_depth" : [10],
            "base_estimator__min_samples_split" : [2],
            "n_estimators" : [50],
            "learning_rate" : [0.1, 1],
        }
    elif which_classifier == "svc":
        param_grid = {
            "C" : [1],
            "kernel" : ["linear", "poly", "rbf", "sigmoid"],
            "degree" : np.arange(2, 6),
            "gamma" : ["scale"]
        }
    else:
        print(f"wrong option : {which_classifier}")
        return

    list_all_params = sorted(list(param_grid.keys()))

    file_log_results=f"image_classification_gs_cv_{which_classifier}.csv"
    cv_col_names = ["features", "num_dims", "preprocess", "classifier"] + list_all_params + ["best_cv_score"]
    cv_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_fft_data, file_log_results), cv_col_names)
    preprocess_methods = ["none", "std_scaler"]

    for num_dims in range(start_num_dims, end_num_dims, 10):
        train_x = np.load(os.path.join(dir_fft_data, f"train_fft_{num_dims}.npy"))

        if which_classifier == "knn":
            classifier = get_knn_classifier()
        elif which_classifier == "adaboost":
            classifier = get_adaboost_classifier()
        elif which_classifier == "svc":
            classifier = get_svc_classifier()

        for preprocess_method in preprocess_methods:
            if preprocess_method == "std_scaler":
                pipeline = [(preprocess_method, get_standard_scaler()), ("grid_search", get_grid_search_classifier(classifier, param_grid))]
            else:
                pipeline = [("grid_search", get_grid_search_classifier(classifier, param_grid))]
            learning_pipeline = get_learning_pipeline(pipeline)
            print(f"num dims : {num_dims}")
            print("learning pipeline")
            print(learning_pipeline)
            learning_pipeline.fit(train_x, train_y)
            print(f"best params : {learning_pipeline['grid_search'].best_params_}")
            print(f"best score : {learning_pipeline['grid_search'].best_score_}\n\n")

            param_values = []
            for param_key in list_all_params:
                param_values.append(learning_pipeline["grid_search"].best_params_[param_key])

            row_ = ["fft", num_dims, preprocess_method, which_classifier] \
                + param_values \
                + [round(learning_pipeline["grid_search"].best_score_, 4)]
            csv_writer.write_row(row_)
    csv_writer.close()
    return

"""
def do_cv_nb_image_classification_fft(dir_fft_data, start_num_dims=None, end_num_dims=None):
    if start_num_dims is None:
        start_num_dims = 50
    if end_num_dims is None:
        end_num_dims = 130

    scoring_fn = get_accuracy_scoring_fn()
    train_y = np.load(os.path.join(dir_fft_data, "train_labels.npy"))

    file_log_results = "image_classification_cv_gaussian_nb.csv"
    cv_col_names = ["features", "num_dims", "preprocess", "classifier", "best_cv_score"]
    cv_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_fft_data, file_log_results), cv_col_names)

    preprocess_methods = ["none", "std_scaler"]

    for num_dims in range(start_num_dims, end_num_dims, 10):
        train_x = np.load(os.path.join(dir_fft_data, f"train_fft_{num_dims}.npy"))

        for preprocess_method in preprocess_methods:
            if preprocess_method == "std_scaler":
                pipeline = [(preprocess_method, get_standard_scaler()), ("gaussian_nb", get_gaussian_nb_classifier())]
            else:
                pipeline = [("gaussian_nb", get_gaussian_nb_classifier())]

            learning_pipeline = get_learning_pipeline(pipeline)
            clf_cv = cross_validate(learning_pipeline, train_x, train_y, cv=5, scoring=scoring_fn)

            row_ = ["fft", num_dims, preprocess_method, "gaussian_nb"] \
                + [round(np.mean(clf_cv["test_score"]), 4)]
            csv_writer.write_row(row_)
    csv_writer.close()
    return

def test_nb_image_classifier_fft(dir_fft_data, start_num_dims=None, end_num_dims=None):
    if start_num_dims is None:
        start_num_dims = 50
    if end_num_dims is None:
        end_num_dims = 130

    scoring_fn = get_accuracy_scoring_fn()
    train_y = np.load(os.path.join(dir_fft_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_fft_data, "test_labels.npy"))

    file_log_results = "image_classification_test_gaussian_nb.csv"
    test_col_names = ["features", "num_dims", "preprocess", "classifier", "test_acc", "test_f1", "test_roc_auc"]
    test_results_rows = []
    csv_writer = CSVWriter(os.path.join(dir_fft_data, file_log_results), test_col_names)

    preprocess_methods = ["none", "std_scaler"]

    for num_dims in range(start_num_dims, end_num_dims, 10):
        print(f"num dims : {num_dims}")
        train_x = np.load(os.path.join(dir_fft_data, f"train_fft_{num_dims}.npy"))
        test_x = np.load(os.path.join(dir_fft_data, f"test_fft_{num_dims}.npy"))

        for preprocess_method in preprocess_methods:

            if preprocess_method == "std_scaler":
                pipeline = [(preprocess_method, get_standard_scaler()), ("gaussian_nb", get_gaussian_nb_classifier())]
            else:
                pipeline = [("gaussian_nb", get_gaussian_nb_classifier())]

            learning_pipeline = get_learning_pipeline(pipeline)
            learning_pipeline.fit(train_x, train_y)
            test_pred = learning_pipeline.predict(test_x)
            test_pred_prob = learning_pipeline.predict_proba(test_x)
            test_acc, test_f1, test_roc_auc, test_cm = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)

            row_ = ["fft", num_dims, preprocess_method, "gaussian_nb", round(test_acc, 4), round(test_f1, 4), round(test_roc_auc, 4)]
            print_classification_metrics(test_acc, test_f1, test_roc_auc, test_cm)
            csv_writer.write_row(row_)
    csv_writer.close()
    return
"""
