import os
import numpy as np
import pandas as pd

from learning_utils import get_standard_scaler, get_tfidf_transformer
from learning_utils import get_knn_classifier, get_grid_search_classifier, get_adaboost_classifier
from learning_utils import get_learning_pipeline, compute_classification_metrics_test_data, print_classification_metrics

def test_knn_classifier(dir_bovw_data="bovw_features_data/", file_csv_gs_cv="image_classification_gs_cv_knn.csv"):
    file_csv_gs_cv = os.path.join(dir_bovw_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(df_gs_cv)

    train_y = np.load(os.path.join(dir_bovw_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_bovw_data, "test_labels.npy"))

    list_train_acc = []
    list_test_acc = []
    list_f1 = []

    for idx_clf in range(df_gs_cv.shape(0)):
        num_visual_words = df_gs_cv.num_visual_words[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        metric = df_gs_cv.preprocess[idx_clf]
        n_neighbors = df_gs_cv.n_neighbors[idx_clf]
        weights = df_gs_cv.n_neighbors[weights]

        train_x = np.load(os.path.join(dir_bovw_data, f"train_kmeans_{num_visual_words}.npy"))
        test_x = np.load(os.path.join(dir_bovw_data, f"test_kmeans_{num_visual_words}.npy"))

        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
        elif preprocess == "tf_idf":
            preprocessor = get_tfidf_transformer()
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        classifier = get_knn_classifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
        pipeline = [("std_scaler", preprocessor), ("knn", classifier)]
        learning_pipeline = get_learning_pipeline(pipeline)
        print("learning pipeline")
        print(learning_pipeline)

        learning_pipeline.fit(train_x, train_y)
        train_acc = learning_pipeline.score(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_cm, test_auc = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print(f"train acc: {train_acc:.4f}")
        print_classification_metrics(test_acc, test_f1, test_cm)
    return

def test_adaboost_classifier(dir_bovw_data="bovw_features_data/", file_csv_gs_cv="image_classification_gs_cv_knn.csv"):
    file_csv_gs_cv = os.path.join(dir_bovw_data, file_csv_gs_cv)
    df_gs_cv = pd.read_csv(df_gs_cv)

    train_y = np.load(os.path.join(dir_bovw_data, "train_labels.npy"))
    test_y = np.load(os.path.join(dir_bovw_data, "test_labels.npy"))

    list_train_acc = []
    list_test_acc = []
    list_f1 = []

    for idx_clf in range(df_gs_cv.shape(0)):
        num_visual_words = df_gs_cv.num_visual_words[idx_clf]
        preprocess = df_gs_cv.preprocess[idx_clf]
        metric = df_gs_cv.preprocess[idx_clf]
        n_neighbors = df_gs_cv.n_neighbors[idx_clf]
        weights = df_gs_cv.n_neighbors[weights]

        train_x = np.load(os.path.join(dir_bovw_data, f"train_kmeans_{num_visual_words}.npy"))
        test_x = np.load(os.path.join(dir_bovw_data, f"test_kmeans_{num_visual_words}.npy"))

        if preprocess == "std_scaler":
            preprocessor = get_standard_scaler()
        elif preprocess == "tf_idf":
            preprocessor = get_tfidf_transformer()
        else:
            print(f"wrong option for preprocess: {preprocess} found, so exiting....")
            return

        classifier = get_knn_classifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
        pipeline = [("std_scaler", preprocessor), ("knn", classifier)]
        learning_pipeline = get_learning_pipeline(pipeline)
        print("learning pipeline")
        print(learning_pipeline)

        learning_pipeline.fit(train_x, train_y)
        train_acc = learning_pipeline.score(train_x, train_y)
        test_pred = learning_pipeline.predict(test_x)
        test_pred_prob = learning_pipeline.predict_proba(test_x)
        test_acc, test_f1, test_cm, test_auc = compute_classification_metrics_test_data(test_y, test_pred, test_pred_prob)
        print(f"train acc: {train_acc:.4f}")
        print_classification_metrics(test_acc, test_f1, test_cm)
    return


def do_cv_image_classification(dir_bovw_data="bovw_features_data/", which_classifier="knn", start_num_visual_words=None, end_num_visual_words=None):
    if start_num_visual_words is None:
        start_num_visual_words = 50
    if end_num_visual_words is None:
        end_num_visual_words = 205

    train_y = np.load(os.path.join(dir_bovw_data, "train_labels.npy"))

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
    else:
        print(f"wrong option : {which_classifier}")
        return

    list_all_params = sorted(list(param_grid.keys()))

    file_log_results=f"image_classification_gs_cv_{which_classifier}.csv"
    cv_col_names = ["clustering_algo", "num_visual_words", "preprocess", "classifier"] + list_all_params + ["best_cv_score"]
    cv_results_rows = []

    for num_visual_words in range(start_num_visual_words, end_num_visual_words, 5):
        train_x = np.load(os.path.join(dir_bovw_data, f"train_kmeans_{num_visual_words}.npy"))

        if which_classifier == "knn":
            classifier = get_knn_classifier()
        elif which_classifier == "adaboost":
            classifier = get_adaboost_classifier()

        list_pipelines = [
            [("std_scaler", get_standard_scaler()), ("grid_search", get_grid_search_classifier(classifier, param_grid))],
            [("tf_idf", get_tfidf_transformer()), ("grid_search", get_grid_search_classifier(classifier, param_grid))],
        ]

        for pipeline_idx in range(len(list_pipelines)):
            learning_pipeline = get_learning_pipeline(list_pipelines[pipeline_idx])
            print(f"num visual words : {num_visual_words}")
            print("learning pipeline")
            print(learning_pipeline)
            learning_pipeline.fit(train_x, train_y)
            print(f"best params : {learning_pipeline['grid_search'].best_params_}")
            print(f"best score : {learning_pipeline['grid_search'].best_score_}\n\n")

            if pipeline_idx == 0:
                preprocess_method = "std_scaler"
            else:
                preprocess_method = "tf_idf"

            param_values = []
            for param_key in list_all_params:
                param_values.append(learning_pipeline["grid_search"].best_params_[param_key])

            row_ = ["kmeans", num_visual_words, preprocess_method, which_classifier] \
                + param_values \
                + [round(learning_pipeline["grid_search"].best_score_, 4)]
            cv_results_rows.append(row_)

    df_cv_results = pd.DataFrame(cv_results_rows, columns=cv_col_names)
    df_cv_results.to_csv(os.path.join(dir_bovw_data, file_log_results), index=False)
    return
