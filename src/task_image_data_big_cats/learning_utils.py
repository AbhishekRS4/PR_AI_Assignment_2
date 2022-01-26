import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, roc_auc_score, silhouette_score


def split_data(x, y, test_size=0.2, random_state=4):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return train_x, test_x, train_y, test_y

def get_encoded_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return label_encoder, encoded_labels

def get_standard_scaler():
    std_scaler = StandardScaler()
    return std_scaler

def get_tfidf_transformer():
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer

def get_logistic_regression_classifier(max_iter=100):
    clf_log_reg = LogisticRegression(max_iter=max_iter)
    return clf_log_reg

def get_gaussian_nb_classifier():
    clf_gaussian_nb = GaussianNB()
    return clf_gaussian_nb

def get_svc_classifier(C=None, kernel=None, degree=None, gamma=None, probability=False, random_state=4):
    if C is None:
        clf_svc = SVC(probability=probability, random_state=random_state)
    else:
        clf_svc = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=probability, random_state=random_state)
    return clf_svc

def get_knn_classifier(n_neighbors=None, metric=None, weights=None):
    if n_neighbors is None:
        clf_knn = KNeighborsClassifier()
    else:
        clf_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    return clf_knn

def get_adaboost_classifier(n_estimators=None, learning_rate=None, criterion=None, splitter=None, max_depth=None, min_samples_split=None, random_state=4):
    if n_estimators is None:
        clf_dt = DecisionTreeClassifier(random_state=random_state)
        clf_adaboost = AdaBoostClassifier(clf_dt, random_state=random_state)
    else:
        clf_dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
        clf_adaboost = AdaBoostClassifier(clf_dt, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    return clf_adaboost

def get_grid_search_classifier(estimator, param_grid, cv=5):
    scoring_func = make_scorer(accuracy_score)
    grid_search_cv = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring_func)
    return grid_search_cv

def get_learning_pipeline(pipeline_list):
    learning_pipeline = Pipeline(pipeline_list)
    return learning_pipeline

def get_accuracy_scoring_fn():
    acc_scoring_fn = make_scorer(accuracy_score)
    return acc_scoring_fn

def compute_classification_metrics_test_data(y_true, y_pred, y_pred_probs):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr")
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, roc_auc, cm

def print_classification_metrics(acc, f1, roc_auc, cm):
    print("----------------------")
    print("classification metrics")
    print("----------------------")
    print(f"accuracy : {acc:.4f}")
    print(f"f1 score : {f1:.4f}")
    print(f"roc auc : {roc_auc:.4f}")
    print("confustion matrix")
    print(cm)
    print()
    return

def get_dbscan_clustering_algo(eps):
    dbscan_clustering_algo = DBSCAN(eps=eps)
    return dbscan_clustering_algo

def compute_silhouette_score(X, labels):
    sil_score = silhouette_score(X, labels)
    return sil_score

def get_nearest_neighbors(n_neighbors=5):
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    return nearest_neighbors
