import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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

def get_logistic_regression_model(max_iter=100):
    log_reg = LogisticRegression(max_iter=max_iter)
    return log_reg

def get_learning_pipeline(pipeline_list):
    learning_pipeline = Pipeline(pipeline_list)
    return learning_pipeline

def compute_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("----------------------")
    print("classification Metrics")
    print("----------------------")
    print(f"accuracy : {acc}")
    print(f"f1 score : {f1}")
    print("confustion matrix")
    print(cm)
    return
