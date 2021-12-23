import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def split_data(x, y, test_size=0.2, random_state=4):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return train_x, test_x, train_y, test_y

def get_encoded_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return label_encoder, encoded_labels

def get_logistic_regression_model():
    log_reg = LogisticRegression()
    return log_reg
