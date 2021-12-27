import os
import cv2
import pandas as pd
import numpy as np

class NumericalDatasetLoader:
    """
    NumericalDatasetLoader class to load numerical datasets, provided that
    the data and the labels are separate csv-files in the same directory

    Attributes
    ----------
    dir_dataset : str
        valid full directory path of the dataset
    num_categories : int
        number of different labels in the dataset
    data : ndarray
        all the data features (as a pandas dataframe)
    labels : ndarray
        all the data labels (as a pandas dataframe)

    Methods
    -------
    load_dataset() :    loads all the feature vectors in the data and the
                        labels, both as a pandas dataframe
    """

    def __init__(self, dir_dataset):
        """
        Arguments
        ---------
        dir_dataset : str
            valid full directory path of the dataset, which should be a directory
            containing both the data and the corresponding labels (either in
            seperate csv-files or one csv-files)
        """

        self.dir_dataset = dir_dataset
        self.num_categories = None
        self.data = None
        self.labels = None


    def load_dataset(self):
        """
        Method
        ------
        loads all the feature vectors in the data and the labels, both as a
        pandas dataframe
        """

        dir = self.dir_dataset
        print(f"dataset directory : {dir}")
        files = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
        csv_files = [file for file in files if file.endswith(".csv")]
        print(f"files in directory: {csv_files}")

        if (os.path.isfile(os.path.join(dir, "data.csv")) and os.path.isfile(os.path.join(dir, "labels.csv"))):
            # The data and the labels are in separate files
            self.data = pd.read_csv(os.path.join(dir, "data.csv"), index_col = 0)
            self.labels = pd.read_csv(os.path.join(dir, "labels.csv"))
        else: # this is of course not always true but works for the intended purpose
            # The data and the labels are in one file, like in the creditcards.csv file
            data_csv = head(csv_files)
            data_and_labels = pd.read_csv(data_csv)
            self.data = data_and_labels
            # I chose not to split the "Amount" and "Class" columns from the rest
            # of the data, since I do not know how Luke intends to solve this assignment


        # print(f"shape if the dataset: {}")
