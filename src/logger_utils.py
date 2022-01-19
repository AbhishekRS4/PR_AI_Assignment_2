import os
import csv

class CSVWriter:
    """
    for writing tabular data to a csv file
    """
    def __init__(self, file_name, column_names):
        """
        ---------
        Arguments
        ---------
        file_name : str
            full path of csv file
        column_names : list
            a list of columns names to be used to create the csv file
        """
        self.file_name = file_name
        self.column_names = column_names

        if not os.path.isfile(self.file_name):
            self.file_handle = open(self.file_name, "w")
        else:
            self.file_handle = open(self.file_name, "a+")

        self.writer = csv.writer(self.file_handle)
        if not os.path.isfile(self.file_name):
            self.write_header()
            print(f"{self.file_name} created successfully with header row")
        else:
            print(f"{self.file_name} already exists, so will append results")

    def write_header(self):
        """
        writes header into csv file
        """
        self.write_row(self.column_names)

    def write_row(self, row):
        """
        writes a row into csv file
        """
        self.writer.writerow(row)

    def close(self):
        """
        close the file
        """
        self.file_handle.close()
