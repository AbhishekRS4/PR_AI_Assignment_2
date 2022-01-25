import numpy as np
import matplotlib.pyplot as plt

def plot_compare_scores(arr_num_visual_words, arr_scores_1, arr_scores_2, str_title, what_comparison="cv_test", str_label_plot_1=None, str_label_plot_2=None):
    if what_comparison == "cv_test":
        str_label_plot_1 = "best accuracy score - cross validation"
        str_label_plot_2 = "best accuracy score - test"
    elif what_comparison == "preprocess":
        str_label_plot_1 = str_label_plot_1
        str_label_plot_2 = str_label_plot_2
    else:
        print(f"Unknown option : {what_comparison}, so exiting")

    str_x_label = "Number of visual words"
    str_y_label = "Accuracy Score"

    fig = plt.figure(figsize=(12, 9))
    plt.plot(arr_num_visual_words, arr_scores_1, label=str_label_plot_1)
    plt.plot(arr_num_visual_words, arr_scores_2, label=str_label_plot_2)
    plt.title(str_title)
    plt.xlabel(str_x_label)
    plt.ylabel(str_y_label)
    plt.legend()
    fig.show()
    return
