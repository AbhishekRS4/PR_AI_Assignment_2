import os
import numpy as np

from logger_utils import CSVWriter
from learning_utils import compute_silhouette_score, get_dbscan_clustering_algo, get_tfidf_transformer, get_learning_pipeline

def do_image_clustering_experiments(start_num_visual_words=5, end_num_visual_words=205, dir_sift_data="sift_features"):
    file_cluster_results = "image_clustering_dbscan.csv"
    col_names = ["clustering_algo", "features", "num_visual_words", "eps", "num_clusters", "silhouette_score"]
    csv_writer = CSVWriter(os.path.join(dir_sift_data, file_cluster_results), col_names)

    for num_visual_words in range(start_num_visual_words, end_num_visual_words, 5):
        train_x = np.load(os.path.join(dir_sift_data, f"train_sift_{num_visual_words}.npy"))
        test_x = np.load(os.path.join(dir_sift_data, f"test_sift_{num_visual_words}.npy"))

        # for clustering, concatenate train and test data which was split specifically for classification task
        all_train_data = np.concatenate((train_x, test_x), axis=0)
        print(all_train_data.shape)

        preprocess = "tf_idf"
        preprocessor = get_tfidf_transformer()
        for eps in np.arange(0.3, 0.45, 0.005):
            print(f"Num visual words : {num_visual_words}")
            try:
                dbscan_clustering_algo = get_dbscan_clustering_algo(eps=eps)
                pipeline = [(preprocess, preprocessor), ("dbscan", dbscan_clustering_algo)]
                clustering_pipeline = get_learning_pipeline(pipeline)
                clustering_pipeline.fit(all_train_data)

                cluster_labels = clustering_pipeline["dbscan"].labels_
                sil_score = compute_silhouette_score(all_train_data, cluster_labels)
                num_clusters = len(np.unique(cluster_labels))
                #print(cluster_labels)
                print(f"Num clusters detected : {num_clusters}")
                print(f"silhouette_score : {sil_score:.6f}\n")
                row_ = ["dbscan", "sift", num_visual_words, round(eps, 4), num_clusters, round(sil_score, 6)]
                csv_writer.write_row(row_)
            except:
                print("All samples belong to same cluster, so skipping...")
    csv_writer.close()
    return
