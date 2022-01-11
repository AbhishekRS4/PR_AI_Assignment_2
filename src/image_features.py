import cv2
import numpy as np

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

class SIFTBagofVisualWordsFeatureExtractor:
    def __init__(self, train_images, clustering_method="kmeans", num_visual_words=100):
        """
        SIFTFeatureExtractor class to create Bag of Visual Words features

        Attributes
        ----------
        train_images : ndarray
            N images
        clustering_method : str (default=kmeans) (options=["kmeans", "hierarchical"])
            clustering method to be used for creating bag of visual words features
        num_visual_words : int
            number of visual words to use for creating bag of visual words features
        """
        self.train_images = train_images
        self.num_train_images = len(self.train_images)
        self.num_visual_words = num_visual_words
        self.clustering_method = clustering_method.lower()

        self.k_means_batch_size = 128
        self.clustering_algo = None

        self.sift = None
        self.train_image_descriptors = []
        self.all_train_descriptors = None

    def init_sift(self):
        self.sift = cv2.SIFT_create()

    def compute_descriptor_on_train_set(self):
        for image in self.train_images:
            kp, desc = self.sift.detectAndCompute(image, None)
            self.train_image_descriptors.append(desc)

        all_train_descriptors = self.train_image_descriptors[0]
        for desc in self.train_image_descriptors[1:]:
            all_train_descriptors = np.vstack((all_train_descriptors, desc))

        self.all_train_descriptors = all_train_descriptors.astype(np.float64)

    def init_clustering(self):
        if self.clustering_method == "kmeans":
            self.clustering_algo = MiniBatchKMeans(n_clusters=self.num_visual_words, batch_size=self.k_means_batch_size)
        elif self.clustering_method == "hierarchical":
            self.clustering_algo = AgglomerativeClustering(n_clusters=self.num_visual_words)
        else:
            print(f"wrong argument passed entered for {self.clustering_method}")

    def fit_clustering_on_train_set(self):
        self.clustering_algo.fit(self.all_train_descriptors)

    def get_train_image_histograms(self):
        train_histograms = np.zeros((self.num_train_images, self.num_visual_words), dtype=np.float32)

        for image_index in range(self.num_train_images):
            train_desc = self.train_image_descriptors[image_index].astype(np.float64)
            for desc_index in range(len(train_desc)):
                visual_word_index = self.clustering_algo.predict([train_desc[desc_index]])
                train_histograms[image_index, visual_word_index] += 1
        return train_histograms

    def get_test_image_histograms(self, test_images):
        num_test_images = len(test_images)
        test_histograms = np.zeros((num_test_images, self.num_visual_words), dtype=np.float32)

        for image_index in range(num_test_images):
            image = test_images[image_index]
            kp, test_desc = self.sift.detectAndCompute(image, None)

            for desc_index in range(len(test_desc)):
                visual_word_index = self.clustering_algo.predict([test_desc[desc_index].astype(np.float64)])
                test_histograms[image_index, visual_word_index] += 1
        return test_histograms
