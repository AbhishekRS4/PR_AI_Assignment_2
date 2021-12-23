import cv2
import numpy as np

from sklearn.cluster import MiniBatchKMeans

class SIFTFeatureExtractor:
    def __init__(self, train_images, num_visual_words=100, k_means_batch_size=128):
        self.train_images = train_images
        self.num_train_images = len(self.train_images)
        self.num_visual_words = num_visual_words

        self.k_means_batch_size = k_means_batch_size
        self.k_means = None

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

    def init_k_means(self):
        self.k_means = MiniBatchKMeans(n_clusters=self.num_visual_words, batch_size=self.k_means_batch_size)

    def fit_k_means_on_train_set(self):
        self.k_means.fit(self.all_train_descriptors)

    def get_train_image_histograms(self):
        train_histograms = np.zeros((self.num_train_images, self.num_visual_words), dtype=np.float32)

        for image_index in range(self.num_train_images):
            train_desc = self.train_image_descriptors[image_index].astype(np.float64)
            for desc_index in range(len(train_desc)):
                visual_word_index = self.k_means.predict([train_desc[desc_index]])
                train_histograms[image_index, visual_word_index] += 1
        return train_histograms

    def get_test_image_histograms(self, test_images):
        num_test_images = len(test_images)
        test_histograms = np.zeros((num_test_images, self.num_visual_words), dtype=np.float32)

        for image_index in range(num_test_images):
            image = test_images[image_index]
            kp, test_desc = self.sift.detectAndCompute(image, None)

            for desc_index in range(len(test_desc)):
                visual_word_index = self.k_means.predict([test_desc[desc_index].astype(np.float64)])
                test_histograms[image_index, visual_word_index] += 1
        return test_histograms
