import cv2
import numpy as np

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

class SIFTBagofVisualWordsFeatureExtractor:
    """
    SIFTFeatureExtractor class to create Bag of Visual Words features
    """
    def __init__(self, train_images, clustering_method="kmeans", num_visual_words=100):
        """
        ----------
        Attributes
        ----------
        train_images : ndarray
            N images
        clustering_method : str (default=kmeans) (options=["kmeans"])
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

class FastFourierTransformFeatureExtractor:
    """
    FastFourierTransformFeatureExtractor class to create image features using FFT
    """
    def __init__(self, radius=20, target_dim=(50, 50), magnitude_spectrum_weight=20):
        """
        ----------
        Attributes
        ----------
        radius : int
            radius to control the amount of high frequency to be allowed [in other words to block low frequency]
        target_dim : tuple of ints
            dimension of output image features
        magnitude_spectrum_weight : int
            weight for magnitude spectrum function
        """
        self.radius = radius
        self.target_dim = target_dim
        self.magnitude_spectrum_weight = magnitude_spectrum_weight
        self.img_magnitude_spectrum = None
        self.feats_magnitude_spectrum = None

    def get_mask_high_pass_filter(self, img_shape):
        """
        ----------
        Attributes
        ----------
        img : tuple of ints
            image shape (H, W)
        """
        H, W = img_shape
        mask = np.ones((H, W), np.uint8)

        center_H, center_W = int(H / 2), int(W / 2)
        center = [center_H, center_W]

        x, y = np.ogrid[:H, :W]
        # center area in fourier space is the low frequency
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= self.radius ** 2
        # block low frequency i.e. center area in the fourier space
        mask[mask_area] = 0
        # inverse of this mask gives the mask for low pass filter
        return mask

    def get_features_using_fft(self, img):
        """
        ----------
        Attributes
        ----------
        img : ndarray
            grayscale image
        """
        # convert image from image domain to frequency domain
        freq_domain = np.fft.fft2(img)

        # Apply shift to image in freq domain so that origin is shifted from top left to center of the image
        freq_domain_shifted = np.fft.fftshift(freq_domain)
        self.img_magnitude_spectrum = self.magnitude_spectrum_weight * np.log(np.abs(freq_domain_shifted))

        # Apply mask
        mask = self.get_mask_high_pass_filter(img.shape)
        freq_domain_shifted_masked = freq_domain_shifted * mask

        # Compute magnitude spectrum of the filtered frequency domain image
        self.feats_magnitude_spectrum = self.magnitude_spectrum_weight * np.log(np.abs(freq_domain_shifted_masked) + 1)

        # Apply inverse shift so that origin is shifted from center to top left
        freq_domain_inv_shifted_masked = np.fft.ifftshift(freq_domain_shifted_masked)

        # Apply inverse FFT to convert from frequency domain to image domain
        img_feats = np.fft.ifft2(freq_domain_inv_shifted_masked, self.target_dim)
        img_feats = np.abs(img_feats)
        return img_feats
