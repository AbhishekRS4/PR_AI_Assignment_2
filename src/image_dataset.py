import os
import cv2
import numpy as np

class ImageDatasetLoader:
    """
    ImageDatasetLoader class to load image datasets

    Attributes
    ----------
    dir_dataset : str
        valid full directory path of the dataset with multiple categories
    is_gray_scale : bool
        whether to return BGR or Grayscale images
    image_file_suffix : tuple
        tuple of image suffix to be considered for processing
    num_categories : int
        num categories in the image dataset
    images : ndarray
        all the image features
    labels : ndarray
        all the image labels

    Methods
    -------
    load_dataset() : loads all the images and labels as numpy arrays
    """

    def __init__(self, dir_dataset, is_gray_scale=True, image_file_suffix=("jpg", "jpeg", "png")):
        """
        Arguments
        ---------
        dir_dataset : str
            valid full directory path of the dataset with multiple categories
        is_gray_scale : bool (optional)
            whether to return BGR or Grayscale images
        image_file_suffix : tuple (optional)
            tuple of image suffix to be considered for processing
        """

        self.dir_dataset = dir_dataset
        self.is_gray_scale = is_gray_scale
        self.image_file_suffix = image_file_suffix
        self.num_categories = None
        self.images = None
        self.labels = None

    def _read_image(self, file_image):
        """
        Arguments
        ---------
        file_image : str
            valid full image path

        Returns
        -------
        image_bgr : ndarray
            BGR image
        """
        image_bgr = cv2.imread(file_image)
        return image_bgr

    def _convert_image_bgr_to_gray(self, image_bgr):
        """
        Arguments
        ---------
        image_bgr : ndarray
            BGR image

        Returns
        -------
        image_gray : ndarray
            Grayscale image
        """
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return image_gray

    def load_dataset(self):
        """
        Method
        ------
        loads ndarray of images and labels to attributes images and labels
        """
        all_image_categories = [dir for dir in os.listdir(self.dir_dataset) if os.path.isdir(os.path.join(self.dir_dataset, dir))]

        print(f"dataset directory : {self.dir_dataset}")
        print(f"found the following image categories in the dataset directory")
        print(all_image_categories)
        self.num_categories = len(all_image_categories)

        list_all_images = []
        list_all_labels = []

        for image_category in all_image_categories:
            dir_image_category = os.path.join(self.dir_dataset, image_category)
            list_images_category = [f_i  for f_i in os.listdir(dir_image_category) if f_i.endswith(self.image_file_suffix)]
            num_images_category = len(list_images_category)
            print(f"num images : {num_images_category}, image category: {image_category}")

            for file_name_image in list_images_category:
                file_image = os.path.join(dir_image_category, file_name_image)
                image = self._read_image(file_image)
                if self.is_gray_scale:
                    image = self._convert_image_bgr_to_gray(image)
                list_all_images.append(image)
            list_labels_image_category = [image_category] * num_images_category
            list_all_labels = list_all_labels + list_labels_image_category

        self.images = np.array(list_all_images)
        self.labels = np.array(list_all_labels)
