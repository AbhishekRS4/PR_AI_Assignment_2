import os
import cv2
import numpy as np

def read_image(file_image):
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

def convert_image_bgr_to_gray(image_bgr):
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

def load_image_dataset(dir_dataset, gray=True, image_suffix=("jpg", "jpeg", "png")):
    """
    Arguments
    ---------
    dir_dataset : str
        valid full directory path of the dataset with multiple classes
    gray : bool (optional)
        whether to return BGR or Grayscale images
    image_suffix : tuple (optional)
        tuple of image suffix to be considered for processing

    Returns
    -------
    ndarray of images and labels
    """
    all_image_classes = [dir for dir in os.listdir(dir_dataset) if os.path.isdir(os.path.join(dir_dataset, dir))]

    print(f"dataset directory : {dir_dataset}")
    print(f"found the following image classes in the dataset directory")
    print(all_image_classes)

    list_all_images = []
    list_all_labels = []

    for image_class in all_image_classes:
        dir_image_class = os.path.join(dir_dataset, image_class)
        list_images_class = [f_i  for f_i in os.listdir(dir_image_class) if f_i.endswith(image_suffix)]
        num_images_class = len(list_images_class)
        print(f"num images : {num_images_class}, image class: {image_class}")

        for file_name_image in list_images_class:
            file_image = os.path.join(dir_image_class, file_name_image)
            image = read_image(file_image)
            if gray:
                image = convert_image_bgr_to_gray(image)
            list_all_images.append(image)
        list_labels_image_class = [image_class] * num_images_class
        list_all_labels = list_all_labels + list_labels_image_class

    list_all_images = np.array(list_all_images)
    list_all_labels = np.array(list_all_labels)
    return list_all_images, list_all_labels
