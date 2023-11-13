import os
import time

import cv2
import uuid
from statistics import median

import numpy as np


class Face:
    """
    Class for facial analysis.

    This class provides methods for detecting faces in an image, analyzing
    facial skin brightness, and determining the type of facial skin.

    Attributes:
    image_name (str): The file path of the image to be analyzed.

    Methods:
    face_detection(): Detects faces in an image and analyzes facial skin.
    __skin_brightness_detection(coordinates, sizes, gray_image_name):
        Detects facial skin brightness.
    __skin_type_detection(result): Determines the type of facial skin.
    __remove_double_detection(rectangles): Removes rectangles containing others.
    """
    def __init__(self, image_name):
        """
        Initializes a Face object.

        Keyword arguments::
        image_name (str): The file path of the image to be analyzed.
        """
        self.image_name = image_name

    def face_detection(self):
        """
        Method that detects faces in an image.

        This method utilizes a Haar Cascade classifier to detect faces in the
        provided image.
        Detected faces are processed to remove rectangles containing others,
        and the remaining rectangles are analyzed for facial skin brightness.

        Returns:
        dict -- Dictionary containing the number of faces detected and
        information about the facial skin.
        Example: {'Number_of_faces_detected': 2, 'facial_skin': [...]}
        """
        gray = None
        cascade_paths = [
            os.path.abspath('face/classifiers/haarcascade_frontalface_default.xml'),
            os.path.abspath('face/classifiers/haarcascade_frontalface_alt_tree.xml'),
            os.path.abspath('face/classifiers/haarcascade_frontalface_alt2.xml'),
            os.path.abspath('face/classifiers/haarcascade_profileface.xml'),
        ]


        for rotation in [0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10, 20, -20, 30, -30, 40, -40, 50, -50, 60, -60, 70, -70, 80, -80, 180, -180, 175, -175, 170, -170, 160, -160, 150, -150, 140, -140, 130, -130, 120, -120, 110, -110, 100, -100, 0]:
            faces_detected_results = []
            img = cv2.imread(self.image_name)
            img = rotate_image(img, rotation)
            for cascade_path in cascade_paths:
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                black_and_white = cv2.equalizeHist(gray)
                height, width, channels = img.shape
                min_size_height = int(height / 13)
                min_size_width = int(width / 13)
                size = min([min_size_height, min_size_width])
                faces_detected = face_cascade.detectMultiScale(
                    black_and_white,
                    scaleFactor=1.45, minNeighbors=5,
                    minSize=(size, size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces_detected) > 0:
                    faces_detected_results.append(np.array(faces_detected))

                faces_detected = face_cascade.detectMultiScale(
                    black_and_white,
                    scaleFactor=1.48, minNeighbors=6,
                    minSize=(size, size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces_detected) > 0:
                    faces_detected_results.append(np.array(faces_detected))
                faces_detected = face_cascade.detectMultiScale(
                    black_and_white,
                    scaleFactor=1.55, minNeighbors=6,
                    minSize=(size, size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces_detected) > 0:
                    faces_detected_results.append(np.array(faces_detected))
                faces_detected = face_cascade.detectMultiScale(
                    black_and_white,
                    scaleFactor=1.60, minNeighbors=6,
                    minSize=(size, size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces_detected) > 0:
                    faces_detected_results.append(np.array(faces_detected))
                faces_detected = face_cascade.detectMultiScale(
                    black_and_white,
                    scaleFactor=1.70, minNeighbors=6,
                    minSize=(size, size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces_detected) > 0:
                    faces_detected_results.append(np.array(faces_detected))

            if len(faces_detected_results) > 0:
                break
        first_iteration = True
        results = []
        for faces_detected_result in faces_detected_results:
            if first_iteration:
                results = faces_detected_result
                first_iteration = False
            else:
                if len(faces_detected_result) > 0:
                    results = np.concatenate((results, faces_detected_result))
        sorted_faces = sorted(results, key=lambda x: x[2] * x[3], reverse=True)
        faces = self.__remove_double_detection(sorted_faces)
        faces = self.__remove_containing_rectangles(faces)

        skin_brightness = []
        _, img_extension = os.path.splitext(self.image_name)
        gray_image_name = "images/" + str(uuid.uuid4()) + img_extension.lower()
        if gray is not None:
            cv2.imwrite(gray_image_name, gray)
        for x, y, w, h in faces:
            coordinates = {'x': x, 'y': y}
            sizes = {'w': w, 'h': h}
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            skin_brightness_result = self.__skin_brightness_detection(
                coordinates, sizes, self.image_name
            )
            skin_brightness.append(skin_brightness_result)
        cv2.imwrite(self.image_name, img)
        return {"Number_of_faces_detected": len(faces),
                "facial_skin": skin_brightness}

    @staticmethod
    def __skin_brightness_detection(coordinates, sizes, gray_image_name):
        """
       Method that detects facial skin brightness.

       This method analyzes the brightness of facial skin in the specified
       region of the grayscale image.

       Keyword arguments:
       coordinates (dict): Dictionary containing the 'x' and 'y' coordinates of
       the region.
       sizes (dict): Dictionary containing the width ('w') and height ('h') of
       the region.
       gray_image_name (str): The file path of the grayscale image to be
       analyzed.

       Returns:
        dict: A dictionary containing the brightness of the facial skin in the
        specified region rounded to two decimal places and skin information.
        Example: {'skin_brightness': 0.75, 'skin_info': 'light skin'}
       """
        img = cv2.imread(gray_image_name)
        distance = int(sizes['h'] / 2)
        ordinate = [
            coordinates['y'] + distance,
            coordinates['y'] + distance + int((distance/6)),
            coordinates['y'] + distance + int((distance/3))
        ]
        brightness_data = []
        for y in ordinate:
            sampling_distance = int((sizes['w'] / 1.5) / 3)
            x = coordinates['x'] + int(sizes['w'] / 3)
            coords_pixels = [
                {'x': x, 'y': y},
                {'x': x + sampling_distance, 'y': y},
                {'x': x + int(sampling_distance * 2), 'y': y}
            ]
            for coords_pixel in coords_pixels:
                brightness_data.append(
                    int(img[coords_pixel['y'], coords_pixel['x']][0])
                )
        result = round(((100 / 255) * median(brightness_data)) / 100, 2)
        skin_info = Face.__skin_type_detection(result)
        return {"skin_brightness": result, "skin_info": skin_info}

    @staticmethod
    def __skin_type_detection(result):
        """
        Method that detects the type of facial skin based on brightness result.

        Keyword arguments:
        result (float): The brightness result of the facial skin in the
        specified region.

        Returns:
        str: String indicating the detected type of facial skin.
        Possible values: 'dark skin', 'matte skin', 'light skin', 'no skin'.
        """
        if 0 < result < 0.42:
            return "dark skin"
        elif 0.42 <= result <= 0.60:
            return "matte skin"
        elif 0.60 < result:
            return "light skin"
        return "no skin"

    @staticmethod
    def __remove_double_detection(rectangles):
        """
        Method that removes double detection, the method that removes
        rectangles containing others from the list.

        This method takes a list of rectangles represented as tuples
        (x, y, w, h).
        It removes rectangles that contain other rectangles.

        Keyword arguments:
        rectangles -- List of rectangles to be processed.
                      Represented as tuples (x, y, w, h).

        Returns:
        List -- List of rectangles that do not contain any other rectangles.
        """
        retained_rectangles = []
        for i in range(len(rectangles)):
            x_i, y_i, w_i, h_i = rectangles[i]
            is_contained = False
            for j in range(len(rectangles)):
                if i != j:
                    x_j, y_j, w_j, h_j = rectangles[j]
                    if (x_i <= x_j and y_i <= y_j and x_i + w_i >= x_j + w_j
                            and y_i + h_i >= y_j + h_j):
                        is_contained = True
                        break

            if not is_contained:
                retained_rectangles.append((x_i, y_i, w_i, h_i))
        return retained_rectangles

    @staticmethod
    def __remove_containing_rectangles(rectangles):
        """
        Method that removes rectangles containing others from the list.

        This method takes a list of rectangles represented as tuples (x, y, w, h).
        It removes rectangles that contain other rectangles, even if there is partial overlap.

        Keyword arguments:
        rectangles -- List of rectangles to be processed.
                      Represented as tuples (x, y, w, h).

        Returns:
        List -- List of rectangles that do not contain any other rectangles.
        """
        retained_rectangles = []

        for i in range(len(rectangles)):
            x_i, y_i, w_i, h_i = rectangles[i]
            is_contained_by_other = any(
                (
                    x_i < x_r + w_r
                    and x_i + w_i > x_r
                    and y_i < y_r + h_r
                    and y_i + h_i > y_r
                )
                for (x_r, y_r, w_r, h_r) in retained_rectangles
            )

            if not is_contained_by_other:
                retained_rectangles.append((x_i, y_i, w_i, h_i))

        return retained_rectangles


def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image

