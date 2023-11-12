import os
import cv2
import uuid
from statistics import median


class Face:
    def __init__(self, image_name):
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
        cascade_path = os.path.abspath(
            'face/classifiers/haarcascade_frontalface_default.xml'
        )
        face_cascade = cv2.CascadeClassifier(cascade_path)
        img = cv2.imread(self.image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_extension = os.path.splitext(self.image_name)
        gray_image_name = "images/" + str(uuid.uuid4()) + img_extension.lower()
        cv2.imwrite(gray_image_name, gray)
        black_and_white = cv2.equalizeHist(gray)
        faces_detected = face_cascade.detectMultiScale(
            black_and_white,
            scaleFactor=1.45, minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        sorted_faces = sorted(faces_detected, key=lambda x: x[2] * x[3], reverse=True)
        faces = self.__remove_double_detection(sorted_faces)
        skin_brightness = []
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
