import os
import cv2
import uuid
from statistics import median


class Face:
    def __init__(self, image_name):
        self.image_name = image_name

    def face_detection(self):
        """
        Method that detects faces in an image

        This method saves the image, the edge of which represents the detected
        face, and returns features relating to this image.

        Keyword arguments:
        image -- (django.core.files.uploadedfile.InMemoryUploadedFile) The
        image to be analyzed

        Returns:
        int -- The number of faces detected
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
        faces = face_cascade.detectMultiScale(
            black_and_white,
            scaleFactor=1.45, minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
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
        A method that detects facial skin brightness.

        This method analyzes the brightness of facial skin in the specified
        region of the grayscale image.

        Keyword arguments:
        coordinates -- Dictionary containing the 'x' and 'y' coordinates of
        the region.
        sizes -- Dictionary containing the width ('w') and height ('h') of
        the region.
        gray_image_name -- The file path of the grayscale image to be analyzed.

        Returns:
        float -- The brightness of the facial skin in the specified region,
        rounded to two decimal places.
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
        if 0 < result < 0.42:
            skin_info = "dark skin"
        elif 0.42 <= result <= 0.60:
            skin_info = "matte skin"
        elif 0.60 < result:
            skin_info = "light skin"
        else:
            skin_info = "no skin"
        return {"skin_brightness": result, "skin_info": skin_info}
