import os
import cv2


class Face:
    @staticmethod
    def face_detection(image):
        """
        Method that detects faces in an image

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
        img = cv2.imread(image.name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_and_white = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            black_and_white,
            scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(image.name, img)
        return len(faces)
