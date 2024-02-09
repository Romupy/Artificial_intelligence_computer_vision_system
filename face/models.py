from django.db import models
from .face_analysis_algorithm import FaceAnalysisAlgorithm


class Image(models.Model):
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return {"image_name": self.image.name}

    def analyze(self):
        face_analysis_algorithm = FaceAnalysisAlgorithm(self.image.name)
        return face_analysis_algorithm.face_detection()
