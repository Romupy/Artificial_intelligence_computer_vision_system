from django.db import models
from .face import Face


class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    faces = models.IntegerField(default=0)

    def __str__(self):
        return {"image_name": self.image.name}

    def analyze(self):
        face = Face(self.image.name)
        return face.face_detection()
