from django.db import models


class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    faces = models.IntegerField(default=0)

    def __str__(self):
        return self.image.name
