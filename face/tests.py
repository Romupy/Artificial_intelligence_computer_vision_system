import os

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase


class FaceViewTests(APITestCase):
    def test_image_with_1_face(self):
        url = reverse('face_analysis')
        image_data = {
            'image': open(
                os.path.abspath('face/test_images/image_with_1_face.jpg'), 'rb'
            ),
        }
        result = self.client.post(url, data=image_data, format='multipart')
        content = result.data
        self.assertEqual(result.status_code, status.HTTP_201_CREATED)
        self.assertGreater(len(content['image']), 0)
        self.assertEqual(
            content['analysis_results'],
            {"Number of faces detected": 1}
        )

    def test_image_with_2_faces(self):
        url = reverse('face_analysis')
        image_data = {
            'image': open(
                os.path.abspath('face/test_images/image_with_2_faces.jpg'),
                'rb'
            )
        }
        result = self.client.post(url, data=image_data, format='multipart')
        content = result.data
        self.assertEqual(result.status_code, status.HTTP_201_CREATED)
        self.assertGreater(len(content['image']), 0)
        self.assertEqual(
            content['analysis_results'],
            {"Number of faces detected": 2}
        )

    def test_image_without_face(self):
        url = reverse('face_analysis')
        image_data = {
            'image': open(
                os.path.abspath('face/test_images/image_without_face.jpg'),
                'rb'
            )
        }
        result = self.client.post(url, data=image_data, format='multipart')
        content = result.data
        self.assertEqual(result.status_code, status.HTTP_201_CREATED)
        self.assertGreater(len(content['image']), 0)
        self.assertEqual(
            content['analysis_results'],
            {"Number of faces detected": 0}
        )
