from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .serializers import ImageSerializer


@api_view(['POST'])
def face_analysis(request):
    """
    View that enables image reception and recording

    Keyword arguments:
    request -- (django.http.HttpRequest) The HttpRequest object containing all
    information about HTTP request.

    Returns:
    django.http.HttpResponse -- An HttpResponse object representing the view
    response
    """
    serializer = ImageSerializer(data=request.data)
    if serializer.is_valid():
        image = serializer.save()
        analysis_results = image.analyze()
        data = serializer.data
        data['analysis_results'] = analysis_results
        return Response(data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
