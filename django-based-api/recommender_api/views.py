from rest_framework.response import Response
from rest_framework import status, viewsets, mixins
from rest_framework.decorators import action
from pydantic import BaseModel, ValidationError
from pyngo import drf_error_details
from loguru import logger

from recommender_api import serializers, models
from recommender_api import schemas
from recommender_api import __version__ as api_version
from recommender_api.config import settings
from recommender_model import recommend, __version__ as model_version


class RecommenderApiView(viewsets.GenericViewSet, mixins.RetrieveModelMixin):
    serializer_class = serializers.UserSerialzer
    
    @action(methods=['get'], detail=False)
    def health(self, request, format=None):
        """Health end point return model and api version"""
        
        health = schemas.Health(
            name=settings.PROJECT_NAME, api_version=api_version, model_version=
            model_version)
        
        return Response(health.dict())

    @action(methods=['post'], detail=False)
    def recommend(self, request):
        _input_data = request.data['inputs']
        results = recommend.make_recommendation(input_data=_input_data)
        
        # model input data validator
        try:
            schemas.RecommendationResults.parse_obj(results)
            logger.info(f"Recommended books: {results.get('recommendations')}")
        except ValidationError as e:
            return Response({'errors': drf_error_details(e)}, 
                status=status.HTTP_400_BAD_REQUEST)

        # model output data validator

        if results["errors"] is not None:
            logger.warning(f"Translation validation error: {results.get('errors')}")
            return Response(
                {"errors": results.get('errors')}, 
                status=status.HTTP_400_BAD_REQUEST)
        else:  
            return Response(results)
