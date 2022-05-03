from django.urls import path, include
from rest_framework.routers import DefaultRouter
from recommender_api import views

router = DefaultRouter() 
router.register('api', views.RecommenderApiView, basename='recommender-viewset')



urlpatterns = [
    path('', include(router.urls)),
]
