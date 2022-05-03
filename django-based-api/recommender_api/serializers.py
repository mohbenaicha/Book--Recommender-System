from rest_framework import serializers


class UserSerialzer(serializers.Serializer):
    inputs = serializers.ListSerializer(child=serializers.CharField())
