from rest_framework import serializers
from .models import Assignment, KnowledgebaseDoc


class AssignmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Assignment
        fields = '__all__'


class KnowledgebaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgebaseDoc
        fields = '__all__'
        
        
class ExampleSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=100)