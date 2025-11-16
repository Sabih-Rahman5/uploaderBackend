import os
import uuid

from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .modelManager import GPUModelManager
from .models import Assignment, KnowledgebaseDoc
from .serializers import AssignmentSerializer, KnowledgebaseSerializer


@csrf_exempt
def UploadAssignment(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        # Ensure folder exists
        save_dir = os.path.join(settings.MEDIA_ROOT, 'assignments')
        os.makedirs(save_dir, exist_ok=True)

        # Give the file a unique name
        filename = f"{uuid.uuid4()}.pdf"
        filepath = os.path.join(save_dir, filename)

        # Save file manually
        with open(filepath, "wb+") as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        return JsonResponse({
            "id": filename,
            "url": f"{settings.MEDIA_URL}assignments/{filename}"
        })

    return JsonResponse({"error": "Invalid request method"}, status=405)


class UploadKnowledgebase(APIView):
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        doc = KnowledgebaseDoc.objects.create(file=file)
        serializer = KnowledgebaseSerializer(doc)
        return Response(serializer.data)


class AssignmentList(APIView):
    def get(self, request):
        qs = Assignment.objects.all().order_by('-uploaded_at')
        serializer = AssignmentSerializer(qs, many=True)
        return Response(serializer.data)


class GetAssignmentText(APIView):
    def get(self, request, pk):
        assignment = get_object_or_404(Assignment, pk=pk)
        return Response({'id': assignment.id, 'text': assignment.extracted_text})


class UpdateAssignmentText(APIView):
    def post(self, request, pk):
        assignment = get_object_or_404(Assignment, pk=pk)
        new_text = request.data.get('text', '')
        assignment.extracted_text = new_text
        assignment.save()
        return Response({'status': 'ok', 'id': assignment.id})


class LoadModel(APIView):
    def post(self, request):
        model_name = request.data.get('selected_model', None)
        if model_name:
            print(f"RunModel called with: {model_name}")
            modelManager = GPUModelManager.getInstance()
            try:
                modelManager.loadModel(model_name)
                return Response({"message": "Model loaded successfully!"}, status=status.HTTP_200_OK)
            except ValueError as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"error": "No model selected"}, status=status.HTTP_400_BAD_REQUEST)
    
class ExampleView(APIView):
    def get(self, request):
        data = {'message': 'Hello from Django!'}
        return Response(data)
    
    
class ModelStatus(APIView):
    def get(self, request):
        modelManager = GPUModelManager.getInstance()
        return Response({
            "state": modelManager.getState(),
            "model_name": modelManager.getLoadedModel()
        })
