from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Assignment, KnowledgebaseDoc
from .serializers import AssignmentSerializer, KnowledgebaseSerializer
import PyPDF2
from django.shortcuts import get_object_or_404
from .modelManager import GPUModelManager
import time


class UploadAssignment(APIView):
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        assignment = Assignment.objects.create(file=file)
        # extract text
        try:
            reader = PyPDF2.PdfReader(assignment.file)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or '')
                assignment.extracted_text = "\n".join(text)
                assignment.save()
        except Exception as e:
            print('PDF extract error:', e)
        serializer = AssignmentSerializer(assignment)
        return Response(serializer.data)


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


class RunModel(APIView):
    def post(self, request):
        model_name = request.data.get('selected_model', None)
        if model_name:
            print(f"RunModel called with: {model_name}")
            # Simulate a long-running task
            for i in range(10):  # Simulate a task with 10 steps
                time.sleep(1)  # Simulate model loading time
                # Normally, you'd load the model here, e.g. modelManager.loadModel(model_name)
                print(f"Loading {model_name}... {i*10}% complete")

            return Response({"message": "Item printed successfully"}, status=status.HTTP_200_OK)
            
            return Response({"message": "Item printed successfully"}, status=status.HTTP_200_OK)
    # In real life you'd call the selected LLM here.
        return Response({"error": "No item selected"}, status=status.HTTP_400_BAD_REQUEST)
    
class ExampleView(APIView):
    def get(self, request):
        data = {'message': 'Hello from Django!'}
        return Response(data)