import os
import uuid

from django.conf import settings
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .modelManager import GPUModelManager

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
        modelManager = GPUModelManager.getInstance()
        modelManager.assignmentPath = filepath
        print(filepath)

        return JsonResponse({
            "id": filename,
            "url": f"{settings.MEDIA_URL}assignments/{filename}"
        })

    return JsonResponse({"error": "Invalid request method"}, status=405)

@api_view(['POST'])
def RunInference(request):
    
    detailed = request.data.get('detailedOutput', False)
    print(f"RunInference called with detailedOutput={detailed}")
    modelManager = GPUModelManager.getInstance()    
    if str(modelManager.getLoadedModel) == "None": 
        return Response({"error": "Select a model!!"}, status=status.HTTP_400_BAD_REQUEST)
    if modelManager.assignmentPath == "":
        return Response({"error": "No Assignment Uploaded!!"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        if modelManager.runInference(None, detailed=detailed):
            print("1. Inference finished. Locating file...")

            # --- STRATEGY 1: Check Base Directory (Root of repo) ---
            # This is where files usually land if you don't specify a path
            file_name = 'output.pdf'
            pdf_path = os.path.join(settings.BASE_DIR, file_name)

            # --- STRATEGY 2: Check Media Root (If configured) ---
            if not os.path.exists(pdf_path):
                print(f"File not found at {pdf_path}, checking MEDIA_ROOT...")
                if hasattr(settings, 'MEDIA_ROOT'):
                    pdf_path = os.path.join(settings.MEDIA_ROOT, file_name)

            print(f"2. Checking existence of: {pdf_path}")

            if os.path.exists(pdf_path):
                print("3. File found! Sending response.")
                # Open file in binary mode
                file_handle = open(pdf_path, 'rb')
                
                response = FileResponse(file_handle, content_type='application/pdf')
                response['Content-Disposition'] = 'attachment; filename="output.pdf"'
                return response
            else:
                print(f"CRITICAL ERROR: File not found at {pdf_path}")
                # This will tell you exactly where it looked
                return Response({"error": f"PDF generated but not found at {pdf_path}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except Exception as e:
        import traceback
        traceback.print_exc() # Print the crash details to terminal
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({"error": "PDF generation failed or model error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    
    

@csrf_exempt
def UploadKnowledgebase(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        # Ensure folder exists
        save_dir = os.path.join(settings.MEDIA_ROOT, 'knowledgebase')
        os.makedirs(save_dir, exist_ok=True)

        # Give the file a unique name
        filename = f"{uuid.uuid4()}.pdf"
        filepath = os.path.join(save_dir, filename)

        # Save file manually
        with open(filepath, "wb+") as dest:
            for chunk in file.chunks():
                dest.write(chunk)
        modelManager = GPUModelManager.getInstance()
        if not modelManager.setKnowledgebase(filepath):
            return JsonResponse({"error": "Failed to set knowledgebase"}, status=500)
        print(filepath)

        return JsonResponse({
            "id": filename,
            "url": f"{settings.MEDIA_URL}knowledgebase/{filename}"
        })

    return JsonResponse({"error": "Invalid request method"}, status=405)



# class AssignmentList(APIView):
#     def get(self, request):
#         qs = Assignment.objects.all().order_by('-uploaded_at')
#         serializer = AssignmentSerializer(qs, many=True)
#         return Response(serializer.data)


# class GetAssignmentText(APIView):
#     def get(self, request, pk):
#         assignment = get_object_or_404(Assignment, pk=pk)
#         return Response({'id': assignment.id, 'text': assignment.extracted_text})


# class UpdateAssignmentText(APIView):
#     def post(self, request, pk):
#         assignment = get_object_or_404(Assignment, pk=pk)
#         new_text = request.data.get('text', '')
#         assignment.extracted_text = new_text
#         assignment.save()
#         return Response({'status': 'ok', 'id': assignment.id})


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
