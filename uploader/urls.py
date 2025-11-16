from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import UploadAssignment, UploadKnowledgebase, LoadModel, ExampleView, ModelStatus, RunInference


urlpatterns = [
path('run-inference', RunInference),
path('upload-assignment/', UploadAssignment),
path('upload-knowledgebase/', UploadKnowledgebase),
# path('assignments/', AssignmentList.as_view()),
# path('assignment/<int:pk>/text/', GetAssignmentText.as_view()),
# path('assignment/<int:pk>/update-text/', UpdateAssignmentText.as_view()),
path('run-model/', LoadModel.as_view()),
path('example/', ExampleView.as_view(), name='example'),
path("model-status/", ModelStatus.as_view()),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)