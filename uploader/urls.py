from django.urls import path
from .views import UploadAssignment, UploadKnowledgebase, AssignmentList, GetAssignmentText, UpdateAssignmentText, LoadModel, ExampleView, ModelStatus


urlpatterns = [
path('upload-assignment/', UploadAssignment.as_view()),
path('upload-knowledgebase/', UploadKnowledgebase.as_view()),
path('assignments/', AssignmentList.as_view()),
path('assignment/<int:pk>/text/', GetAssignmentText.as_view()),
path('assignment/<int:pk>/update-text/', UpdateAssignmentText.as_view()),
path('run-model/', LoadModel.as_view()),
path('example/', ExampleView.as_view(), name='example'),
path("model-status/", ModelStatus.as_view()),

]