from django.db import models

# Create your models here.

class Assignment(models.Model):
    file = models.FileField(upload_to='assignments/')
    extracted_text = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)


def __str__(self):
    return f"Assignment {self.id} - {self.file.name}"


class KnowledgebaseDoc(models.Model):
    file = models.FileField(upload_to='knowledgebase/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


def __str__(self):
    return f"Knowledgebase {self.id} - {self.file.name}"