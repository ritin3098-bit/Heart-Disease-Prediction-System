from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class HeartDiseaseData(models.Model):
    age=models.IntegerField()
    sex=models.IntegerField()
    cp=models.IntegerField()
    trestbps=models.IntegerField()
    chol=models.IntegerField()
    fbs=models.IntegerField()
    restecg=models.IntegerField()
    thalach=models.IntegerField()
    exang=models.IntegerField()
    oldpeak=models.IntegerField()
    slope=models.IntegerField()
    ca=models.IntegerField()
    thal=models.IntegerField()
    target=models.IntegerField()
    created_at=models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Patient {self.id} - Age {self.age}"


class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    input_data = models.JSONField()
    result = models.BooleanField()
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.user.username}'s prediction on {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def details(self):
        """Return input data as a formatted dictionary"""
        return self.input_data
    
    @property
    def result_display(self):
        """Return result as a human-readable string"""
        return 'Positive' if self.result else 'Negative'

# Create your models here.
