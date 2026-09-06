import sys
import os
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'heart_disease_prediction.settings')
django.setup()

from prediction.ml_models import heart_predictor

if __name__ == '_main_':
    print("Starting model training...")
    print("=" * 50)
    
    results = heart_predictor.train_models()
    
    print("\nTraining completed!")
    print("=" * 50)
    print("\nModel Accuracies:")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']*100:.2f}%")
    
    print("\nModels saved successfully!")