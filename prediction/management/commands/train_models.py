from django.core.management.base import BaseCommand
from prediction.ml_models import kaggle_heart_predictor
import os

class Command(BaseCommand):
    help = 'Train the heart disease prediction models'

    def handle(self, *args, **options):
        self.stdout.write("🚀 Starting model training...")
        
        # Create necessary directories
        os.makedirs('ml_models/trained_models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Train the models
        try:
            results = kaggle_heart_predictor.train_models_with_kaggle_data()
            if results:
                self.stdout.write(
                    self.style.SUCCESS('✅ Successfully trained and saved all models!')
                )
                # Print model accuracies
                self.stdout.write("\nModel Training Results:")
                self.stdout.write("-" * 50)
                for name, metrics in results.items():
                    self.stdout.write(
                        f"{name}: Accuracy={metrics.get('accuracy', 0):.4f}, "
                        f"AUC={metrics.get('auc_score', 0):.4f}"
                    )
            else:
                self.stdout.write(
                    self.style.ERROR('❌ Model training failed. Check the logs for details.')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error during model training: {str(e)}')
            )
