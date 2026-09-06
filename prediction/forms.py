from django import forms
from django.core.validators import FileExtensionValidator

class HeartDiseaseForm(forms.Form):
    age = forms.IntegerField(
        label='Age',
        min_value=1,
        max_value=120,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter age'})
    )
    sex = forms.ChoiceField(
        label='Sex',
        choices=[(1, 'Male'), (0, 'Female')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    cp = forms.ChoiceField(
        label='Chest Pain Type',
        choices=[
            (0, 'Typical Angina'),
            (1, 'Atypical Angina'),
            (2, 'Non-Anginal Pain'),
            (3, 'Asymptomatic'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    trestbps = forms.IntegerField(
        label='Resting Blood Pressure (mm Hg)',
        min_value=80,
        max_value=220,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 120'})
    )
    chol = forms.IntegerField(
        label='Serum Cholesterol (mg/dl)',
        min_value=100,
        max_value=600,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 200'})
    )
    fbs = forms.ChoiceField(
        label='Fasting Blood Sugar >120 mg/dl',
        choices=[(1, 'Yes'), (0, 'No')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    restecg = forms.ChoiceField(
        label='Resting ECG Results',
        choices=[(0, 'Normal'), (1, 'ST-T wave Abnormality'), (2, 'Left Ventricular Hypertrophy')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    thalach = forms.IntegerField(
        label='Maximum Heart Rate Achieved',
        min_value=60,
        max_value=220,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 150'})
    )
    exang = forms.ChoiceField(
        label='Exercise Induced Angina',
        choices=[(1, 'Yes'), (0, 'No')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    oldpeak = forms.FloatField(
        label='ST Depression (oldpeak)',
        min_value=0.0,
        max_value=10.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'e.g., 2.3'})
    )
    slope = forms.ChoiceField(
        label='Slope of Peak Exercise ST Segment',
        choices=[(0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    ca = forms.ChoiceField(
        label='Number of Major Vessels (0-3)',
        choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    thal = forms.ChoiceField(
        label='Thalassemia',
        choices=[(3, 'Normal'), (6, 'Fixed Defect'), (7, 'Reversible Defect')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )


class DatasetUploadForm(forms.Form):
    """Form for uploading heart disease dataset"""
    dataset = forms.FileField(
        label='Heart Disease Dataset',
        help_text='Upload a CSV file containing heart disease data',
        validators=[
            FileExtensionValidator(
                allowed_extensions=['csv'],
                message='Only CSV files are allowed'
            )
        ],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv',
            'id': 'dataset-upload'
        })
    )
    
    def clean_dataset(self):
        """Validate the uploaded dataset"""
        dataset = self.cleaned_data.get('dataset')
        
        if not dataset:
            raise forms.ValidationError('No file was uploaded')
            
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if dataset.size > max_size:
            raise forms.ValidationError('File size should not exceed 10MB')
            
        # Read and validate the CSV file
        try:
            import pandas as pd
            from io import TextIOWrapper
            
            # Handle both in-memory and temporary file uploads
            if hasattr(dataset, 'temporary_file_path'):
                df = pd.read_csv(dataset.temporary_file_path())
            else:
                # For in-memory files
                dataset.seek(0)
                df = pd.read_csv(TextIOWrapper(dataset.file, encoding=dataset.charset))
            
            # Check required columns
            required_columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]
            
            missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
            
            if missing_columns:
                raise forms.ValidationError(
                    f'Missing required columns: {", ".join(missing_columns)}'
                )
                
            # Store the dataframe in cleaned_data for later use
            self.cleaned_data['dataframe'] = df
            
        except pd.errors.EmptyDataError:
            raise forms.ValidationError('The uploaded file is empty')
        except pd.errors.ParserError:
            raise forms.ValidationError('Invalid CSV file format')
        except Exception as e:
            raise forms.ValidationError(f'Error reading the file: {str(e)}')
            
        return dataset
    
    