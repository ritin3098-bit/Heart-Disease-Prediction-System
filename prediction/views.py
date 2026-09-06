# views.py
import os
import logging
from pathlib import Path

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.http import JsonResponse
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, update_session_auth_hash,logout
from django.views.decorators.http import require_http_methods,require_POST
from django.conf import settings
from django.utils import timezone
from django.core.cache import cache

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

from .forms import HeartDiseaseForm
from .ml_models import kaggle_heart_predictor
from .models import PredictionHistory

logger = logging.getLogger(__name__)


def home(request):
    """Home page view with dataset status"""
    data_path = os.path.join(settings.MEDIA_ROOT or settings.BASE_DIR, 'heart.csv')
    context = {'dataset_exists': os.path.exists(data_path)}
    return render(request, 'prediction/home.html', context)


@login_required
def profile(request):
    if request.method == 'POST':
        form = UserChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Your profile was successfully updated!')
            return redirect('profile')
        messages.error(request, 'Please correct the error below.')
    else:
        form = UserChangeForm(instance=request.user)
    return render(request, 'registration/profile.html', {'form': form})


@login_required
def history(request):
    predictions = PredictionHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'prediction/history.html', {
        'predictions': predictions,
        'has_history': predictions.exists()
    })


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful! You are now logged in.')
            return redirect('home')
        messages.error(request, 'Please correct the error below.')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})


@require_http_methods(["GET", "POST"])
@login_required
def predict(request):
    """Handle the prediction form, call the model and save results to session."""
    if request.method == 'POST':
        form = HeartDiseaseForm(request.POST)
        # Support both normal form POST and AJAX (fetch/XHR)
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        if not form.is_valid():
            if is_ajax:
                return JsonResponse({
                    'success': False,
                    'errors': form.errors,
                    'message': 'Please correct the errors in the form.'
                }, status=400)
            messages.error(request, 'Please correct the errors in the form.')
            return render(request, 'prediction/predict.html', {'form': form})

        feature_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        # Check for missing fields in cleaned_data
        missing = [f for f in feature_order if f not in form.cleaned_data]
        if missing:
            msg = f"Missing form fields: {', '.join(missing)}"
            logger.error(msg)
            if is_ajax:
                return JsonResponse({'success': False, 'message': msg, 'errors': {'__all__': [msg]}}, status=400)
            messages.error(request, msg)
            return render(request, 'prediction/predict.html', {'form': form})

        try:
            input_list = [form.cleaned_data[f] for f in feature_order]
            input_dict = {f: form.cleaned_data[f] for f in feature_order}
            logger.info("Prediction input: %s", input_dict)

            # Train/load model on demand (safe to call)
            if not hasattr(kaggle_heart_predictor, 'ensemble_model') or kaggle_heart_predictor.ensemble_model is None:
                logger.info("On-demand training: training models before prediction.")
                kaggle_heart_predictor.train_models_with_kaggle_data()

            logger.info("Starting prediction...")
            model_predictions, model_probabilities = kaggle_heart_predictor.predict(input_list)
            logger.info("Model returned predictions=%s probabilities=%s", repr(model_predictions), repr(model_probabilities))

            # Process results from all algorithms
            all_results = {}
            
            if isinstance(model_predictions, dict) and isinstance(model_probabilities, dict):
                # Handle multiple model results
                for model_name in model_predictions.keys():
                    pred = model_predictions.get(model_name, 0)
                    prob = model_probabilities.get(model_name, [0.5, 0.5])
                    
                    if hasattr(pred, 'tolist'):
                        pred = pred.tolist()
                    if hasattr(prob, 'tolist'):
                        prob = prob.tolist()
                        
                    if not isinstance(pred, (list, tuple)):
                        pred = [pred]
                    if not isinstance(prob, (list, tuple)):
                        prob = [1 - float(prob), float(prob)]
                        
                    # Calculate confidence
                    pred_val = int(pred[0]) if pred else 0
                    has_disease = bool(pred_val == 1)
                    confidence = 80.0
                    
                    if prob and len(prob) > 1:
                        confidence = float(prob[1] if has_disease else prob[0]) * 100
                    elif prob and len(prob) == 1:
                        confidence = float(prob[0]) * 100 if has_disease else (1.0 - float(prob[0])) * 100
                    
                    all_results[model_name] = {
                        'prediction': pred_val,
                        'probability': confidence / 100.0,
                        'has_disease': has_disease,
                        'confidence': min(max(confidence, 0.0), 100.0)
                    }
                
                # Set ensemble as the primary result if available, otherwise use the first model
                primary_model = 'Ensemble' if 'Ensemble' in all_results else next(iter(all_results))
                primary_result = all_results[primary_model]
                
            else:
                # Single model result (backward compatibility)
                if hasattr(model_predictions, 'tolist'):
                    predictions = model_predictions.tolist()
                elif isinstance(model_predictions, (list, tuple)):
                    predictions = list(model_predictions)
                else:
                    predictions = [model_predictions]

                if hasattr(model_probabilities, 'tolist'):
                    probabilities = model_probabilities.tolist()
                elif isinstance(model_probabilities, (list, tuple)):
                    probabilities = list(model_probabilities)
                else:
                    probabilities = [model_probabilities]

                if not predictions or len(predictions) == 0:
                    raise ValueError("Model returned no predictions.")

                # Calculate confidence for single model
                pred_val = int(predictions[0]) if predictions else 0
                has_disease = bool(pred_val == 1)
                confidence = 80.0
                
                if probabilities and len(probabilities) > 0:
                    first = probabilities[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 2:
                        confidence = float(first[1] if has_disease else first[0]) * 100
                    elif isinstance(first, (int, float)):
                        confidence = float(first) * 100 if has_disease else (1.0 - float(first)) * 100
                
                primary_result = {
                    'prediction': pred_val,
                    'probability': confidence / 100.0,
                    'has_disease': has_disease,
                    'confidence': min(max(confidence, 0.0), 100.0)
                }
                all_results['Model'] = primary_result

            confidence = min(max(confidence, 0.0), 100.0)

            # Save session data with all results
            session_data = {
                'input_data': input_dict,
                'all_results': all_results,
                'last_prediction': {
                    'result': primary_result['has_disease'],
                    'confidence': primary_result['confidence'],
                    'timestamp': timezone.now().isoformat(),
                    'all_results': all_results  # Include all results in last_prediction for easy access
                }
            }

            for k, v in session_data.items():
                request.session[k] = v

            request.session.modified = True
            try:
                request.session.save()
                logger.info("Session saved. Keys: %s", list(request.session.keys()))
            except Exception as e:
                logger.exception("Failed to save session: %s", e)
                if is_ajax:
                    return JsonResponse({'success': False, 'message': 'Could not save prediction result. Please try again.', 'error': str(e)}, status=500)
                messages.error(request, "Could not save prediction result to session. Try again.")
                return render(request, 'prediction/predict.html', {'form': form})

            # Save prediction history (best-effort)
            try:
                PredictionHistory.objects.create(
                    user=request.user,
                    input_data=input_dict,
                    result=has_disease,
                    confidence=confidence
                )
                logger.info("Prediction history saved for user %s", request.user.username)
            except Exception as e:
                logger.exception("Failed to save prediction history (non-fatal): %s", str(e))

            # Return success (AJAX) or redirect to result
            if is_ajax:
                return JsonResponse({
                    'success': True,
                    'redirect_url': reverse('result'),
                    'prediction': has_disease,
                    'confidence': round(confidence, 2)
                })
            return redirect('result')

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.exception("Error in predict view: %s", error_msg)
            if is_ajax:
                return JsonResponse({'success': False, 'message': error_msg, 'error': str(e)}, status=500)
            messages.error(request, error_msg)
            return render(request, 'prediction/predict.html', {'form': form})

    # GET
    form = HeartDiseaseForm()
    return render(request, 'prediction/predict.html', {'form': form})


@login_required
def result(request):
    """Render the prediction result page using data from session.
    Displays results from all algorithms used in the prediction.
    """
    try:
        logger.info("Entering result view. Session keys: %s", list(request.session.keys()))

        # Get the last prediction data
        last_prediction = request.session.get('last_prediction', {})
        all_results = last_prediction.get('all_results', request.session.get('all_results', {}))
        
        if not all_results:
            messages.warning(request, 'No prediction results found. Please make a prediction first.')
            return redirect('predict')
            
        # Get input data and format it for display
        input_data = request.session.get('input_data', {})
        formatted_input = {}
        for k, v in input_data.items():
            if isinstance(v, str) and v.isdigit() and k != 'sex':
                v = int(v)
            formatted_input[k] = v
            
        # Prepare context with all results
        context = {
            'all_results': all_results,
            'primary_result': None,
            'input_data': formatted_input,
            'timestamp': last_prediction.get('timestamp', timezone.now().isoformat())
        }
        
        # Set primary result (use Ensemble or first available)
        if 'Ensemble' in all_results:
            context['primary_result'] = all_results['Ensemble']
        elif all_results:
            context['primary_result'] = next(iter(all_results.values()))
            
        # For backward compatibility
        if context['primary_result']:
            context.update({
                'prediction': 1 if context['primary_result']['has_disease'] else 0,
                'confidence': round(context['primary_result']['confidence'], 2),
                'has_heart_disease': bool(context['primary_result']['has_disease'])
            })
        else:
            # Backward-compatible: read older session keys
            predictions = request.session.get('predictions')
            probabilities = request.session.get('probabilities')
            input_data = request.session.get('input_data', {})

            if not predictions or probabilities is None:
                messages.warning(request, 'No prediction results found. Please make a prediction first.')
                return redirect('predict')

            # Normalize
            if hasattr(predictions, 'tolist'):
                predictions = predictions.tolist()
            if hasattr(probabilities, 'tolist'):
                probabilities = probabilities.tolist()
            if not isinstance(predictions, (list, tuple)):
                predictions = [predictions]
            if not isinstance(probabilities, (list, tuple)):
                probabilities = [probabilities]

            pred_val = predictions[0] if len(predictions) > 0 else 0

            # Compute confidence
            confidence = 80.0
            try:
                if probabilities and len(probabilities) > 0:
                    first = probabilities[0]
                    if isinstance(first, (list, tuple)) and len(first) > 1:
                        confidence = float(first[1] if int(pred_val) == 1 else first[0]) * 100
                    elif isinstance(first, (int, float)):
                        confidence = float(first) * 100 if int(pred_val) == 1 else (1.0 - float(first)) * 100
            except Exception:
                logger.exception("Error computing confidence; using default 80.0")

            confidence = min(max(confidence, 0.0), 100.0)
            has_disease = int(pred_val) == 1

            # Format input values for display (avoid converting sex)
            formatted_input = {}
            for k, v in input_data.items():
                if isinstance(v, str) and v.isdigit() and k != 'sex':
                    v = int(v)
                formatted_input[k] = v

            context = {
                'prediction': int(pred_val) if pred_val is not None else 0,
                'confidence': round(confidence, 2),
                'input_data': formatted_input,
                'has_heart_disease': has_disease,
                'timestamp': timezone.now().isoformat()
            }

        logger.info("Rendering result with context: %s", context)

        # Debug: stripped template to isolate frontend asset issues
        if request.GET.get('debug') == '1':
            return render(request, 'prediction/result_debug.html', {
                'prediction': context['prediction'],
                'confidence': context['confidence'],
                'input_data': context['input_data']
            })

        return render(request, 'prediction/prediction_result.html', context)

    except Exception as e:
        logger.exception("Error rendering result view: %s", e)
        messages.error(request, f'Error displaying results: {str(e)}')
        return redirect('predict')


@require_http_methods(["POST"])
def train_models(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        messages.error(request, 'You do not have permission to perform this action.')
        return redirect('home')
    try:
        result = kaggle_heart_predictor.train_models()
        messages.success(request, result)
    except Exception as e:
        logger.exception("Model training failed: %s", e)
        messages.error(request, 'Failed to train models. Please try again later.')
    return redirect('home')


def data_analysis(request):
    if not request.user.is_authenticated:
        messages.warning(request, 'Please log in to view data analysis.')
        return redirect('login')

    cache_key = f'data_analysis_results_{request.user.id}'
    context = cache.get(cache_key)
    if context:
        return render(request, 'prediction/data_analysis.html', context)

    try:
        # Try multiple possible paths for the dataset
        possible_paths = [
            os.path.join(settings.BASE_DIR, 'heart_disese_prediction', 'data', 'Heart_Disease_Prediction.csv'),
            os.path.join(settings.BASE_DIR, 'data', 'Heart_Disease_Prediction.csv'),
            os.path.join(settings.MEDIA_ROOT or settings.BASE_DIR, 'heart.csv'),
            os.path.join(settings.BASE_DIR, 'heart.csv')
        ]

        data_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not data_path:
            logger.warning("Dataset not found in any of %s", possible_paths)
            messages.warning(request, 'Heart disease dataset not found. Please ensure dataset is uploaded.')
            return render(request, 'prediction/data_analysis.html', 
                         {'error': 'Dataset not found', 'show_upload_help': True})

        # Load and standardize column names
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError("Dataset is empty")
            
        # Standardize column names (convert to lowercase and replace spaces with underscores)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle different target column names
        target_col = next((col for col in ['heart_disease', 'target', 'disease'] if col in df.columns), None)
        if not target_col:
            raise ValueError("Could not find target column in dataset")
            
        # Convert target to binary if needed
        if df[target_col].dtype == 'object':
            df['target'] = df[target_col].apply(lambda x: 1 if str(x).lower() in ['presence', 'yes', '1', 'true'] else 0)
            target_col = 'target'
            
        # Get basic statistics
        total_patients = len(df)
        heart_disease_count = df[target_col].sum()
        no_heart_disease_count = total_patients - heart_disease_count
        
        # Prepare data for visualization
        df_viz = df.copy()
        df_viz['heart_disease'] = df_viz[target_col].map({1: 'Presence', 0: 'Absence'})
        
        # Generate correlation matrix (only numeric columns)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
            
        corr = df[numeric_cols].corr()
        
        # Create correlation heatmap
        corr_fig = px.imshow(
            corr, 
            labels=dict(color="Correlation"),
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1, 
            zmax=1
        )
        corr_fig.update_layout(
            title='Feature Correlation Matrix',
            width=800,
            height=700
        )
        corr_plot = pio.to_html(corr_fig, full_html=False)
        
        # Create target distribution plot
        target_dist = px.pie(
            df_viz, 
            names='heart_disease',
            title='Heart Disease Distribution',
            color='heart_disease',
            color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'},
            category_orders={'heart_disease': ['Presence', 'Absence']}
        )
        target_dist.update_traces(textposition='inside', textinfo='percent+label')
        target_dist_plot = pio.to_html(target_dist, full_html=False)
        
        # Create age distribution plot
        age_dist = px.histogram(
            df_viz, 
            x='age', 
            color='heart_disease',
            title='Age Distribution by Heart Disease Status',
            barmode='overlay',
            color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'},
            opacity=0.7,
            nbins=20
        )
        age_dist_plot = pio.to_html(age_dist, full_html=False)
        
        # Create cholesterol distribution plot (if column exists)
        chol_plot = None
        if 'cholesterol' in df.columns:
            chol_dist = px.box(
                df_viz, 
                x='heart_disease', 
                y='cholesterol',
                title='Cholesterol Distribution by Heart Disease Status',
                color='heart_disease',
                color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'},
                category_orders={'heart_disease': ['Presence', 'Absence']}
            )
            chol_dist.update_layout(
                xaxis_title='Heart Disease',
                yaxis_title='Cholesterol Level',
                showlegend=False
            )
            chol_plot = pio.to_html(chol_dist, full_html=False)
        
        # Prepare context
        context = {
            'total_patients': total_patients,
            'heart_disease_count': heart_disease_count,
            'no_heart_disease_count': no_heart_disease_count,
            'corr_plot': corr_plot,
            'target_dist_plot': target_dist_plot,
            'age_dist_plot': age_dist_plot,
            'chol_dist_plot': chol_plot,
            'columns': df.columns.tolist(),
            'head': df.head(10).to_dict('records'),
            'dataset_info': f"Analyzing {total_patients} patient records with {len(df.columns)} features."
        }
        
        # Cache the results for 1 hour
        cache.set(cache_key, context, timeout=3600)
        
        return render(request, 'prediction/data_analysis.html', context)
        
    except Exception as e:
        logger.exception("Error in data analysis view")
        return render(request, 'prediction/data_analysis.html', 
                     {'error': f'Error processing data: {str(e)}', 'show_upload_help': True})

# ... (rest of the code remains the same)
    df = df.copy()
    df.columns = df.columns.str.lower()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            mode = df[col].mode()
            df[col].fillna(mode.iloc[0] if not mode.empty else "", inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

    
@require_POST
@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'Logged out successfully!')
    return redirect(settings.LOGIN_URL)

# Enhanced views.py section for data_analysis
import os
import logging
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.cache import cache
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

logger = logging.getLogger(__name__)


@login_required
def data_analysis(request):
    """Enhanced data analysis view with all visualizations"""
    cache_key = f'data_analysis_results_{request.user.id}'
    context = cache.get(cache_key)
    
    if context:
        return render(request, 'prediction/data_analysis.html', context)

    try:
        # Find dataset
        possible_paths = [
            os.path.join(settings.BASE_DIR, 'heart_disese_prediction', 'data', 'Heart_Disease_Prediction.csv'),
            os.path.join(settings.BASE_DIR, 'data', 'Heart_Disease_Prediction.csv'),
            os.path.join(settings.MEDIA_ROOT or settings.BASE_DIR, 'heart.csv'),
        ]

        data_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not data_path:
            messages.warning(request, 'Dataset not found. Please ensure dataset is uploaded.')
            return render(request, 'prediction/data_analysis.html', 
                         {'error': 'Dataset not found', 'show_upload_help': True})

        # Load data
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle target column
        target_col = next((col for col in ['heart_disease', 'target', 'disease'] if col in df.columns), None)
        if not target_col:
            raise ValueError("Could not find target column")
            
        if df[target_col].dtype == 'object':
            df['target'] = df[target_col].apply(lambda x: 1 if str(x).lower() in ['presence', 'yes', '1', 'true'] else 0)
            target_col = 'target'
        
        # Basic statistics
        total_patients = len(df)
        heart_disease_count = df[target_col].sum()
        no_heart_disease_count = total_patients - heart_disease_count
        
        # Prepare visualization data
        df_viz = df.copy()
        df_viz['heart_disease_label'] = df_viz[target_col].map({1: 'Presence', 0: 'Absence'})
        
        # 1. Target Distribution Pie Chart
        target_dist = px.pie(
            df_viz, 
            names='heart_disease_label',
            title='Heart Disease Distribution',
            color='heart_disease_label',
            color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'},
            hole=0.4
        )
        target_dist.update_traces(textposition='inside', textinfo='percent+label')
        target_dist_plot = pio.to_html(target_dist, full_html=False)
        
        # 2. Age Distribution
        age_dist = px.histogram(
            df_viz, 
            x='age', 
            color='heart_disease_label',
            title='Age Distribution by Heart Disease Status',
            barmode='overlay',
            color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'},
            opacity=0.7,
            nbins=20
        )
        age_dist.update_layout(xaxis_title='Age', yaxis_title='Count')
        age_dist_plot = pio.to_html(age_dist, full_html=False)
        
        # 3. Age Box Plot
        age_box = px.box(
            df_viz, 
            x='heart_disease_label', 
            y='age',
            title='Age Distribution Comparison',
            color='heart_disease_label',
            color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'}
        )
        age_box.update_layout(showlegend=False)
        age_box_plot = pio.to_html(age_box, full_html=False)
        
        # 4. Blood Pressure Analysis (if exists)
        bp_plot = None
        if 'bp' in df.columns or 'trestbps' in df.columns:
            bp_col = 'bp' if 'bp' in df.columns else 'trestbps'
            bp_box = px.box(
                df_viz, 
                x='heart_disease_label', 
                y=bp_col,
                title='Blood Pressure Distribution',
                color='heart_disease_label',
                color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'}
            )
            bp_box.update_layout(showlegend=False, yaxis_title='Blood Pressure')
            bp_plot = pio.to_html(bp_box, full_html=False)
        
        # 5. Cholesterol Analysis
        chol_plot = None
        if 'cholesterol' in df.columns or 'chol' in df.columns:
            chol_col = 'cholesterol' if 'cholesterol' in df.columns else 'chol'
            chol_box = px.box(
                df_viz, 
                x='heart_disease_label', 
                y=chol_col,
                title='Cholesterol Distribution',
                color='heart_disease_label',
                color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'}
            )
            chol_box.update_layout(showlegend=False, yaxis_title='Cholesterol Level')
            chol_plot = pio.to_html(chol_box, full_html=False)
        
        # 6. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr = df[numeric_cols].corr()
        
        corr_fig = px.imshow(
            corr, 
            labels=dict(color="Correlation"),
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1, 
            zmax=1,
            title='Feature Correlation Matrix'
        )
        corr_fig.update_layout(width=900, height=700)
        corr_plot = pio.to_html(corr_fig, full_html=False)
        
        # 7. Sex Distribution
        sex_plot = None
        if 'sex' in df.columns:
            sex_counts = df_viz.groupby(['sex', 'heart_disease_label']).size().reset_index(name='count')
            sex_bar = px.bar(
                sex_counts,
                x='sex',
                y='count',
                color='heart_disease_label',
                title='Heart Disease Distribution by Sex',
                barmode='group',
                color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'}
            )
            sex_bar.update_layout(xaxis_title='Sex (0=Female, 1=Male)', yaxis_title='Count')
            sex_plot = pio.to_html(sex_bar, full_html=False)
        
        # 8. Chest Pain Type Distribution
        cp_plot = None
        if 'chest_pain_type' in df.columns or 'cp' in df.columns:
            cp_col = 'chest_pain_type' if 'chest_pain_type' in df.columns else 'cp'
            cp_counts = df_viz.groupby([cp_col, 'heart_disease_label']).size().reset_index(name='count')
            cp_bar = px.bar(
                cp_counts,
                x=cp_col,
                y='count',
                color='heart_disease_label',
                title='Chest Pain Type Distribution',
                barmode='group',
                color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'}
            )
            cp_bar.update_layout(xaxis_title='Chest Pain Type', yaxis_title='Count')
            cp_plot = pio.to_html(cp_bar, full_html=False)
        
        # 9. Max Heart Rate Analysis
        hr_plot = None
        if 'max_hr' in df.columns or 'thalach' in df.columns:
            hr_col = 'max_hr' if 'max_hr' in df.columns else 'thalach'
            hr_box = px.box(
                df_viz, 
                x='heart_disease_label', 
                y=hr_col,
                title='Maximum Heart Rate Distribution',
                color='heart_disease_label',
                color_discrete_map={'Presence': '#ef4444', 'Absence': '#60a5fa'}
            )
            hr_box.update_layout(showlegend=False, yaxis_title='Max Heart Rate')
            hr_plot = pio.to_html(hr_box, full_html=False)
        
        # 10. Train models and generate confusion matrices
        feature_cols = [col for col in df.columns 
                       if col not in ['heart_disease', 'target', 'heart_disease_label'] 
                       and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models and get confusion matrices
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        confusion_matrices = []
        model_accuracies = []
        
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Create confusion matrix plot
            cm_fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Disease', 'Disease'],
                y=['No Disease', 'Disease'],
                color_continuous_scale='Blues',
                title=f'{model_name} - Accuracy: {accuracy:.2%}'
            )
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    cm_fig.add_annotation(
                        x=j, y=i,
                        text=f'{cm[i,j]}<br>({cm[i,j]/cm.sum()*100:.1f}%)',
                        showarrow=False,
                        font=dict(size=14, color='white' if cm[i,j] > cm.max()/2 else 'black')
                    )
            
            cm_fig.update_layout(width=500, height=450)
            confusion_matrices.append({
                'name': model_name,
                'plot': pio.to_html(cm_fig, full_html=False),
                'accuracy': accuracy
            })
            model_accuracies.append({'model': model_name, 'accuracy': accuracy})
        
        # 11. Model Comparison Plot
        accuracy_df = pd.DataFrame(model_accuracies)
        comparison_fig = px.bar(
            accuracy_df,
            x='model',
            y='accuracy',
            title='Model Accuracy Comparison',
            color='accuracy',
            color_continuous_scale='Viridis',
            text='accuracy'
        )
        comparison_fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        comparison_fig.update_layout(
            yaxis_title='Accuracy',
            xaxis_title='Model',
            yaxis_range=[0, 1.1],
            showlegend=False
        )
        comparison_plot = pio.to_html(comparison_fig, full_html=False)
        
        # Prepare context
        context = {
            'total_patients': total_patients,
            'heart_disease_count': heart_disease_count,
            'no_heart_disease_count': no_heart_disease_count,
            'target_dist_plot': target_dist_plot,
            'age_dist_plot': age_dist_plot,
            'age_box_plot': age_box_plot,
            'bp_plot': bp_plot,
            'chol_dist_plot': chol_plot,
            'corr_plot': corr_plot,
            'sex_plot': sex_plot,
            'cp_plot': cp_plot,
            'hr_plot': hr_plot,
            'confusion_matrices': confusion_matrices,
            'comparison_plot': comparison_plot,
            'dataset_info': f"Analyzing {total_patients} patient records with {len(df.columns)} features"
        }
        
        # Cache for 1 hour
        cache.set(cache_key, context, timeout=3600)
        
        return render(request, 'prediction/data_analysis.html', context)
        
    except Exception as e:
        logger.exception("Error in data analysis view")
        return render(request, 'prediction/data_analysis.html', 
                     {'error': f'Error processing data: {str(e)}', 'show_upload_help': True})