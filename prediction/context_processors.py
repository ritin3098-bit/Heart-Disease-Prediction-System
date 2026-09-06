from django.conf import settings

def site_info(request):
    return {
        'SITE_NAME': 'Heart Disease Prediction',
        'SITE_DESCRIPTION': 'Advanced Heart Disease Prediction System',
        'CONTACT_EMAIL': 'contact@heartdiseaseprediction.com',
        'VERSION': '1.0.0',
        'DEBUG': settings.DEBUG,
    }
