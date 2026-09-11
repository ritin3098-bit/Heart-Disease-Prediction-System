# project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

# Use absolute import for your app's views (no leading dot)
from prediction import views as prediction_views

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # Main app URLs
    path('', include('prediction.urls')),

    # Authentication URLs under /accounts/
    path('accounts/', include([
        path('signup/', prediction_views.signup, name='signup'),
        path('login/', auth_views.LoginView.as_view(
            template_name='registration/login.html',
            redirect_authenticated_user=True
        ), name='login'),
        path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
        path('password_change/', auth_views.PasswordChangeView.as_view(
            template_name='registration/password_change.html'
        ), name='password_change'),
        path('password_change/done/', auth_views.PasswordChangeDoneView.as_view(
            template_name='registration/password_change_done.html'
        ), name='password_change_done'),
        path('password_reset/', auth_views.PasswordResetView.as_view(
            template_name='registration/password_reset.html',
            email_template_name='registration/password_reset_email.html',
            subject_template_name='registration/password_reset_subject.txt'
        ), name='password_reset'),
        path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(
            template_name='registration/password_reset_done.html'
        ), name='password_reset_done'),
        path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
            template_name='registration/password_reset_confirm.html'
        ), name='password_reset_confirm'),
        path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
            template_name='registration/password_reset_complete.html'
        ), name='password_reset_complete'),
        path('profile/', prediction_views.profile, name='profile'),
    ])),

    # Static pages
    path('about/', TemplateView.as_view(template_name='pages/about.html'), name='about'),
    path('contact/', TemplateView.as_view(template_name='pages/contact.html'), name='contact'),
    path('privacy/', TemplateView.as_view(template_name='pages/privacy.html'), name='privacy'),
    path('terms/', TemplateView.as_view(template_name='pages/terms.html'), name='terms'),
    path('faq/', TemplateView.as_view(template_name='pages/faq.html'), name='faq'),
    path('blog/', TemplateView.as_view(template_name='pages/blog.html'), name='blog'),
]

# Serve static/media in development
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Service worker / manifest
urlpatterns += [
    path('sw.js', TemplateView.as_view(template_name='sw.js', content_type='application/javascript'), name='sw_js'),
    path('manifest.json', TemplateView.as_view(template_name='manifest.json', content_type='application/json'), name='manifest_json'),
]
