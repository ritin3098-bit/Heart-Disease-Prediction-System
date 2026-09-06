/**
 * Heart Disease Prediction System - Main JavaScript
 * Handles UI interactions, form validation, and dynamic content
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Heart Disease Prediction System initialized');
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Back to top button
    const backToTopButton = document.getElementById('backToTop');
    if (backToTopButton) {
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) {
                backToTopButton.classList.add('show');
            } else {
                backToTopButton.classList.remove('show');
            }
        });
        
        backToTopButton.addEventListener('click', (e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
    
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
    
    // Theme switcher
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        // Check for saved user preference, if any, on load
        const savedTheme = localStorage.getItem('theme') || 'light';
        setTheme(savedTheme);
        
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-bs-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
    
    // Initialize charts if Chart.js is available
    if (typeof Chart !== 'undefined') {
        initializeCharts();
    }
    
    // Initialize any prediction forms
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        initializePredictionForm(predictionForm);
    }
});

/**
 * Set the theme for the application
 * @param {string} theme - 'light' or 'dark'
 */
function setTheme(theme) {
    const html = document.documentElement;
    const themeToggle = document.getElementById('themeToggle');
    
    if (theme === 'dark') {
        html.setAttribute('data-bs-theme', 'dark');
        if (themeToggle) {
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            themeToggle.setAttribute('aria-label', 'Switch to light mode');
        }
    } else {
        html.setAttribute('data-bs-theme', 'light');
        if (themeToggle) {
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
            themeToggle.setAttribute('aria-label', 'Switch to dark mode');
        }
    }
}

/**
 * Initialize any charts on the page
 */
function initializeCharts() {
    const chartElements = document.querySelectorAll('.chart-container');
    
    chartElements.forEach(container => {
        const canvas = container.querySelector('canvas');
        if (!canvas) return;
        
        const chartType = container.dataset.chartType || 'bar';
        const chartData = JSON.parse(container.dataset.chartData || '{}');
        const chartOptions = JSON.parse(container.dataset.chartOptions || '{}');
        
        new Chart(canvas, {
            type: chartType,
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-color')
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted')
                        },
                        grid: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--gray-200')
                        }
                    },
                    x: {
                        ticks: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted')
                        },
                        grid: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--gray-200')
                        }
                    }
                },
                ...chartOptions
            }
        });
    });
}

/**
 * Initialize prediction form with dynamic behavior
 * @param {HTMLElement} form - The prediction form element
 */
function initializePredictionForm(form) {
    // Add real-time validation
    const inputs = form.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('input', () => {
            if (input.checkValidity()) {
                input.classList.remove('is-invalid');
                input.classList.add('is-valid');
            } else {
                input.classList.remove('is-valid');
                input.classList.add('is-invalid');
            }
        });
    });
    
    // Create loading overlay
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'prediction-loading-overlay';
    loadingOverlay.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-3">Processing your prediction. Please wait...</p>
    `;
    loadingOverlay.style.display = 'none';
    loadingOverlay.style.position = 'fixed';
    loadingOverlay.style.top = '0';
    loadingOverlay.style.left = '0';
    loadingOverlay.style.width = '100%';
    loadingOverlay.style.height = '100%';
    loadingOverlay.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
    loadingOverlay.style.zIndex = '9999';
    loadingOverlay.style.display = 'flex';
    loadingOverlay.style.flexDirection = 'column';
    loadingOverlay.style.justifyContent = 'center';
    loadingOverlay.style.alignItems = 'center';
    loadingOverlay.style.display = 'none';
    document.body.appendChild(loadingOverlay);
    
    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!form.checkValidity()) {
            e.stopPropagation();
            form.classList.add('was-validated');
            return;
        }
        
        const formData = new FormData(form);
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        
        // Set a timeout to prevent infinite loading
        const timeoutId = setTimeout(() => {
            showAlert('warning', 'The prediction is taking longer than expected. Please wait...');
        }, 10000);
        
        try {
            // Show loading state
            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            
            // Submit form data
            const response = await fetch(form.action, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': getCookie('csrftoken')
                }
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || 'Server returned an error');
            }
            
            const data = await response.json();
            
            if (data.redirect_url) {
                // Add a small delay to ensure the user sees the loading state
                setTimeout(() => {
                    window.location.href = data.redirect_url;
                }, 500);
            } else {
                throw new Error('No redirect URL provided');
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            showAlert('danger', `Error: ${error.message || 'An unexpected error occurred. Please try again.'}`);
            loadingOverlay.style.display = 'none';
            document.body.style.overflow = '';
        } finally {
            // Reset button state
            submitButton.disabled = false;
            submitButton.innerHTML = originalButtonText;
            clearTimeout(timeoutId);
        }
    });
}

/**
 * Show a Bootstrap alert message
 * @param {string} type - Alert type (success, danger, warning, info)
 * @param {string} message - The message to display
 * @param {number} [timeout=5000] - Time in ms before auto-dismissal
 */
function showAlert(type, message, timeout = 5000) {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.role = 'alert';
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const alertsContainer = document.querySelector('.alerts-container') || document.body;
    alertsContainer.prepend(alertContainer);
    
    // Auto-dismiss after timeout
    setTimeout(() => {
        const alert = new bootstrap.Alert(alertContainer);
        alert.close();
    }, timeout);
}

/**
 * Get cookie value by name
 * @param {string} name - Cookie name
 * @returns {string} Cookie value or empty string if not found
 */
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue || '';
}

// Export functions for use in other modules if needed
window.HeartPredictionApp = {
    showAlert,
    setTheme,
    initializeCharts
};