/**
 * Fertilizer AI Agent — Frontend Logic
 * Handles form interactions, AJAX predictions, and dynamic UI updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    initSliders();
    initForm();
    initPerformanceDashboard();
    initModelBadge();
});

/* ==========================================
   Range Slider Bindings
   ========================================== */
function initSliders() {
    const sliders = [
        { id: 'temperature', suffix: '°C' },
        { id: 'humidity', suffix: '%' },
        { id: 'moisture', suffix: '%' },
        { id: 'nitrogen', suffix: '' },
        { id: 'potassium', suffix: '' },
        { id: 'phosphorous', suffix: '' },
    ];

    sliders.forEach(({ id, suffix }) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(id + 'Value');
        if (!slider || !display) return;

        // Set initial value
        display.textContent = slider.value + suffix;

        // Update on input
        slider.addEventListener('input', () => {
            display.textContent = slider.value + suffix;
            // Update slider track fill
            updateSliderTrack(slider);
        });

        // Initialize track fill
        updateSliderTrack(slider);
    });
}

function updateSliderTrack(slider) {
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, 
        rgba(34,197,94,0.5) 0%, rgba(34,197,94,0.5) ${percent}%, 
        rgba(255,255,255,0.1) ${percent}%, rgba(255,255,255,0.1) 100%)`;
}

/* ==========================================
   Form Submission
   ========================================== */
function initForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await makePrediction();
    });
}

async function makePrediction() {
    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnIcon = btn.querySelector('.btn-icon');
    const btnLoading = btn.querySelector('.btn-loading');

    // Show loading state
    btnText.style.display = 'none';
    btnIcon.style.display = 'none';
    btnLoading.style.display = 'inline-flex';
    btn.disabled = true;

    const payload = {
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        moisture: document.getElementById('moisture').value,
        soil_type: document.getElementById('soil_type').value,
        crop_type: document.getElementById('crop_type').value,
        nitrogen: document.getElementById('nitrogen').value,
        potassium: document.getElementById('potassium').value,
        phosphorous: document.getElementById('phosphorous').value,
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            showResult(data);
        }
    } catch (err) {
        showError('Connection error. Make sure the server is running.');
    } finally {
        // Reset button
        btnText.style.display = 'inline';
        btnIcon.style.display = 'inline';
        btnLoading.style.display = 'none';
        btn.disabled = false;
    }
}

/* ==========================================
   Display Result
   ========================================== */
function showResult(data) {
    const section = document.getElementById('resultSection');
    section.classList.remove('hidden');

    // Scroll to result
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Re-trigger animation
    section.style.animation = 'none';
    section.offsetHeight; // force reflow
    section.style.animation = '';

    // Fertilizer name
    document.getElementById('fertilizerName').textContent = data.fertilizer;

    // Confidence
    const confValue = document.getElementById('confidenceValue');
    if (data.confidence !== null && data.confidence !== undefined) {
        confValue.textContent = data.confidence + '%';
        document.getElementById('confidenceBadge').style.display = 'flex';
    } else {
        document.getElementById('confidenceBadge').style.display = 'none';
    }

    // Tips
    document.getElementById('tipText').textContent = data.tips || 'Follow standard application guidelines.';

    // Model
    document.getElementById('modelUsed').textContent = data.model_used;

    // Input summary
    const grid = document.getElementById('detailsGrid');
    grid.innerHTML = '';

    if (data.input_summary) {
        const labels = {
            temperature: '🌡️ Temperature',
            humidity: '💧 Humidity',
            moisture: '🌊 Moisture',
            soil_type: '🌍 Soil Type',
            crop_type: '🌾 Crop Type',
            nitrogen: '🔵 Nitrogen',
            potassium: '🟡 Potassium',
            phosphorous: '🔴 Phosphorous',
        };

        Object.entries(data.input_summary).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'detail-item';

            const suffix = key === 'temperature' ? '°C' :
                          (key === 'humidity' || key === 'moisture') ? '%' : '';

            item.innerHTML = `
                <span class="detail-label">${labels[key] || key}</span>
                <span class="detail-value">${value}${suffix}</span>
            `;
            grid.appendChild(item);
        });
    }

    // Result icon based on fertilizer
    const iconMap = {
        'Urea': '🧪',
        'DAP': '⚗️',
        '14-35-14': '🌿',
        '28-28': '🌾',
        '17-17-17': '🍀',
        '20-20': '🌻',
        '10-26-26': '🌺',
    };
    document.getElementById('resultIcon').textContent = iconMap[data.fertilizer] || '🌱';
}

function showError(message) {
    const section = document.getElementById('resultSection');
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    document.getElementById('resultIcon').textContent = '⚠️';
    document.getElementById('resultTitle').textContent = 'Prediction Error';
    document.getElementById('resultSubtitle').textContent = message;
    document.getElementById('fertilizerName').textContent = 'Error';
    document.getElementById('confidenceBadge').style.display = 'none';
    document.getElementById('tipText').textContent = 'Please check your inputs and try again.';
    document.getElementById('detailsGrid').innerHTML = '';
}

/* ==========================================
   Performance Dashboard
   ========================================== */
function initPerformanceDashboard() {
    if (!MODEL_METADATA || !MODEL_METADATA.results) {
        document.getElementById('performanceSection').style.display = 'none';
        return;
    }

    const grid = document.getElementById('performanceGrid');
    grid.innerHTML = '';

    const results = MODEL_METADATA.results;
    const bestModel = MODEL_METADATA.best_model;

    // Sort by accuracy descending
    const sorted = Object.entries(results).sort((a, b) => b[1].accuracy - a[1].accuracy);

    sorted.forEach(([name, metrics], index) => {
        const isBest = name === bestModel;
        const card = document.createElement('div');
        card.className = `perf-card ${isBest ? 'best' : ''}`;

        card.innerHTML = `
            <div class="perf-card-header">
                <span class="perf-card-name">${name}</span>
                ${isBest ? '<span class="perf-best-badge">🏆 Best</span>' : ''}
            </div>
            <div class="perf-metrics">
                ${renderMetric('Accuracy', metrics.accuracy, 'accuracy')}
                ${renderMetric('Precision', metrics.precision, 'precision')}
                ${renderMetric('Recall', metrics.recall, 'recall')}
                ${renderMetric('F1 Score', metrics.f1_score, 'f1')}
            </div>
        `;

        grid.appendChild(card);

        // Animate bars after a small delay
        setTimeout(() => {
            card.querySelectorAll('.perf-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.value + '%';
            });
        }, 200 + index * 150);
    });
}

function renderMetric(label, value, className) {
    return `
        <div class="perf-metric">
            <div class="perf-metric-header">
                <span class="perf-metric-label">${label}</span>
                <span class="perf-metric-value">${value}%</span>
            </div>
            <div class="perf-bar">
                <div class="perf-bar-fill ${className}" data-value="${value}"></div>
            </div>
        </div>
    `;
}

/* ==========================================
   Model Badge
   ========================================== */
function initModelBadge() {
    const nameEl = document.getElementById('modelName');
    if (MODEL_METADATA && MODEL_METADATA.best_model) {
        nameEl.textContent = MODEL_METADATA.best_model + ' Active';
    } else {
        nameEl.textContent = 'No model loaded';
        document.querySelector('.badge-dot').style.background = '#ef4444';
    }
}
