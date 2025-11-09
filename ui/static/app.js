// TrustScore UI - Main JavaScript

const API_BASE_URL = window.location.origin;

// DOM Elements
const promptInput = document.getElementById('prompt');
const responseInput = document.getElementById('response');
const modelInput = document.getElementById('model');
const useMockCheckbox = document.getElementById('useMock');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');

// Result display elements
const trustScoreEl = document.getElementById('trustScore');
const trustScoreBar = document.getElementById('trustScoreBar');
const trustScoreCI = document.getElementById('trustScoreCI');
const scoreT = document.getElementById('scoreT');
const scoreE = document.getElementById('scoreE');
const scoreB = document.getElementById('scoreB');
const confT = document.getElementById('confT');
const confE = document.getElementById('confE');
const confB = document.getElementById('confB');
const highlightedResponse = document.getElementById('highlightedResponse');
const errorsList = document.getElementById('errorsList');
const errorCount = document.getElementById('errorCount');

// Event Listeners
analyzeBtn.addEventListener('click', handleAnalyze);

// Handle form submission
async function handleAnalyze() {
    const prompt = promptInput.value.trim();
    const response = responseInput.value.trim();
    const model = modelInput.value.trim() || 'unknown';
    const useMock = useMockCheckbox.checked;

    // Validation
    if (!prompt || !response) {
        showError('Please provide both a prompt and a response.');
        return;
    }

    // Show loading, hide results
    showLoading();
    hideError();
    hideResults();

    try {
        const response_data = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt,
                response,
                model,
                use_mock: useMock
            })
        });

        const data = await response_data.json();

        if (data.success) {
            displayResults(data.result);
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        showError(`Network error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(result) {
    // Display TrustScore
    const trustScore = result.summary.trust_score;
    trustScoreEl.textContent = trustScore.toFixed(3);
    
    // Calculate percentage for bar (assuming score is 0-1, lower is better)
    // For display, we'll show it as a "quality" score (1 - trustScore)
    const qualityPercent = Math.max(0, Math.min(100, (1 - trustScore) * 100));
    trustScoreBar.style.width = `${qualityPercent}%`;
    
    // Display confidence interval
    const ci = result.summary.trust_score_ci;
    if (ci.lower !== null && ci.upper !== null) {
        trustScoreCI.textContent = `95% CI: [${ci.lower.toFixed(3)}, ${ci.upper.toFixed(3)}]`;
    }

    // Display category scores
    const categories = result.summary.categories;
    scoreT.textContent = categories.trustworthiness.score.toFixed(3);
    scoreE.textContent = categories.explainability.score.toFixed(3);
    scoreB.textContent = categories.bias.score.toFixed(3);
    
    confT.textContent = `Confidence: ${(categories.trustworthiness.confidence * 100).toFixed(1)}%`;
    confE.textContent = `Confidence: ${(categories.explainability.confidence * 100).toFixed(1)}%`;
    confB.textContent = `Confidence: ${(categories.bias.confidence * 100).toFixed(1)}%`;

    // Display highlighted response
    displayHighlightedResponse(result.response, result.errors);

    // Display errors
    displayErrors(result.errors);
    errorCount.textContent = result.error_count;

    // Show results section
    showResults();
}

// Display response with error highlights
function displayHighlightedResponse(response, errors) {
    // Sort errors by start position
    const sortedErrors = [...errors].sort((a, b) => {
        const startA = a.span ? a.span.start : 0;
        const startB = b.span ? b.span.start : 0;
        return startA - startB;
    });

    let html = '';
    let lastIndex = 0;

    sortedErrors.forEach((error, index) => {
        if (!error.span) return;

        const { start, end } = error.span;
        const errorType = error.type.toLowerCase();

        // Add text before error
        if (start > lastIndex) {
            html += escapeHtml(response.substring(lastIndex, start));
        }

        // Add highlighted error text
        const errorText = response.substring(start, end);
        html += `<span class="highlight ${errorType}" data-error-index="${index}" title="${escapeHtml(error.explanation)}">${escapeHtml(errorText)}</span>`;

        lastIndex = end;
    });

    // Add remaining text
    if (lastIndex < response.length) {
        html += escapeHtml(response.substring(lastIndex));
    }

    highlightedResponse.innerHTML = html || escapeHtml(response);
}

// Display errors list
function displayErrors(errors) {
    if (errors.length === 0) {
        errorsList.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No errors detected! 🎉</p>';
        return;
    }

    errorsList.innerHTML = errors.map((error, index) => {
        const errorType = error.type.toLowerCase();
        const severity = error.severity_bucket.toLowerCase();
        
        return `
            <div class="error-card ${errorType}" data-error-index="${index}">
                <div class="error-header">
                    <div class="error-type">
                        <span class="error-badge ${errorType}">${error.type}</span>
                        <span class="error-subtype">${error.subtype.replace(/_/g, ' ')}</span>
                    </div>
                    <span class="severity-badge ${severity}">${severity}</span>
                </div>
                <div class="error-explanation">${escapeHtml(error.explanation)}</div>
                <div class="error-metrics">
                    <div class="metric">
                        <span class="metric-label">Severity Score</span>
                        <span class="metric-value">${error.severity_score.toFixed(3)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value">${(error.confidence_level * 100).toFixed(1)}%</span>
                    </div>
                    ${error.span ? `<div class="metric">
                        <span class="metric-label">Location</span>
                        <span class="metric-value">Chars ${error.span.start}-${error.span.end}</span>
                    </div>` : ''}
                </div>
                ${error.severity_score_ci.lower !== null ? `
                    <div class="error-metrics" style="margin-top: 0.5rem;">
                        <div class="metric">
                            <span class="metric-label">Severity CI</span>
                            <span class="metric-value">[${error.severity_score_ci.lower.toFixed(3)}, ${error.severity_score_ci.upper.toFixed(3)}]</span>
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoading() {
    loadingIndicator.classList.remove('hidden');
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingIndicator.classList.add('hidden');
    analyzeBtn.disabled = false;
}

function showResults() {
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// Add some example data on page load for demo purposes
window.addEventListener('load', () => {
    // You can pre-fill with example data if needed
    // promptInput.value = "What is the capital of France?";
    // responseInput.value = "The capital of France is Paris.";
});

