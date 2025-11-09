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

// Configuration elements
const weightT = document.getElementById('weightT');
const weightE = document.getElementById('weightE');
const weightB = document.getElementById('weightB');
const weightValidation = document.getElementById('weightValidation');
const judgeCountT = document.getElementById('judgeCountT');
const judgeCountE = document.getElementById('judgeCountE');
const judgeCountB = document.getElementById('judgeCountB');

// Result display elements
const trustScoreEl = document.getElementById('trustScore');
const trustScoreBar = document.getElementById('trustScoreBar');
const trustScoreCI = document.getElementById('trustScoreCI');
const severityBadge = document.getElementById('severityBadge');
const confidenceBadge = document.getElementById('confidenceBadge');
const scoreT = document.getElementById('scoreT');
const scoreE = document.getElementById('scoreE');
const scoreB = document.getElementById('scoreB');
const confT = document.getElementById('confT');
const confE = document.getElementById('confE');
const confB = document.getElementById('confB');
const countT = document.getElementById('countT');
const countE = document.getElementById('countE');
const countB = document.getElementById('countB');
const highlightedResponse = document.getElementById('highlightedResponse');
const errorsList = document.getElementById('errorsList');
const errorCount = document.getElementById('errorCount');

// Modal elements
const categoryModal = document.getElementById('categoryModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const modalClose = document.getElementById('modalClose');

// Store current results for filtering
let currentResults = null;

// Event Listeners
analyzeBtn.addEventListener('click', handleAnalyze);
weightT.addEventListener('input', validateWeights);
weightE.addEventListener('input', validateWeights);
weightB.addEventListener('input', validateWeights);
modalClose.addEventListener('click', () => categoryModal.classList.add('hidden'));
categoryModal.addEventListener('click', (e) => {
    if (e.target === categoryModal) {
        categoryModal.classList.add('hidden');
    }
});

// Category card click handlers
document.getElementById('categoryT').addEventListener('click', () => showCategoryErrors('T', 'Trustworthiness'));
document.getElementById('categoryE').addEventListener('click', () => showCategoryErrors('E', 'Explainability'));
document.getElementById('categoryB').addEventListener('click', () => showCategoryErrors('B', 'Bias'));

// Validate weights sum to 1.0
function validateWeights() {
    const t = parseFloat(weightT.value) || 0;
    const e = parseFloat(weightE.value) || 0;
    const b = parseFloat(weightB.value) || 0;
    const sum = t + e + b;
    
    if (Math.abs(sum - 1.0) > 0.01) {
        weightValidation.textContent = `Weights must sum to 1.0 (currently: ${sum.toFixed(2)})`;
        weightValidation.style.color = 'var(--danger-color)';
        return false;
    } else {
        weightValidation.textContent = '';
        return true;
    }
}

// Get configuration from UI
function getConfiguration() {
    return {
        weights: {
            trustworthiness: parseFloat(weightT.value) || 0.5,
            explainability: parseFloat(weightE.value) || 0.3,
            bias: parseFloat(weightB.value) || 0.2
        },
        judge_counts: {
            trustworthiness: parseInt(judgeCountT.value) || 3,
            explainability: parseInt(judgeCountE.value) || 3,
            bias: parseInt(judgeCountB.value) || 3
        }
    };
}

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

    if (!validateWeights()) {
        showError('Please fix weight configuration. Weights must sum to 1.0.');
        return;
    }

    // Show loading, hide results
    showLoading();
    hideError();
    hideResults();

    try {
        const config = getConfiguration();
        const response_data = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt,
                response,
                model,
                use_mock: useMock,
                config: config
            })
        });

        const data = await response_data.json();

        if (data.success) {
            currentResults = data.result;
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

// Get severity rating based on score
function getSeverityRating(score) {
    if (score <= 0.5) return { level: 'low', label: 'Low' };
    if (score <= 1.5) return { level: 'medium', label: 'Medium' };
    return { level: 'high', label: 'High' };
}

// Get confidence rating based on confidence value
function getConfidenceRating(confidence) {
    if (confidence >= 0.8) return { level: 'high', label: 'High' };
    if (confidence >= 0.6) return { level: 'medium', label: 'Medium' };
    return { level: 'low', label: 'Low' };
}

// Display results
function displayResults(result) {
    // Display TrustScore
    const trustScore = result.summary.trust_score;
    trustScoreEl.textContent = trustScore.toFixed(3);
    
    // Calculate percentage for bar (assuming score is 0-10, lower is better)
    const qualityPercent = Math.max(0, Math.min(100, (10 - trustScore) / 10 * 100));
    trustScoreBar.style.width = `${qualityPercent}%`;
    
    // Display severity and confidence ratings
    const severityRating = getSeverityRating(trustScore);
    severityBadge.textContent = severityRating.label;
    severityBadge.className = `rating-badge severity-${severityRating.level}`;
    
    const confidenceRating = getConfidenceRating(result.summary.trust_confidence);
    confidenceBadge.textContent = confidenceRating.label;
    confidenceBadge.className = `rating-badge confidence-${confidenceRating.level}`;
    
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

    // Count errors by category
    const errorCounts = { T: 0, E: 0, B: 0 };
    result.errors.forEach(error => {
        errorCounts[error.type] = (errorCounts[error.type] || 0) + 1;
    });
    countT.textContent = `(${errorCounts.T})`;
    countE.textContent = `(${errorCounts.E})`;
    countB.textContent = `(${errorCounts.B})`;

    // Display highlighted response
    displayHighlightedResponse(result.response, result.errors);

    // Display errors
    displayErrors(result.errors);
    errorCount.textContent = result.error_count;

    // Show results section
    showResults();
}

// Display response with error highlights (with severity-based colors and tooltips)
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
        const severity = error.severity_bucket.toLowerCase();

        // Add text before error
        if (start > lastIndex) {
            html += escapeHtml(response.substring(lastIndex, start));
        }

        // Build tooltip content
        const tooltipContent = buildTooltipContent(error);

        // Add highlighted error text with tooltip
        const errorText = response.substring(start, end);
        html += `<span class="highlight ${errorType} severity-${severity}" data-error-index="${index}">
            ${escapeHtml(errorText)}
            <div class="highlight-tooltip">${tooltipContent}</div>
        </span>`;

        lastIndex = end;
    });

    // Add remaining text
    if (lastIndex < response.length) {
        html += escapeHtml(response.substring(lastIndex));
    }

    highlightedResponse.innerHTML = html || escapeHtml(response);
}

// Build tooltip content for error
function buildTooltipContent(error) {
    let html = `<div class="tooltip-explanation">${escapeHtml(error.explanation)}</div>`;
    
    // Add judge analyses if available
    if (error.judge_analyses && Object.keys(error.judge_analyses).length > 0) {
        html += '<div class="tooltip-judges"><strong>Judge Analyses:</strong>';
        for (const [judgeName, analysis] of Object.entries(error.judge_analyses)) {
            html += `<div class="tooltip-judge-item">
                <strong>${escapeHtml(judgeName)}</strong> (${escapeHtml(analysis.model)}):
                Severity: ${analysis.severity_score.toFixed(3)}, 
                Confidence: ${(analysis.confidence * 100).toFixed(1)}%
            </div>`;
        }
        html += '</div>';
    }
    
    // Add CIs
    if (error.severity_score_ci.lower !== null) {
        html += `<div class="tooltip-ci">
            <strong>Severity CI:</strong> [${error.severity_score_ci.lower.toFixed(3)}, ${error.severity_score_ci.upper.toFixed(3)}]
        </div>`;
    }
    
    if (error.confidence_ci.lower !== null) {
        html += `<div class="tooltip-ci">
            <strong>Confidence CI:</strong> [${error.confidence_ci.lower.toFixed(3)}, ${error.confidence_ci.upper.toFixed(3)}]
        </div>`;
    }
    
    return html;
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

// Show category-specific errors in modal
function showCategoryErrors(category, categoryName) {
    if (!currentResults) return;
    
    const filteredErrors = currentResults.errors.filter(error => error.type === category);
    
    modalTitle.textContent = `${categoryName} Errors (${filteredErrors.length})`;
    
    if (filteredErrors.length === 0) {
        modalBody.innerHTML = '<p style="text-align: center; padding: 2rem; color: var(--text-secondary);">No errors in this category.</p>';
    } else {
        modalBody.innerHTML = filteredErrors.map((error, index) => {
            const errorType = error.type.toLowerCase();
            const severity = error.severity_bucket.toLowerCase();
            
            return `
                <div class="error-card ${errorType}" style="margin-bottom: 1rem;">
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
    
    categoryModal.classList.remove('hidden');
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
