// TEBScore Leaderboards UI

const API = window.location.origin;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let taxonomyTree = {};
let judgeDomains = {};
let currentTaxonomyPath = [];  // array of segment strings

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function $(id) { return document.getElementById(id); }

function rankCell(value) {
    if (value == null) return '<span class="rank-cell missing">-</span>';
    return `<span class="rank-cell">${value}</span>`;
}

function coverageBadges(sources) {
    return sources.map(s => {
        const cls = s === 'nvidia' ? 'badge-nvidia' : s === 'prollm' ? 'badge-prollm' : 'badge-domain';
        return `<span class="badge ${cls}">${s.toUpperCase()}</span>`;
    }).join(' ');
}

function showError(el, msg) { el.innerHTML = `<div class="error-msg">${msg}</div>`; }
function clearEl(el) { el.innerHTML = ''; }

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        $('tab-' + btn.dataset.tab).classList.add('active');
    });
});

// ---------------------------------------------------------------------------
// Init: load taxonomy + domains
// ---------------------------------------------------------------------------

async function init() {
    try {
        const resp = await fetch(`${API}/api/recommender/taxonomy`);
        const data = await resp.json();
        taxonomyTree = data.taxonomy || {};
        judgeDomains = data.judge_domains || {};
        populateDomainDropdown();
        renderTaxonomy();
    } catch (e) {
        console.error('Failed to load taxonomy:', e);
    }
}

function populateDomainDropdown() {
    const sel = $('judgeDomain');
    sel.innerHTML = '';
    for (const [key, val] of Object.entries(judgeDomains)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = val.display_name;
        sel.appendChild(opt);
    }
}

// ---------------------------------------------------------------------------
// Judge flow
// ---------------------------------------------------------------------------

$('btnJudge').addEventListener('click', async () => {
    const domain = $('judgeDomain').value;
    const topK = parseInt($('judgeTopK').value, 10) || 3;
    const evalName = $('evalModelName').value.trim() || null;
    const evalFamily = $('evalModelFamily').value.trim() || null;
    const exclude = $('excludeSameFamily').checked;

    clearEl($('judgeError'));
    clearEl($('judgeResults'));
    $('judgeLoading').classList.add('show');
    $('btnJudge').disabled = true;

    try {
        const resp = await fetch(`${API}/api/recommend-judges`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                domain, top_k: topK,
                evaluated_model_name: evalName,
                evaluated_model_family: evalFamily,
                exclude_same_family: exclude,
            }),
        });
        const data = await resp.json();
        if (!data.success) {
            showError($('judgeError'), data.error || 'Unknown error');
            return;
        }
        renderJudgeResults(data.result);
    } catch (e) {
        showError($('judgeError'), `Request failed: ${e.message}`);
    } finally {
        $('judgeLoading').classList.remove('show');
        $('btnJudge').disabled = false;
    }
});

function renderJudgeResults(result) {
    const models = result.recommended_models || [];
    const benchmarks = (result.used_domain_benchmarks || []).join(', ') || 'none';

    let html = `<div class="results-card">`;
    html += `<h3>Judge Recommendations</h3>`;
    html += `<div class="summary-bar">
        <span>Domain: <strong>${result.domain}</strong></span>
        <span>Candidate Pool: <strong>${result.candidate_pool_size}</strong></span>
        <span>Benchmarks: <strong>${benchmarks}</strong></span>
    </div>`;

    if (models.length === 0) {
        html += `<p style="color:var(--text-secondary)">No models found for this domain.</p>`;
    } else {
        html += `<table class="rank-table">
            <thead><tr>
                <th>Rank</th><th>Model</th><th>Family</th>
                <th>NVIDIA</th><th>ProLLM</th><th>Domain</th>
                <th>Coverage</th>
            </tr></thead><tbody>`;
        for (const m of models) {
            html += `<tr>
                <td class="rank-cell">${m.final_rank_position}</td>
                <td><strong>${m.model_name}</strong></td>
                <td>${m.model_family}</td>
                <td>${rankCell(m.nvidia_rank)}</td>
                <td>${rankCell(m.prollm_rank)}</td>
                <td>${rankCell(m.domain_rank)}</td>
                <td>${coverageBadges(m.coverage_sources)}</td>
            </tr>`;
        }
        html += `</tbody></table>`;
    }

    if (result.decision_trace && result.decision_trace.length) {
        html += `<details class="trace-section">
            <summary>Decision Trace</summary>
            <pre>${result.decision_trace.join('\n')}</pre>
        </details>`;
    }

    html += `</div>`;
    $('judgeResults').innerHTML = html;
}

// ---------------------------------------------------------------------------
// Model flow — taxonomy navigation
// ---------------------------------------------------------------------------

function renderTaxonomy() {
    renderBreadcrumb();
    renderCards();
    clearEl($('modelResults'));
    clearEl($('modelError'));
}

function renderBreadcrumb() {
    const el = $('taxonomyBreadcrumb');
    let html = `<span class="taxonomy-crumb" onclick="navigateTo(-1)">Root</span>`;
    for (let i = 0; i < currentTaxonomyPath.length; i++) {
        const node = resolveNode(currentTaxonomyPath.slice(0, i + 1));
        const display = node ? (node.display_name || currentTaxonomyPath[i]) : currentTaxonomyPath[i];
        html += ` <span class="taxonomy-sep">&rsaquo;</span> `;
        html += `<span class="taxonomy-crumb" onclick="navigateTo(${i})">${display}</span>`;
    }
    el.innerHTML = html;
}

function resolveNode(segments) {
    let node = taxonomyTree;
    for (const seg of segments) {
        if (!node[seg]) return null;
        node = node[seg];
        if (node.children && !node.benchmark) {
            node = node.children;
        }
    }
    return node;
}

function getChildren(segments) {
    let node = taxonomyTree;
    for (const seg of segments) {
        if (!node[seg]) return {};
        node = node[seg];
        if (node.children && !node.benchmark) {
            node = node.children;
        }
    }
    if (node.benchmark) return {};  // leaf
    if (node.children) return node.children;
    // node itself might be the children dict
    return node;
}

function isLeaf(node) {
    return node && node.benchmark !== undefined;
}

function renderCards() {
    const container = $('taxonomyCards');
    const children = currentTaxonomyPath.length === 0
        ? taxonomyTree
        : getChildren(currentTaxonomyPath);

    let html = '';
    for (const [key, child] of Object.entries(children)) {
        if (typeof child !== 'object' || child === null) continue;
        // Skip metadata keys that aren't child nodes
        if (['display_name', 'children', 'benchmark', 'metric_name', 'metric_units', 'metric_direction'].includes(key)) continue;

        const leaf = isLeaf(child);
        const display = child.display_name || key;
        const typeLabel = leaf ? 'Benchmark' : 'Category';
        html += `<div class="taxonomy-card" onclick="selectTaxonomyNode('${key}')">
            <div class="card-title">${display}</div>
            <div class="card-type">${typeLabel}</div>
        </div>`;
    }

    if (!html) {
        // Check if current node is a leaf
        const node = currentTaxonomyPath.length > 0
            ? resolveNode(currentTaxonomyPath)
            : null;
        if (node && isLeaf(node)) {
            container.innerHTML = '';
            fetchModelResults(node);
            return;
        }
        html = '<p style="color:var(--text-secondary)">No sub-categories available.</p>';
    }
    container.innerHTML = html;
}

function navigateTo(index) {
    if (index < 0) {
        currentTaxonomyPath = [];
    } else {
        currentTaxonomyPath = currentTaxonomyPath.slice(0, index + 1);
    }
    renderTaxonomy();
}
// Make globally accessible for inline onclick handlers
window.navigateTo = navigateTo;

function selectTaxonomyNode(key) {
    currentTaxonomyPath.push(key);

    const node = resolveNode(currentTaxonomyPath);
    if (node && isLeaf(node)) {
        renderBreadcrumb();
        $('taxonomyCards').innerHTML = '';
        fetchModelResults(node);
    } else {
        renderTaxonomy();
    }
}
window.selectTaxonomyNode = selectTaxonomyNode;

async function fetchModelResults(node) {
    const taxonomyPath = currentTaxonomyPath.join('.');
    const topK = parseInt($('modelTopK').value, 10) || 5;

    clearEl($('modelError'));
    clearEl($('modelResults'));
    $('modelLoading').classList.add('show');

    try {
        const resp = await fetch(`${API}/api/recommend-model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ taxonomy_path: taxonomyPath, top_k: topK }),
        });
        const data = await resp.json();
        if (!data.success) {
            showError($('modelError'), data.error || 'Unknown error');
            return;
        }
        renderModelResults(data.result);
    } catch (e) {
        showError($('modelError'), `Request failed: ${e.message}`);
    } finally {
        $('modelLoading').classList.remove('show');
    }
}

function renderModelResults(result) {
    const models = result.recommended_models || [];
    const direction = result.metric_direction === 'lower_better' ? 'lower is better' : 'higher is better';
    const units = result.metric_units ? ` (${result.metric_units})` : '';

    let html = `<div class="results-card">`;
    html += `<h3>${result.display_name}</h3>`;
    html += `<div class="benchmark-info">
        <span>Benchmark: <strong>${result.benchmark_name}</strong></span>
        <span>Metric: <strong>${result.metric_name}${units}</strong> (${direction})</span>
        <span>Snapshot: <strong>${result.snapshot_date || 'N/A'}</strong></span>
    </div>`;

    if (models.length === 0) {
        html += `<p style="color:var(--text-secondary)">No ranked models available for this benchmark.</p>`;
    } else {
        html += `<table class="rank-table">
            <thead><tr>
                <th>Rank</th><th>Model</th><th>Family</th><th>Score</th>
            </tr></thead><tbody>`;
        for (const m of models) {
            const scoreStr = m.score != null ? m.score : '-';
            html += `<tr>
                <td class="rank-cell">${m.rank}</td>
                <td><strong>${m.model_name}</strong></td>
                <td>${m.model_family}</td>
                <td class="rank-cell">${scoreStr}</td>
            </tr>`;
        }
        html += `</tbody></table>`;
    }

    if (result.source_url) {
        html += `<p style="margin-top:0.75rem;font-size:0.85rem;color:var(--text-secondary)">
            Source: <a href="${result.source_url}" target="_blank" rel="noopener">${result.source_url}</a>
        </p>`;
    }

    html += `</div>`;
    $('modelResults').innerHTML = html;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

init();
