const API_URL = 'http://localhost:8000';

// DOM Elements
const profileForm = document.getElementById('profile-form');
const profileSection = document.getElementById('profile-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const resultsGrid = document.getElementById('results-grid');
const backBtn = document.getElementById('back-btn');
const resultsSubtitle = document.getElementById('results-subtitle');
const mockNotice = document.getElementById('mock-notice');

// Loading step animation
function animateLoadingSteps() {
    const steps = ['step-1', 'step-2', 'step-3'];
    let current = 0;
    const interval = setInterval(() => {
        if (current > 0) {
            const prev = document.getElementById(steps[current - 1]);
            if (prev) {
                prev.classList.remove('active');
                prev.classList.add('done');
            }
        }
        if (current < steps.length) {
            const el = document.getElementById(steps[current]);
            if (el) el.classList.add('active');
            current++;
        } else {
            clearInterval(interval);
        }
    }, 2500);
    return interval;
}

// Form submit handler
profileForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Switch to loading state
    profileSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    let stepInterval = animateLoadingSteps();

    // Gather form data
    const englishTest = document.getElementById('english_test').value;
    const englishScoreRaw = document.getElementById('english_score').value;

    const formData = {
        name: document.getElementById('name').value,
        cgpa: parseFloat(document.getElementById('cgpa').value),
        cgpa_scale: parseFloat(document.getElementById('cgpa_scale').value),
        degree: document.getElementById('degree').value,
        field: document.getElementById('field').value,
        research_papers: parseInt(document.getElementById('research_papers').value) || 0,
        internships: parseInt(document.getElementById('internships').value) || 0,
        experience_years: 0,
        extracurriculars: document.getElementById('extracurriculars').value
            .split(',').map(s => s.trim()).filter(s => s !== ""),
        target_degree: document.getElementById('target_degree').value,
        english_test: englishTest || null,
        english_score: englishScoreRaw ? parseFloat(englishScoreRaw) : null,
    };

    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        clearInterval(stepInterval);

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error ${response.status}`);
        }

        const data = await response.json();

        // Show mock notice if in mock mode
        if (data.mock) {
            mockNotice.classList.remove('hidden');
        } else {
            mockNotice.classList.add('hidden');
        }

        // Show results
        renderResults(data.results, data.total || data.results.length, formData);

        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

    } catch (error) {
        clearInterval(stepInterval);
        loadingSection.classList.add('hidden');
        profileSection.classList.remove('hidden');
        showError(error.message);
    }
});

// Back button
backBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    profileSection.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Error toast
function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `<i data-lucide="alert-circle"></i> <span>${message}</span>`;
    document.body.appendChild(toast);
    lucide.createIcons();
    setTimeout(() => toast.remove(), 5000);
}

// Get score color based on value  — now with real range awareness
function getScoreColor(score) {
    if (score >= 80) return '#10b981';       // excellent — emerald
    if (score >= 60) return '#22d3ee';       // good — cyan
    if (score >= 40) return '#f59e0b';       // moderate — amber
    if (score >= 20) return '#f97316';       // weak — orange
    return '#ef4444';                         // poor — red
}

// Get score label
function getScoreLabel(score) {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Moderate';
    if (score >= 20) return 'Weak';
    return 'Low';
}

// Get eligibility class
function getEligibilityClass(eligibility) {
    const lower = eligibility.toLowerCase();
    if (lower.includes('not')) return 'not';
    if (lower.includes('partial')) return 'partial';
    if (lower.includes('eligible')) return 'eligible';
    return 'partial';
}

// Render results grid
function renderResults(results, total, profile) {
    resultsGrid.innerHTML = '';

    if (!results || results.length === 0) {
        resultsGrid.innerHTML = `
            <div class="no-results">
                <i data-lucide="search-x"></i>
                <p>No matching scholarships found. Try broadening your profile.</p>
            </div>`;
        lucide.createIcons();
        return;
    }

    // Update subtitle
    resultsSubtitle.textContent = `Found ${total} scholarships analyzed for ${profile.name || 'your profile'} — sorted by hybrid fit score`;

    results.forEach((s, index) => {
        const card = document.createElement('div');
        card.className = 'scholarship-card';
        card.style.animationDelay = `${index * 0.08}s`;

        const radius = 26;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (s.score / 100) * circumference;
        const scoreColor = getScoreColor(s.score);
        const scoreLabel = getScoreLabel(s.score);
        const eligClass = getEligibilityClass(s.eligibility);

        // Build programs/countries line
        const metaLine = buildMetaLine(s);

        // Build the URL — ensure it's valid
        const link = s.url && s.url !== '#' ? s.url : null;

        // Build scoring breakdown tooltip
        const breakdownHTML = s.score_breakdown 
            ? `<div class="score-breakdown">${buildBreakdownBars(s.score_breakdown)}</div>`
            : '';
        
        // Score source indicator
        const scoreSourceHTML = (s.algo_score != null || s.llm_score != null)
            ? `<div class="score-sources">
                   ${s.algo_score != null ? `<span class="source-tag algo">Algo: ${s.algo_score}%</span>` : ''}
                   ${s.llm_score != null ? `<span class="source-tag llm">AI: ${s.llm_score}%</span>` : ''}
               </div>`
            : '';

        card.innerHTML = `
            <div class="card-body">
                <div class="card-top">
                    <div class="card-tags">
                        <span class="region-tag">
                            <i data-lucide="map-pin"></i>
                            ${s.region || 'Europe'}
                        </span>
                        ${s.funding_type ? `<span class="funding-tag ${s.funding_type.toLowerCase()}">${s.funding_type} Funding</span>` : ''}
                    </div>
                    <div class="score-badge">
                        <svg class="score-svg" width="64" height="64" viewBox="0 0 64 64">
                            <circle class="score-circle-bg" cx="32" cy="32" r="${radius}" />
                            <circle class="score-circle" cx="32" cy="32" r="${radius}"
                                style="stroke: ${scoreColor}; stroke-dasharray: ${circumference}; stroke-dashoffset: ${circumference};" />
                        </svg>
                        <div class="score-label">
                            <span class="score-text" style="color: ${scoreColor};">${s.score}%</span>
                            <span class="score-sub">${scoreLabel}</span>
                        </div>
                    </div>
                </div>

                <h3 class="card-title">${s.title || 'Unknown Program'}</h3>
                ${metaLine ? `<div class="program-meta">${metaLine}</div>` : ''}
                <span class="eligibility-tag ${eligClass}">${s.eligibility}</span>

                ${scoreSourceHTML}
                ${breakdownHTML}

                <p class="analysis-text">${s.analysis || 'No analysis available.'}</p>
            </div>

            <div class="card-footer">
                ${s.gaps ? `
                    <div class="gaps-section">
                        <p class="gaps-title"><i data-lucide="target"></i> Gap Analysis</p>
                        <p class="gaps-text">${s.gaps}</p>
                    </div>
                ` : ''}
                ${link
                    ? `<a href="${link}" target="_blank" rel="noopener noreferrer" class="btn-link">
                           View Program <i data-lucide="external-link"></i>
                       </a>`
                    : `<span class="btn-link disabled">No link available</span>`
                }
            </div>
        `;

        resultsGrid.appendChild(card);

        // Animate the score circle after mount
        setTimeout(() => {
            const circle = card.querySelector('.score-circle');
            if (circle) circle.style.strokeDashoffset = offset;
        }, 150 + index * 80);
    });

    lucide.createIcons();
}

// Build score breakdown mini-bars
function buildBreakdownBars(breakdownStr) {
    if (!breakdownStr) return '';
    
    const dims = breakdownStr.split('|').map(s => s.trim());
    const dimLabels = {
        'GPA': '📊',
        'FIELD': '🎯',
        'DEGREE': '🎓',
        'RESEARCH': '📄',
        'LANGUAGE': '🌐',
        'EXPERIENCE': '💼',
    };
    
    let html = '<div class="breakdown-bars">';
    dims.forEach(dim => {
        const [name, value] = dim.split(':').map(s => s.trim());
        const numVal = parseInt(value) || 0;
        const icon = dimLabels[name] || '•';
        const barColor = numVal >= 70 ? '#10b981' : numVal >= 40 ? '#f59e0b' : '#ef4444';
        
        html += `
            <div class="breakdown-item" title="${name}: ${numVal}/100">
                <span class="bd-label">${icon} ${name.charAt(0) + name.slice(1).toLowerCase()}</span>
                <div class="bd-bar-bg">
                    <div class="bd-bar" style="width: ${numVal}%; background: ${barColor};"></div>
                </div>
                <span class="bd-value">${numVal}</span>
            </div>
        `;
    });
    html += '</div>';
    return html;
}

// Build meta line: degree + countries + funding
function buildMetaLine(s) {
    const parts = [];
    if (s.degree) parts.push(`<i data-lucide="book-open"></i> ${s.degree}`);
    if (s.countries) parts.push(`<i data-lucide="globe"></i> ${s.countries}`);
    if (s.universities) parts.push(`<i data-lucide="building-2"></i> ${s.universities}`);
    return parts.join('<span class="meta-sep">·</span>');
}
