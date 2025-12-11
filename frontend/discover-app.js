// Configuration
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '/api';

// State
let insightsData = null;
let expandedArea = null;
let areaPapersCache = {};

// Load insights on page load
window.addEventListener('DOMContentLoaded', () => {
    loadInsights();
});

// Load research insights
async function loadInsights() {
    const loadingState = document.getElementById('loading-state');
    const emptyState = document.getElementById('empty-state');
    const contentContainer = document.getElementById('content-container');

    loadingState.classList.remove('hidden');
    emptyState.classList.add('hidden');
    contentContainer.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE_URL}/research/insights`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        insightsData = data;

        loadingState.classList.add('hidden');

        // Check if we have any data
        if (!data.kb_context || data.kb_context.total_papers === 0) {
            emptyState.classList.remove('hidden');
            return;
        }

        contentContainer.classList.remove('hidden');

        // Update statistics
        document.getElementById('stat-papers').textContent = data.kb_context.total_papers;
        document.getElementById('stat-categories').textContent = Object.keys(data.kb_context.categories || {}).length;

        // Show generation time if available
        if (data.generation_metadata && data.generation_metadata.generation_time_seconds) {
            document.getElementById('generation-time').textContent =
                `Generated in ${data.generation_metadata.generation_time_seconds}s`;
        }

        // Render sections
        renderResearchAreas(data.research_areas || []);
        renderTrendingTopics(data.trending_topics || []);
        renderResearchGaps(data.research_gaps || []);
        renderNextSteps(data.next_steps || []);

    } catch (error) {
        console.error('Failed to load insights:', error);
        loadingState.classList.add('hidden');
        contentContainer.classList.add('hidden');

        // Show error state
        document.getElementById('empty-state').innerHTML = `
            <div class="text-6xl mb-4">⚠️</div>
            <h2 class="text-2xl font-bold text-gray-900 mb-2">Failed to Load Insights</h2>
            <p class="text-gray-600 mb-6">${error.message}</p>
            <button onclick="loadInsights()" class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                Try Again
            </button>
        `;
        document.getElementById('empty-state').classList.remove('hidden');
    }
}

// Refresh insights (clear cache)
async function refreshInsights() {
    areaPapersCache = {};
    expandedArea = null;
    await loadInsights();
}

// Render research areas as clickable cards
function renderResearchAreas(areas) {
    const container = document.getElementById('research-areas');

    if (!areas || areas.length === 0) {
        container.innerHTML = `
            <div class="col-span-full text-center py-8 text-gray-500">
                No research areas identified yet. Index more papers to see clustering.
            </div>
        `;
        return;
    }

    container.innerHTML = areas.map((area, index) => `
        <div class="area-card bg-white rounded-lg shadow p-6 border-2 border-transparent" id="area-card-${index}" onclick="toggleAreaPapers(${index}, '${escapeHtml(area.name)}')">
            <div class="flex items-start justify-between mb-3">
                <h3 class="text-lg font-semibold text-gray-900">${escapeHtml(area.name)}</h3>
                <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">
                    ${area.paper_count || '?'} papers
                </span>
            </div>
            <p class="text-gray-600 text-sm mb-3">${escapeHtml(area.description || '')}</p>
            ${area.significance ? `<p class="text-xs text-purple-600"><strong>Significance:</strong> ${escapeHtml(area.significance)}</p>` : ''}
            <div class="mt-3 flex items-center text-sm text-blue-600">
                <span id="area-toggle-${index}">Click to view papers →</span>
            </div>
            <div class="papers-panel mt-4" id="area-papers-${index}">
                <div class="border-t pt-4">
                    <div id="area-papers-content-${index}" class="space-y-3">
                        <!-- Papers will be loaded here -->
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Toggle area papers panel
async function toggleAreaPapers(index, areaName) {
    const card = document.getElementById(`area-card-${index}`);
    const panel = document.getElementById(`area-papers-${index}`);
    const toggle = document.getElementById(`area-toggle-${index}`);
    const content = document.getElementById(`area-papers-content-${index}`);

    // If already expanded, collapse it
    if (expandedArea === index) {
        card.classList.remove('expanded');
        panel.classList.remove('expanded');
        toggle.textContent = 'Click to view papers →';
        expandedArea = null;
        return;
    }

    // Collapse any other expanded area
    if (expandedArea !== null) {
        const prevCard = document.getElementById(`area-card-${expandedArea}`);
        const prevPanel = document.getElementById(`area-papers-${expandedArea}`);
        const prevToggle = document.getElementById(`area-toggle-${expandedArea}`);
        if (prevCard) prevCard.classList.remove('expanded');
        if (prevPanel) prevPanel.classList.remove('expanded');
        if (prevToggle) prevToggle.textContent = 'Click to view papers →';
    }

    // Expand this area
    card.classList.add('expanded');
    panel.classList.add('expanded');
    toggle.textContent = 'Click to hide papers ↑';
    expandedArea = index;

    // Load papers if not cached
    if (!areaPapersCache[areaName]) {
        content.innerHTML = '<div class="text-center py-4 text-gray-500">Loading papers...</div>';

        try {
            const response = await fetch(`${API_BASE_URL}/research/areas/${encodeURIComponent(areaName)}/papers?limit=10`);
            const data = await response.json();
            areaPapersCache[areaName] = data.papers || [];
        } catch (error) {
            console.error('Failed to load area papers:', error);
            content.innerHTML = '<div class="text-center py-4 text-red-500">Failed to load papers</div>';
            return;
        }
    }

    // Render papers
    const papers = areaPapersCache[areaName];
    if (papers.length === 0) {
        content.innerHTML = '<div class="text-center py-4 text-gray-500">No papers found for this area</div>';
    } else {
        content.innerHTML = papers.map(paper => `
            <div class="paper-mini-card bg-gray-50 p-3 rounded">
                <div class="font-medium text-sm text-gray-900">${escapeHtml(paper.title)}</div>
                <div class="text-xs text-gray-500 mt-1">
                    ${paper.authors ? paper.authors.slice(0, 2).join(', ') : 'Unknown'}
                    ${paper.authors && paper.authors.length > 2 ? ' +' + (paper.authors.length - 2) + ' more' : ''}
                </div>
                <div class="flex items-center justify-between mt-2">
                    <span class="text-xs text-blue-600">Relevance: ${(paper.relevance_score * 100).toFixed(0)}%</span>
                    <a href="index.html?ask=${encodeURIComponent('Tell me about: ' + paper.title)}" class="text-xs text-purple-600 hover:underline">
                        Ask about this →
                    </a>
                </div>
            </div>
        `).join('');
    }
}

// Render trending topics as chips
function renderTrendingTopics(topics) {
    const container = document.getElementById('trending-topics');

    if (!topics || topics.length === 0) {
        container.innerHTML = '<div class="text-gray-500">No trending topics identified yet.</div>';
        return;
    }

    const colors = [
        'bg-blue-100 text-blue-800',
        'bg-green-100 text-green-800',
        'bg-purple-100 text-purple-800',
        'bg-amber-100 text-amber-800',
        'bg-pink-100 text-pink-800',
        'bg-cyan-100 text-cyan-800'
    ];

    container.innerHTML = topics.map((topic, i) => `
        <span class="topic-chip ${colors[i % colors.length]}" onclick="searchTopic('${escapeHtml(topic)}')">
            ${escapeHtml(topic)}
        </span>
    `).join('');
}

// Search for a topic
function searchTopic(topic) {
    window.location.href = `index.html?query=${encodeURIComponent(topic)}`;
}

// Render research gaps
function renderResearchGaps(gaps) {
    const container = document.getElementById('research-gaps');

    if (!gaps || gaps.length === 0) {
        container.innerHTML = '<div class="text-gray-500">No research gaps identified yet.</div>';
        return;
    }

    container.innerHTML = gaps.map(gap => `
        <div class="gap-card bg-white rounded-lg shadow p-4">
            <h4 class="font-semibold text-gray-900">${escapeHtml(gap.gap_title || gap.title || 'Research Gap')}</h4>
            <p class="text-sm text-gray-600 mt-2">${escapeHtml(gap.description || '')}</p>
            ${gap.potential_value ? `
                <p class="text-xs text-green-600 mt-2">
                    <strong>Potential Value:</strong> ${escapeHtml(gap.potential_value)}
                </p>
            ` : ''}
            ${gap.exploration_steps ? `
                <div class="mt-3">
                    <span class="text-xs font-medium text-gray-700">Exploration Steps:</span>
                    <ul class="text-xs text-gray-600 list-disc list-inside mt-1">
                        ${Array.isArray(gap.exploration_steps)
                            ? gap.exploration_steps.map(step => `<li>${escapeHtml(step)}</li>`).join('')
                            : `<li>${escapeHtml(gap.exploration_steps)}</li>`
                        }
                    </ul>
                </div>
            ` : ''}
        </div>
    `).join('');
}

// Render next steps
function renderNextSteps(steps) {
    const container = document.getElementById('next-steps');

    if (!steps || steps.length === 0) {
        container.innerHTML = '<div class="text-gray-500 col-span-full">No next steps suggested yet.</div>';
        return;
    }

    container.innerHTML = steps.map((step, i) => `
        <div class="bg-white rounded-lg shadow p-4 flex items-start space-x-4">
            <div class="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center font-bold">
                ${i + 1}
            </div>
            <div>
                <h4 class="font-medium text-gray-900">${escapeHtml(step.action || step.title || 'Action')}</h4>
                <p class="text-sm text-gray-600 mt-1">${escapeHtml(step.rationale || step.description || '')}</p>
                ${step.estimated_time ? `
                    <span class="text-xs text-gray-500 mt-2 inline-block">
                        ⏱️ ${escapeHtml(step.estimated_time)}
                    </span>
                ` : ''}
            </div>
        </div>
    `).join('');
}

// Utility function
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
