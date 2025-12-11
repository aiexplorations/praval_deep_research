// Configuration
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '/api';

// State
let currentPage = 1;
let pageSize = 20;
let totalPages = 1;
let searchTimeout = null;
let availableCategories = [];

// Load papers on page load
window.addEventListener('DOMContentLoaded', () => {
    loadPapers();
});

// Debounce search input
function debounceSearch() {
    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        currentPage = 1;
        loadPapers();
    }, 300);
}

// Apply filters and reload
function applyFilters() {
    currentPage = 1;
    loadPapers();
}

// Clear all filters
function clearFilters() {
    document.getElementById('search-input').value = '';
    document.getElementById('category-filter').value = '';
    document.getElementById('source-filter').value = '';
    document.getElementById('sort-select').value = 'title-asc';
    currentPage = 1;
    loadPapers();
}

// Build query parameters from current filters
function buildQueryParams() {
    const params = new URLSearchParams();

    const search = document.getElementById('search-input').value.trim();
    if (search) params.set('search', search);

    const category = document.getElementById('category-filter').value;
    if (category) params.set('category', category);

    const source = document.getElementById('source-filter').value;
    if (source) params.set('source', source);

    const sortValue = document.getElementById('sort-select').value;
    const [sort, order] = sortValue.split('-');
    params.set('sort', sort);
    params.set('sort_order', order);

    params.set('page', currentPage);
    params.set('page_size', pageSize);

    return params.toString();
}

// Load all papers with filters
async function loadPapers() {
    const loadingState = document.getElementById('loading-state');
    const emptyState = document.getElementById('empty-state');
    const papersTable = document.getElementById('papers-table');
    const pagination = document.getElementById('pagination');

    loadingState.classList.remove('hidden');
    emptyState.classList.add('hidden');
    pagination.classList.add('hidden');
    papersTable.innerHTML = '';

    try {
        const queryParams = buildQueryParams();
        const response = await fetch(`${API_BASE_URL}/research/knowledge-base/papers?${queryParams}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        loadingState.classList.add('hidden');

        // Update stats
        document.getElementById('total-papers').textContent = data.total_papers;
        document.getElementById('total-vectors').textContent = data.total_vectors;

        // Count KB vs linked papers (from filtered results)
        const kbCount = data.papers.filter(p => !p.is_linked).length;
        const linkedCount = data.papers.filter(p => p.is_linked).length;
        document.getElementById('kb-papers').textContent = kbCount;
        document.getElementById('linked-papers').textContent = linkedCount;

        // Update category filter options
        if (data.available_categories && data.available_categories.length > 0) {
            updateCategoryFilter(data.available_categories);
        }

        // Update pagination
        totalPages = data.total_pages || 1;
        currentPage = data.page || 1;
        updatePagination(data);

        if (!data.papers || data.papers.length === 0) {
            emptyState.classList.remove('hidden');
            return;
        }

        // Render papers
        papersTable.innerHTML = data.papers.map(paper => `
            <tr class="paper-row">
                <td class="px-6 py-4">
                    <div class="text-sm font-medium text-gray-900">${escapeHtml(paper.title)}</div>
                    <div class="text-xs text-gray-500 mt-1">
                        ${paper.authors ? paper.authors.slice(0, 3).join(', ') : 'Unknown authors'}
                        ${paper.authors && paper.authors.length > 3 ? ' +' + (paper.authors.length - 3) + ' more' : ''}
                    </div>
                    ${paper.source_paper_id ? `<div class="text-xs text-amber-600 mt-1">Cited by: ${paper.source_paper_id}</div>` : ''}
                </td>
                <td class="px-6 py-4">
                    ${paper.is_linked
                        ? '<span class="px-2 py-1 text-xs font-medium rounded linked-badge">Linked</span>'
                        : '<span class="px-2 py-1 text-xs font-medium rounded kb-badge">KB</span>'
                    }
                </td>
                <td class="px-6 py-4 text-sm text-gray-500">
                    ${escapeHtml(paper.paper_id)}
                </td>
                <td class="px-6 py-4 text-sm text-gray-500">
                    <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded">${paper.chunk_count}</span>
                </td>
                <td class="px-6 py-4 text-xs text-gray-500">
                    ${paper.categories ? paper.categories.slice(0, 2).map(c =>
                        `<span class="inline-block px-2 py-1 mr-1 mb-1 bg-gray-100 rounded">${c}</span>`
                    ).join('') : 'N/A'}
                </td>
                <td class="px-6 py-4 text-right text-sm">
                    <button
                        onclick="deletePaper('${escapeHtml(paper.paper_id)}', '${escapeHtml(paper.title)}', ${paper.is_linked || false})"
                        class="text-red-600 hover:text-red-900"
                    >
                        🗑️ Delete
                    </button>
                </td>
            </tr>
        `).join('');

        // Show active filters
        updateActiveFiltersDisplay(data.filters_applied);

    } catch (error) {
        console.error('Failed to load papers:', error);
        loadingState.classList.add('hidden');
        papersTable.innerHTML = `
            <tr>
                <td colspan="6" class="px-6 py-4 text-center text-red-600">
                    Failed to load papers: ${error.message}
                </td>
            </tr>
        `;
    }
}

// Update category filter dropdown
function updateCategoryFilter(categories) {
    const select = document.getElementById('category-filter');
    const currentValue = select.value;

    // Only update if categories changed
    if (JSON.stringify(categories) === JSON.stringify(availableCategories)) return;
    availableCategories = categories;

    // Keep first option, replace rest
    select.innerHTML = '<option value="">All Categories</option>';
    categories.forEach(cat => {
        const option = document.createElement('option');
        option.value = cat;
        option.textContent = cat;
        select.appendChild(option);
    });

    // Restore previous selection if still valid
    if (categories.includes(currentValue)) {
        select.value = currentValue;
    }
}

// Update pagination controls
function updatePagination(data) {
    const pagination = document.getElementById('pagination');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    if (data.total_papers > pageSize) {
        pagination.classList.remove('hidden');

        document.getElementById('showing-start').textContent = ((data.page - 1) * pageSize) + 1;
        document.getElementById('showing-end').textContent = Math.min(data.page * pageSize, data.total_papers);
        document.getElementById('showing-total').textContent = data.total_papers;
        document.getElementById('current-page').textContent = data.page;
        document.getElementById('total-pages').textContent = data.total_pages;

        prevBtn.disabled = data.page <= 1;
        nextBtn.disabled = data.page >= data.total_pages;
    } else {
        pagination.classList.add('hidden');
    }
}

// Update active filters display
function updateActiveFiltersDisplay(filters) {
    const container = document.getElementById('active-filters');
    if (!filters) return;

    const hasActiveFilters = filters.search || filters.category || filters.source;

    if (hasActiveFilters) {
        container.classList.remove('hidden');
        let filtersHtml = '<span class="text-sm text-gray-600">Active filters:</span>';

        if (filters.search) {
            filtersHtml += `<span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">Search: "${filters.search}"</span>`;
        }
        if (filters.category) {
            filtersHtml += `<span class="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">Category: ${filters.category}</span>`;
        }
        if (filters.source) {
            filtersHtml += `<span class="px-2 py-1 bg-amber-100 text-amber-800 rounded text-sm">Source: ${filters.source === 'kb' ? 'KB Papers' : 'Linked Papers'}</span>`;
        }

        container.innerHTML = filtersHtml;
    } else {
        container.classList.add('hidden');
    }
}

// Pagination navigation
function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        loadPapers();
    }
}

function nextPage() {
    if (currentPage < totalPages) {
        currentPage++;
        loadPapers();
    }
}

// Delete a paper
async function deletePaper(paperId, title, isLinked) {
    const sourceInfo = isLinked ? ' (linked paper)' : '';
    if (!confirm(`Are you sure you want to delete "${title}"${sourceInfo}?\n\nThis will remove all vectors for this paper. This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/research/knowledge-base/papers/${encodeURIComponent(paperId)}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        alert(`Successfully deleted "${title}" (${data.vectors_deleted} vectors removed)`);

        // Reload papers
        loadPapers();

    } catch (error) {
        console.error('Failed to delete paper:', error);
        alert(`Failed to delete paper: ${error.message}`);
    }
}

// Clear entire knowledge base
async function clearKnowledgeBase() {
    if (!confirm('⚠️ WARNING: This will delete ALL papers and vectors from your knowledge base!\n\nThis action cannot be undone. Are you absolutely sure?')) {
        return;
    }

    if (!confirm('Final confirmation: Delete everything?')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/research/knowledge-base/clear`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        alert('Knowledge base cleared successfully!');

        // Reload
        loadPapers();

    } catch (error) {
        console.error('Failed to clear knowledge base:', error);
        alert(`Failed to clear knowledge base: ${error.message}`);
    }
}

// Utility function
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
