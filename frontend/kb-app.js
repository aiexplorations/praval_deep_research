// Configuration
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '/api';

// Load papers on page load
window.addEventListener('DOMContentLoaded', () => {
    loadPapers();
    loadStats();
});

// Load all papers
async function loadPapers() {
    const loadingState = document.getElementById('loading-state');
    const emptyState = document.getElementById('empty-state');
    const papersTable = document.getElementById('papers-table');

    loadingState.classList.remove('hidden');
    papersTable.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE_URL}/research/knowledge-base/papers`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        loadingState.classList.add('hidden');

        if (!data.papers || data.papers.length === 0) {
            emptyState.classList.remove('hidden');
            return;
        }

        emptyState.classList.add('hidden');

        // Update stats
        document.getElementById('total-papers').textContent = data.total_papers;
        document.getElementById('total-vectors').textContent = data.total_vectors;

        // Render papers
        papersTable.innerHTML = data.papers.map(paper => `
            <tr class="paper-row">
                <td class="px-6 py-4">
                    <div class="text-sm font-medium text-gray-900">${escapeHtml(paper.title)}</div>
                    <div class="text-xs text-gray-500 mt-1">
                        ${paper.authors ? paper.authors.slice(0, 3).join(', ') : 'Unknown authors'}
                        ${paper.authors && paper.authors.length > 3 ? ' +' + (paper.authors.length - 3) + ' more' : ''}
                    </div>
                </td>
                <td class="px-6 py-4 text-sm text-gray-500">
                    ${escapeHtml(paper.paper_id)}
                </td>
                <td class="px-6 py-4 text-sm text-gray-500">
                    <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded">${paper.chunk_count}</span>
                </td>
                <td class="px-6 py-4 text-xs text-gray-500">
                    ${paper.categories ? paper.categories.slice(0, 2).join(', ') : 'N/A'}
                </td>
                <td class="px-6 py-4 text-right text-sm">
                    <button
                        onclick="deletePaper('${escapeHtml(paper.paper_id)}', '${escapeHtml(paper.title)}')"
                        class="text-red-600 hover:text-red-900"
                    >
                        🗑️ Delete
                    </button>
                </td>
            </tr>
        `).join('');

    } catch (error) {
        console.error('Failed to load papers:', error);
        loadingState.classList.add('hidden');
        papersTable.innerHTML = `
            <tr>
                <td colspan="5" class="px-6 py-4 text-center text-red-600">
                    Failed to load papers: ${error.message}
                </td>
            </tr>
        `;
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/research/knowledge-base/stats`);
        const data = await response.json();

        document.getElementById('total-papers').textContent = data.total_papers;
        document.getElementById('total-vectors').textContent = data.total_vectors;
        document.getElementById('avg-chunks').textContent = data.avg_chunks_per_paper.toFixed(1);

    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Delete a paper
async function deletePaper(paperId, title) {
    if (!confirm(`Are you sure you want to delete "${title}"?\n\nThis will remove all vectors for this paper from the knowledge base. This action cannot be undone.`)) {
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
        loadStats();

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
        loadStats();

    } catch (error) {
        console.error('Failed to clear knowledge base:', error);
        alert(`Failed to clear knowledge base: ${error.message}`);
    }
}

// Utility function
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
