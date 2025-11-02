// Configuration
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '/api';

// SSE Connection
let sseConnection = null;

// Connect to SSE for real-time agent updates
function connectSSE() {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    try {
        sseConnection = new EventSource(`${API_BASE_URL}/sse/agent-updates`);

        sseConnection.onopen = () => {
            console.log('SSE Connected');
            statusIndicator.className = 'h-2 w-2 rounded-full bg-green-500';
            statusText.textContent = 'Connected';
        };

        sseConnection.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleAgentUpdate(data);
            } catch (e) {
                console.error('Failed to parse SSE message:', e);
            }
        };

        sseConnection.onerror = () => {
            console.error('SSE Error');
            statusIndicator.className = 'h-2 w-2 rounded-full bg-red-500';
            statusText.textContent = 'Disconnected';

            // Attempt reconnect after 5 seconds
            setTimeout(connectSSE, 5000);
        };

    } catch (e) {
        console.error('Failed to connect SSE:', e);
        statusIndicator.className = 'h-2 w-2 rounded-full bg-yellow-500';
        statusText.textContent = 'SSE Not Available';
    }
}

// Handle agent status updates
function handleAgentUpdate(data) {
    const agentStatus = document.getElementById('agent-status');
    const agentStatusText = document.getElementById('agent-status-text');
    const agentDetails = document.getElementById('agent-details');

    agentStatus.classList.remove('hidden');
    agentStatusText.textContent = data.message || 'Processing...';

    if (data.details) {
        agentDetails.textContent = data.details;
    }

    // Auto-hide after completion
    if (data.status === 'complete') {
        setTimeout(() => {
            agentStatus.classList.add('hidden');
        }, 3000);
    }
}

// Search for papers
async function searchPapers() {
    const query = document.getElementById('search-query').value.trim();
    const domain = document.getElementById('search-domain').value;
    const searchBtn = document.getElementById('search-btn');
    const papersContainer = document.getElementById('papers-container');

    if (!query) {
        alert('Please enter a search query');
        return;
    }

    // Show loading state
    searchBtn.disabled = true;
    searchBtn.textContent = 'Searching...';
    papersContainer.innerHTML = '<div class="col-span-2 text-center text-gray-600">Searching papers and processing with AI agents...</div>';

    // Show agent status
    const agentStatus = document.getElementById('agent-status');
    agentStatus.classList.remove('hidden');
    document.getElementById('agent-status-text').textContent = 'Searching ArXiv...';

    try {
        const response = await fetch(`${API_BASE_URL}/research/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                domain: domain.replace(/_/g, ' '),  // Convert underscores to spaces
                max_results: 10,
                quality_threshold: 0.3  // Lowered from 0.7 to get more results
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Display papers
        displayPapers(data.papers);

        // Update status with more detail
        const statusText = data.total_found > 0
            ? `Found ${data.total_found} papers in ${data.search_time_ms}ms. ${data.optimization_applied ? 'Query optimized. ' : ''}Agents processing documents...`
            : 'No papers found matching your query. Try different search terms.';
        document.getElementById('agent-status-text').textContent = statusText;

        // Log metadata
        console.log('Search complete:', data);

    } catch (error) {
        console.error('Search error:', error);
        papersContainer.innerHTML = `
            <div class="col-span-2 text-center text-red-600">
                <p class="font-medium">Error searching papers</p>
                <p class="text-sm mt-2">${error.message}</p>
            </div>
        `;
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = 'Search';
    }
}

// Store current papers for indexing
let currentPapers = [];

// Display papers in the UI with selection checkboxes
function displayPapers(papers) {
    const container = document.getElementById('papers-container');
    currentPapers = papers || [];

    if (!papers || papers.length === 0) {
        container.innerHTML = '<div class="col-span-2 text-center text-gray-600">No papers found</div>';
        document.getElementById('index-button-container').classList.add('hidden');
        return;
    }

    // Show index button container
    document.getElementById('index-button-container').classList.remove('hidden');
    updateIndexButtonState();

    container.innerHTML = papers.map((paper, index) => `
        <div class="paper-card bg-white border border-gray-200 rounded-lg p-4 transition-all duration-200">
            <div class="flex items-start space-x-3">
                <input
                    type="checkbox"
                    id="paper-${index}"
                    class="paper-checkbox mt-1 h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                    onchange="updateIndexButtonState()"
                />
                <div class="flex-1">
                    <div class="flex items-start justify-between mb-2">
                        <label for="paper-${index}" class="cursor-pointer flex-1">
                            <h3 class="text-lg font-semibold text-gray-900">
                                ${escapeHtml(paper.title)}
                            </h3>
                        </label>
                        ${paper.relevance_score ?
                            `<span class="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded">${(paper.relevance_score * 100).toFixed(0)}%</span>`
                            : ''}
                    </div>

                    <p class="text-sm text-gray-700 mb-3">${escapeHtml(paper.abstract.substring(0, 300))}...</p>

                    <div class="flex items-center justify-between text-sm">
                        <div class="text-gray-600">
                            <span class="font-medium">${paper.authors.slice(0, 3).join(', ')}</span>
                            ${paper.authors.length > 3 ? ` +${paper.authors.length - 3} more` : ''}
                        </div>

                        <div class="flex space-x-2">
                            ${paper.url ?
                                `<a href="${paper.url}" target="_blank" class="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors">
                                    ArXiv
                                </a>`
                                : ''}
                            ${paper.categories ?
                                `<span class="px-3 py-1 bg-gray-100 text-gray-700 rounded text-xs">${paper.categories[0]}</span>`
                                : ''}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Update index button state based on selections
function updateIndexButtonState() {
    const checkboxes = document.querySelectorAll('.paper-checkbox');
    const checkedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
    const btn = document.getElementById('index-selected-btn');

    if (checkedCount > 0) {
        btn.disabled = false;
        btn.textContent = `Index ${checkedCount} Selected Paper${checkedCount > 1 ? 's' : ''}`;
        btn.classList.remove('opacity-50', 'cursor-not-allowed');
    } else {
        btn.disabled = true;
        btn.textContent = 'Select Papers to Index';
        btn.classList.add('opacity-50', 'cursor-not-allowed');
    }
}

// Index selected papers
async function indexSelectedPapers() {
    const checkboxes = document.querySelectorAll('.paper-checkbox');
    const selectedIndices = Array.from(checkboxes)
        .map((cb, idx) => cb.checked ? idx : -1)
        .filter(idx => idx >= 0);

    if (selectedIndices.length === 0) {
        alert('Please select at least one paper to index');
        return;
    }

    const selectedPapers = selectedIndices.map(idx => currentPapers[idx]);
    const btn = document.getElementById('index-selected-btn');
    const originalText = btn.textContent;

    btn.disabled = true;
    btn.textContent = 'Indexing...';

    try {
        const response = await fetch(`${API_BASE_URL}/research/index`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                papers: selectedPapers
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update status
        document.getElementById('agent-status-text').textContent =
            `Successfully indexed ${data.indexed_count} papers! ${data.vectors_stored} vectors stored. Ready for Q&A.`;

        // Uncheck all boxes
        checkboxes.forEach(cb => cb.checked = false);
        updateIndexButtonState();

        console.log('Indexing complete:', data);

    } catch (error) {
        console.error('Indexing error:', error);
        alert(`Error indexing papers: ${error.message}`);
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

// Ask a question
async function askQuestion() {
    const questionInput = document.getElementById('question-input');
    const question = questionInput.value.trim();
    const askBtn = document.getElementById('ask-btn');

    if (!question) {
        alert('Please enter a question');
        return;
    }

    // Add user message to chat
    addMessage('user', question);

    // Clear input
    questionInput.value = '';

    // Show loading state
    askBtn.disabled = true;
    askBtn.textContent = 'Processing...';

    // Add loading message
    const loadingId = addMessage('assistant', 'Searching research papers and generating answer...', true);

    try {
        const response = await fetch(`${API_BASE_URL}/research/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                include_sources: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Remove loading message
        document.getElementById(loadingId)?.remove();

        // Add assistant answer
        addMessage('assistant', data.answer, false, data.sources, data.followup_questions);

        // Display follow-up questions
        displayFollowupQuestions(data.followup_questions || []);

        console.log('Q&A complete:', data);

    } catch (error) {
        console.error('Q&A error:', error);
        document.getElementById(loadingId)?.remove();
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask';
    }
}

// Render Markdown with LaTeX support
function renderMarkdown(text) {
    // First, render markdown
    const html = marked.parse(text, {
        breaks: true,
        gfm: true
    });

    return html;
}

// Render LaTeX equations in an element
function renderLatex(element) {
    // Auto-render LaTeX using KaTeX
    // Supports both inline ($...$) and display ($$...$$) math
    renderMathInElement(element, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false},
            {left: '\\[', right: '\\]', display: true},
            {left: '\\(', right: '\\)', display: false}
        ],
        throwOnError: false,
        errorColor: '#cc0000'
    });
}

// Format sources as BibTeX citations
function formatBibTeX(sources) {
    if (!sources || sources.length === 0) {
        return '';
    }

    const bibtexEntries = sources.map((source, index) => {
        // Generate a citation key from the title
        const citationKey = source.title
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '_')
            .substring(0, 30)
            + (source.paper_id || index);

        // Extract year from paper_id if available (ArXiv format: YYMM.NNNNN)
        let year = 'unknown';
        if (source.paper_id) {
            const match = source.paper_id.match(/^(\d{2})(\d{2})/);
            if (match) {
                const yy = parseInt(match[1]);
                year = (yy > 90 ? '19' : '20') + match[1];
            }
        }

        return `@article{${citationKey},
  title={${source.title}},
  year={${year}},
  note={ArXiv ID: ${source.paper_id || 'unknown'}, Relevance: ${(source.relevance_score * 100).toFixed(0)}\\%}
}`;
    });

    return bibtexEntries.join('\n\n');
}

// Copy message with citations to clipboard
async function copyWithCitations(messageId, content, sources) {
    const copyButton = document.querySelector(`#${messageId} .copy-button`);

    try {
        // Format the content with citations
        let textToCopy = content + '\n\n';

        if (sources && sources.length > 0) {
            textToCopy += '='.repeat(60) + '\n';
            textToCopy += 'REFERENCES (BibTeX format)\n';
            textToCopy += '='.repeat(60) + '\n\n';
            textToCopy += formatBibTeX(sources);
        }

        // Copy to clipboard
        await navigator.clipboard.writeText(textToCopy);

        // Visual feedback
        const originalHTML = copyButton.innerHTML;
        copyButton.innerHTML = '✓ Copied!';
        copyButton.classList.add('copied');

        setTimeout(() => {
            copyButton.innerHTML = originalHTML;
            copyButton.classList.remove('copied');
        }, 2000);

    } catch (error) {
        console.error('Failed to copy:', error);
        copyButton.innerHTML = '✗ Failed';
        setTimeout(() => {
            copyButton.innerHTML = '📋 Copy with Citations';
        }, 2000);
    }
}

// Add message to chat
function addMessage(role, content, isLoading = false, sources = null, followups = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageId = `msg-${Date.now()}`;

    const messageClass = role === 'user' ? 'message-user' : 'message-assistant';
    const alignment = role === 'user' ? 'justify-end' : 'justify-start';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="mt-3 pt-3 border-t border-gray-300">
                <p class="text-xs font-semibold text-gray-700 mb-2">Sources:</p>
                ${sources.map(source => `
                    <div class="source-citation bg-gray-50 p-2 mb-2 rounded text-xs">
                        <p class="font-medium text-gray-900">${escapeHtml(source.title)}</p>
                        <p class="text-gray-600 mt-1">${escapeHtml(source.excerpt || '')}</p>
                        <p class="text-gray-500 mt-1">Relevance: ${(source.relevance_score * 100).toFixed(0)}%</p>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // For user messages, use plain text. For assistant messages, render Markdown
    const contentHtml = role === 'user'
        ? `<p class="text-sm whitespace-pre-wrap">${escapeHtml(content)}</p>`
        : `<div class="text-sm markdown-content">${renderMarkdown(content)}</div>`;

    // Add copy button for assistant messages (but not loading messages)
    const copyButtonHtml = (role === 'assistant' && !isLoading) ? `
        <div class="flex justify-end mt-2">
            <button
                onclick="copyWithCitations('${messageId}', \`${escapeHtml(content).replace(/`/g, '\\`')}\`, ${JSON.stringify(sources || []).replace(/"/g, '&quot;')})"
                class="copy-button text-xs px-3 py-1 rounded bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
                title="Copy answer with BibTeX citations"
            >
                📋 Copy with Citations
            </button>
        </div>
    ` : '';

    const messageHtml = `
        <div id="${messageId}" class="flex ${alignment}">
            <div class="${messageClass} rounded-lg px-4 py-3 max-w-2xl ${isLoading ? 'animate-pulse' : ''}">
                ${contentHtml}
                ${sourcesHtml}
                ${copyButtonHtml}
            </div>
        </div>
    `;

    chatMessages.insertAdjacentHTML('beforeend', messageHtml);

    // If this is an assistant message, render LaTeX equations
    if (role === 'assistant' && !isLoading) {
        const messageElement = document.getElementById(messageId);
        if (messageElement) {
            const markdownContent = messageElement.querySelector('.markdown-content');
            if (markdownContent) {
                renderLatex(markdownContent);
            }
        }
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageId;
}

// Display follow-up questions
function displayFollowupQuestions(questions) {
    const container = document.getElementById('followup-questions');

    if (!questions || questions.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = `
        <p class="text-sm font-medium text-gray-700 mb-2">Suggested follow-up questions:</p>
        <div class="space-y-2">
            ${questions.map(q => `
                <button
                    onclick="askFollowup('${escapeHtml(q).replace(/'/g, "\\'")}')"
                    class="block w-full text-left px-4 py-2 bg-blue-50 text-blue-900 rounded-lg hover:bg-blue-100 transition-colors text-sm"
                >
                    ${escapeHtml(q)}
                </button>
            `).join('')}
        </div>
    `;
}

// Ask a follow-up question
function askFollowup(question) {
    const questionInput = document.getElementById('question-input');
    questionInput.value = question;
    askQuestion();
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize app
function init() {
    console.log('Initializing Praval Deep Research...');
    connectSSE();

    // Add welcome message
    addMessage('assistant', 'Welcome! Search for research papers above, then ask me questions about them. I\'ll use semantic search to find relevant content and provide detailed answers with source citations.');
}

// Start the app when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
