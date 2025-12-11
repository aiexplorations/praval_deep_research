// Configuration
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '/api';

// SSE Connection
let sseConnection = null;

// Message ID counter to ensure uniqueness
let messageCounter = 0;

// Current conversation state
let currentConversationId = null;
let currentMessages = [];  // Track messages with their metadata for branching

// Toast notification counter
let toastCounter = 0;

// Indexing state tracking
let indexingState = {
    isActive: false,
    totalPapers: 0,
    processedPapers: 0,
    currentPaper: null,
    stage: 'idle'
};

// ============================================
// TOAST NOTIFICATION SYSTEM
// ============================================

/**
 * Show a toast notification
 * @param {string} type - 'success', 'info', 'warning', 'error'
 * @param {string} title - Short title for the toast
 * @param {string} message - Detailed message
 * @param {number} duration - Auto-dismiss duration in ms (0 for no auto-dismiss)
 */
function showToast(type, title, message, duration = 5000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toastId = `toast-${++toastCounter}`;

    const icons = {
        success: `<svg class="toast-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
        </svg>`,
        info: `<svg class="toast-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>`,
        warning: `<svg class="toast-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
        </svg>`,
        error: `<svg class="toast-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
        </svg>`
    };

    const toastHtml = `
        <div id="${toastId}" class="toast toast-${type}">
            ${icons[type] || icons.info}
            <div class="toast-content">
                <div class="toast-title">${escapeHtml(title)}</div>
                ${message ? `<div class="toast-message">${escapeHtml(message)}</div>` : ''}
            </div>
            <span class="toast-close" onclick="dismissToast('${toastId}')">&times;</span>
        </div>
    `;

    container.insertAdjacentHTML('beforeend', toastHtml);

    // Auto-dismiss
    if (duration > 0) {
        setTimeout(() => dismissToast(toastId), duration);
    }

    return toastId;
}

/**
 * Dismiss a toast notification with animation
 */
function dismissToast(toastId) {
    const toast = document.getElementById(toastId);
    if (!toast) return;

    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 300);
}

// ============================================
// INDEXING PROGRESS TRACKING
// ============================================

/**
 * Update the indexing progress panel
 */
function updateIndexingProgress(stage, current, total, currentPaperTitle) {
    const panel = document.getElementById('indexing-progress');
    const stageEl = document.getElementById('indexing-stage');
    const countEl = document.getElementById('indexing-count');
    const progressBar = document.getElementById('indexing-progress-bar');
    const currentPaperEl = document.getElementById('indexing-current-paper');

    if (!panel) return;

    indexingState = {
        isActive: stage !== 'complete' && stage !== 'error',
        totalPapers: total,
        processedPapers: current,
        currentPaper: currentPaperTitle,
        stage: stage
    };

    if (stage === 'complete' || stage === 'error') {
        // Hide after brief delay
        setTimeout(() => panel.classList.add('hidden'), 2000);
        stageEl.textContent = stage === 'complete' ? 'Indexing complete!' : 'Indexing failed';
        progressBar.style.width = '100%';
        return;
    }

    panel.classList.remove('hidden');

    const stageLabels = {
        'starting': 'Starting indexing...',
        'downloading': 'Downloading PDFs...',
        'processing': 'Processing documents...',
        'embedding': 'Generating embeddings...',
        'summarizing': 'Creating summaries...',
        'extracting_citations': 'Extracting citations...',
        'indexing_linked': 'Indexing cited papers...'
    };

    stageEl.textContent = stageLabels[stage] || stage;
    countEl.textContent = `${current}/${total}`;
    progressBar.style.width = total > 0 ? `${(current / total) * 100}%` : '0%';
    currentPaperEl.textContent = currentPaperTitle ? `Processing: ${currentPaperTitle.substring(0, 60)}...` : 'Preparing...';
}

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

    // Handle different event types
    const eventType = data.event_type;

    switch (eventType) {
        case 'toast':
            // Show toast notification
            showToast(
                data.toast_type || 'info',
                data.title || 'Notification',
                data.message || '',
                data.duration || 5000
            );
            return;

        case 'indexing_progress':
            // Update indexing progress panel
            updateIndexingProgress(
                data.stage,
                data.current || 0,
                data.total || 0,
                data.current_paper
            );
            return;

        case 'paper_indexed':
            // Show toast for individual paper completion
            showToast(
                'success',
                'Paper Indexed',
                data.title ? `"${data.title.substring(0, 50)}..." added to knowledge base` : 'Paper added to knowledge base',
                4000
            );
            return;

        case 'linked_paper_indexed':
            // Show toast for linked/cited paper
            showToast(
                'info',
                'Cited Paper Indexed',
                data.title ? `"${data.title.substring(0, 50)}..." from citations` : 'Cited paper added',
                4000
            );
            return;

        case 'indexing_complete':
            // Show final completion toast
            updateIndexingProgress('complete', data.total, data.total, null);
            showToast(
                'success',
                'Indexing Complete',
                `${data.papers_indexed || 0} papers indexed with ${data.vectors_stored || 0} vectors`,
                6000
            );
            return;

        case 'indexing_error':
            // Show error toast
            updateIndexingProgress('error', 0, 0, null);
            showToast(
                'error',
                'Indexing Failed',
                data.error || 'An error occurred during indexing',
                8000
            );
            return;

        default:
            // Legacy handling for backward compatibility
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

// Ask a question (optionally as an edit/branch)
async function askQuestion(editMessageId = null, parentMessageId = null) {
    const questionInput = document.getElementById('question-input');
    const question = questionInput.value.trim();
    const askBtn = document.getElementById('ask-btn');

    if (!question) {
        alert('Please enter a question');
        return;
    }

    // Create conversation if needed
    if (!currentConversationId) {
        try {
            const convResponse = await fetch(`${API_BASE_URL}/research/conversations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: null })
            });
            const convData = await convResponse.json();
            currentConversationId = convData.id;
            console.log('Created conversation:', currentConversationId);
        } catch (e) {
            console.error('Failed to create conversation:', e);
        }
    }

    // If editing, handle branch creation
    if (editMessageId) {
        await handleEditMessage(editMessageId, question);
        return;
    }

    // Add user message to chat
    const userMsgId = addMessage('user', question);

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
                include_sources: true,
                conversation_id: currentConversationId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update conversation ID from response if provided
        if (data.conversation_id) {
            currentConversationId = data.conversation_id;
        }

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

// Handle editing a message (creates a new branch)
async function handleEditMessage(messageId, newContent) {
    const askBtn = document.getElementById('ask-btn');
    askBtn.disabled = true;
    askBtn.textContent = 'Creating branch...';

    try {
        // Call the edit endpoint to create a branch
        const response = await fetch(
            `${API_BASE_URL}/research/conversations/${currentConversationId}/messages/${messageId}/edit`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_content: newContent })
            }
        );

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const branchData = await response.json();
        console.log('Branch created:', branchData);

        // Refresh the conversation to show the new branch
        await loadConversation(currentConversationId);

        // Clear input
        document.getElementById('question-input').value = '';

        // Now generate a response for the edited message
        const loadingId = addMessage('assistant', 'Generating response for edited message...', true);

        const qaResponse = await fetch(`${API_BASE_URL}/research/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: newContent,
                include_sources: true,
                conversation_id: currentConversationId
            })
        });

        document.getElementById(loadingId)?.remove();

        if (qaResponse.ok) {
            const qaData = await qaResponse.json();
            addMessage('assistant', qaData.answer, false, qaData.sources, qaData.followup_questions);
            displayFollowupQuestions(qaData.followup_questions || []);
        }

    } catch (error) {
        console.error('Edit message error:', error);
        alert(`Failed to edit message: ${error.message}`);
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask';
    }
}

// Load a conversation and its messages
async function loadConversation(conversationId) {
    try {
        const response = await fetch(`${API_BASE_URL}/research/conversations/${conversationId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        currentConversationId = data.id;
        currentMessages = data.messages || [];

        // Clear chat and re-render messages
        const chatMessages = document.getElementById('chat-messages');
        chatMessages.innerHTML = '';

        // Add welcome message first
        addMessage('assistant', 'Welcome! Search for research papers above, then ask me questions about them. I\'ll use semantic search to find relevant content and provide detailed answers with source citations.');

        // Render all messages from the conversation
        for (const msg of currentMessages) {
            addMessage(msg.role, msg.content, false, msg.sources, null, msg);
        }

        console.log('Loaded conversation:', conversationId, 'with', currentMessages.length, 'messages');

    } catch (error) {
        console.error('Failed to load conversation:', error);
    }
}

// Switch to a different branch
async function switchBranch(messageId, direction) {
    try {
        const response = await fetch(
            `${API_BASE_URL}/research/conversations/${currentConversationId}/switch-branch`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message_id: messageId,
                    direction: direction
                })
            }
        );

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Switched branch:', data);

        // Refresh conversation to show new branch
        await loadConversation(currentConversationId);

    } catch (error) {
        console.error('Failed to switch branch:', error);
        alert(`Failed to switch branch: ${error.message}`);
    }
}

// Start editing a message
function startEditMessage(messageId, originalContent) {
    const questionInput = document.getElementById('question-input');
    questionInput.value = originalContent;
    questionInput.focus();

    // Store the editing state
    questionInput.dataset.editingMessageId = messageId;

    // Update button text
    const askBtn = document.getElementById('ask-btn');
    askBtn.textContent = 'Resubmit';
    askBtn.onclick = () => {
        const editId = questionInput.dataset.editingMessageId;
        delete questionInput.dataset.editingMessageId;
        askBtn.textContent = 'Ask';
        askBtn.onclick = () => askQuestion();
        askQuestion(editId);
    };

    // Show cancel option
    showEditCancelOption();
}

// Show cancel edit option
function showEditCancelOption() {
    const inputContainer = document.getElementById('question-input').parentElement;
    if (!document.getElementById('cancel-edit-btn')) {
        const cancelBtn = document.createElement('button');
        cancelBtn.id = 'cancel-edit-btn';
        cancelBtn.className = 'px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.onclick = cancelEditMessage;
        inputContainer.appendChild(cancelBtn);
    }
}

// Cancel editing
function cancelEditMessage() {
    const questionInput = document.getElementById('question-input');
    questionInput.value = '';
    delete questionInput.dataset.editingMessageId;

    const askBtn = document.getElementById('ask-btn');
    askBtn.textContent = 'Ask';
    askBtn.onclick = () => askQuestion();

    const cancelBtn = document.getElementById('cancel-edit-btn');
    if (cancelBtn) cancelBtn.remove();
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
async function copyWithCitations(messageId) {
    const messageElement = document.getElementById(messageId);
    const copyButton = messageElement.querySelector('.copy-button');

    try {
        // Get content and sources from data attributes
        const content = messageElement.dataset.content || '';
        const sourcesJson = messageElement.dataset.sources || '[]';
        const sources = JSON.parse(sourcesJson);

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
function addMessage(role, content, isLoading = false, sources = null, followups = null, messageMetadata = null) {
    const chatMessages = document.getElementById('chat-messages');
    // Use server ID if available, otherwise generate client-side ID
    const messageId = messageMetadata?.id || `msg-${Date.now()}-${messageCounter++}`;

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

    // Build action buttons based on role and state
    let actionsHtml = '';

    // Edit button for user messages (not loading)
    if (role === 'user' && !isLoading && messageMetadata?.id) {
        actionsHtml += `
            <button
                onclick="startEditMessage('${messageMetadata.id}', '${escapeHtml(content).replace(/'/g, "\\'")}')"
                class="edit-button text-xs px-2 py-1 rounded bg-blue-100 text-blue-700 hover:bg-blue-200 transition-colors opacity-0 group-hover:opacity-100"
                title="Edit & resubmit"
            >
                ✏️ Edit
            </button>
        `;
    }

    // Branch navigation for messages with siblings (< 1/3 > style)
    if (messageMetadata && messageMetadata.sibling_count > 1) {
        const currentIndex = messageMetadata.sibling_index + 1;
        const total = messageMetadata.sibling_count;
        actionsHtml += `
            <div class="branch-nav flex items-center space-x-1 text-xs">
                <button
                    onclick="switchBranch('${messageMetadata.id}', 'left')"
                    class="px-2 py-1 rounded ${currentIndex <= 1 ? 'text-gray-300 cursor-not-allowed' : 'text-gray-600 hover:bg-gray-100'}"
                    ${currentIndex <= 1 ? 'disabled' : ''}
                    title="Previous version"
                >
                    ◀
                </button>
                <span class="text-gray-500">${currentIndex}/${total}</span>
                <button
                    onclick="switchBranch('${messageMetadata.id}', 'right')"
                    class="px-2 py-1 rounded ${currentIndex >= total ? 'text-gray-300 cursor-not-allowed' : 'text-gray-600 hover:bg-gray-100'}"
                    ${currentIndex >= total ? 'disabled' : ''}
                    title="Next version"
                >
                    ▶
                </button>
            </div>
        `;
    }

    // Add copy button for assistant messages (but not loading messages)
    if (role === 'assistant' && !isLoading) {
        actionsHtml += `
            <button
                onclick="copyWithCitations('${messageId}')"
                class="copy-button text-xs px-3 py-1 rounded bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
                title="Copy answer with BibTeX citations"
            >
                📋 Copy with Citations
            </button>
        `;
    }

    // Wrap actions in a flex container
    const actionsContainerHtml = actionsHtml ? `
        <div class="flex items-center justify-between mt-2 space-x-2">
            ${actionsHtml}
        </div>
    ` : '';

    const messageHtml = `
        <div id="${messageId}" class="flex ${alignment} group" data-content="${escapeHtml(content)}" data-sources="${escapeHtml(JSON.stringify(sources || []))}" data-message-id="${messageMetadata?.id || ''}">
            <div class="${messageClass} rounded-lg px-4 py-3 max-w-2xl ${isLoading ? 'animate-pulse' : ''}">
                ${contentHtml}
                ${sourcesHtml}
                ${actionsContainerHtml}
            </div>
        </div>
    `;

    chatMessages.insertAdjacentHTML('beforeend', messageHtml);

    // Track message in currentMessages if it has server metadata
    if (messageMetadata) {
        // Already in currentMessages from loadConversation, don't duplicate
    }

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
