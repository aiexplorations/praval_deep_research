/**
 * Chat Page - Research Q&A Interface
 *
 * Ask questions about indexed papers with semantic search and source citations.
 */

import { useState, useRef, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { apiClient } from '../services/api/client';
import { useChatStore } from '../store/useChatStore';
import type { Message, QuestionRequest } from '../types';

export default function Chat() {
  const [question, setQuestion] = useState('');
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [showSidebar, setShowSidebar] = useState(true);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { currentConversationId, setCurrentConversation, messages, addMessage, clearMessages } = useChatStore();

  // Load conversations list
  const { data: conversationsData, refetch: refetchConversations } = useQuery({
    queryKey: ['conversations'],
    queryFn: async () => {
      const response = await apiClient.listConversations();
      return response.conversations;
    }
  });

  // Create new conversation
  const createConversationMutation = useMutation({
    mutationFn: (title?: string) => apiClient.createConversation(title),
    onSuccess: (data) => {
      console.log('New conversation created:', data);
      setCurrentConversation(data.id);
      clearMessages();
      refetchConversations();
    }
  });

  // Load conversation
  const loadConversationMutation = useMutation({
    mutationFn: (id: string) => apiClient.getConversation(id),
    onSuccess: (data) => {
      console.log('Loaded conversation:', data.id, 'with', data.messages?.length, 'messages');
      setCurrentConversation(data.id);
      clearMessages();
      // Load messages from conversation with branching info
      if (data.messages) {
        data.messages.forEach((msg: any) => {
          addMessage({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            sources: msg.sources,
            timestamp: msg.timestamp,
            parent_message_id: msg.parent_message_id,
            branch_id: msg.branch_id,
            branch_index: msg.branch_index,
            has_branches: msg.has_branches,
            sibling_count: msg.sibling_count,
            sibling_index: msg.sibling_index
          });
        });
      }
    }
  });

  // Delete conversation
  const deleteConversationMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteConversation(id),
    onSuccess: (_, deletedId) => {
      console.log('Deleted conversation:', deletedId);
      // If we deleted the current conversation, clear messages and reset ID
      if (currentConversationId === deletedId) {
        clearMessages();
        setCurrentConversation(null);
      }
      refetchConversations();
    }
  });

  const handleNewChat = async () => {
    // Create new conversation - previous messages are auto-saved
    createConversationMutation.mutate(undefined);
  };

  const handleLoadConversation = (convId: string) => {
    // Load conversation immediately - messages are auto-saved
    loadConversationMutation.mutate(convId);
  };

  const handleDeleteConversation = (convId: string, title: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent loading conversation when clicking delete
    if (confirm(`Delete conversation "${title}"? This cannot be undone.`)) {
      deleteConversationMutation.mutate(convId);
    }
  };

  const toggleSource = (messageId: string, sourceIndex: number) => {
    const key = `${messageId}-${sourceIndex}`;
    const newExpanded = new Set(expandedSources);
    if (newExpanded.has(key)) {
      newExpanded.delete(key);
    } else {
      newExpanded.add(key);
    }
    setExpandedSources(newExpanded);
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-load most recent conversation on initial page load
  useEffect(() => {
    if (conversationsData && conversationsData.length > 0 && !currentConversationId && messages.length === 0) {
      // Load the most recent conversation
      const mostRecent = conversationsData[0];
      console.log('Auto-loading most recent conversation:', mostRecent.id);
      loadConversationMutation.mutate(mostRecent.id);
    }
  }, [conversationsData, currentConversationId, messages.length]);

  // Q&A mutation
  const askMutation = useMutation({
    mutationFn: (request: QuestionRequest) => apiClient.askQuestion(request),
    onSuccess: (data) => {
      const assistantMessage: Message = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        timestamp: new Date().toISOString()
      };
      addMessage(assistantMessage);
      // Refresh conversations list to update titles/timestamps
      refetchConversations();
      // Reload conversation to get server-side message IDs for branching
      if (currentConversationId) {
        setTimeout(() => {
          loadConversationMutation.mutate(currentConversationId);
        }, 500);
      }
    },
    onError: (error: any) => {
      const errorMessage: Message = {
        id: `msg-${Date.now()}-error`,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message || 'Unknown error'}`,
        timestamp: new Date().toISOString()
      };
      addMessage(errorMessage);
    }
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || askMutation.isPending) return;

    // Ensure we have a conversation ID - create one if needed
    let convId = currentConversationId;
    if (!convId) {
      console.log('No conversation ID, creating new conversation...');
      try {
        const newConv = await apiClient.createConversation(question.trim().slice(0, 50));
        convId = newConv.id;
        setCurrentConversation(convId);
        refetchConversations();
        console.log('Created conversation:', convId);
      } catch (error) {
        console.error('Failed to create conversation:', error);
        // Continue anyway, backend will handle it
      }
    }

    // Add user message to UI
    const userMessage: Message = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content: question.trim(),
      timestamp: new Date().toISOString()
    };
    addMessage(userMessage);

    // Clear input
    const questionText = question.trim();
    setQuestion('');

    // Send question to API with conversation ID
    askMutation.mutate({
      question: questionText,
      include_sources: true,
      conversation_id: convId || undefined
    });
  };

  const handleClearChat = () => {
    if (confirm('Clear all messages?')) {
      clearMessages();
      setCurrentConversation(null);
    }
  };

  const copyWithCitations = (message: Message) => {
    let textToCopy = message.content;

    if (message.sources && message.sources.length > 0) {
      textToCopy += '\n\n---\nSources:\n';
      message.sources.forEach((source, idx) => {
        textToCopy += `\n[${idx + 1}] ${source.title}`;
        if (source.paper_id) {
          textToCopy += ` (arXiv:${source.paper_id})`;
        }
        if (source.relevance_score) {
          textToCopy += ` - Relevance: ${(source.relevance_score * 100).toFixed(0)}%`;
        }
      });
    }

    navigator.clipboard.writeText(textToCopy).then(() => {
      // Could show a toast notification here
      console.log('Copied with citations!');
    });
  };

  // Start editing a message
  const handleStartEdit = (message: Message) => {
    setEditingMessageId(message.id);
    setQuestion(message.content);
    inputRef.current?.focus();
  };

  // Cancel editing
  const handleCancelEdit = () => {
    setEditingMessageId(null);
    setQuestion('');
  };

  // Edit message mutation (creates a branch)
  const editMessageMutation = useMutation({
    mutationFn: async ({ messageId, newContent }: { messageId: string; newContent: string }) => {
      if (!currentConversationId) throw new Error('No conversation selected');
      return apiClient.editMessage(currentConversationId, messageId, newContent);
    },
    onSuccess: async (data) => {
      console.log('Branch created:', data);
      setEditingMessageId(null);
      setQuestion('');
      // Reload conversation to get updated branch
      if (currentConversationId) {
        loadConversationMutation.mutate(currentConversationId);
      }
    },
    onError: (error: any) => {
      console.error('Edit message error:', error);
      alert(`Failed to edit message: ${error.message || 'Unknown error'}`);
    }
  });

  // Switch branch mutation
  const switchBranchMutation = useMutation({
    mutationFn: async ({ messageId, direction }: { messageId: string; direction: 'left' | 'right' }) => {
      if (!currentConversationId) throw new Error('No conversation selected');
      return apiClient.switchBranch(currentConversationId, { message_id: messageId, direction });
    },
    onSuccess: (data) => {
      console.log('Branch switched:', data);
      // Update messages from response
      if (data.messages) {
        clearMessages();
        data.messages.forEach((msg: any) => {
          addMessage({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            sources: msg.sources,
            timestamp: msg.timestamp,
            parent_message_id: msg.parent_message_id,
            branch_id: msg.branch_id,
            branch_index: msg.branch_index,
            sibling_count: msg.sibling_count,
            sibling_index: msg.sibling_index
          });
        });
      }
    },
    onError: (error: any) => {
      console.error('Switch branch error:', error);
    }
  });

  // Handle branch navigation
  const handleSwitchBranch = (messageId: string, direction: 'left' | 'right') => {
    switchBranchMutation.mutate({ messageId, direction });
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Conversation Sidebar */}
      <div className={`${showSidebar ? 'w-64' : 'w-0'} border-r border-border bg-card transition-all duration-300 overflow-hidden flex flex-col shrink-0`}>
        <div className="p-4 border-b border-border">
          <button
            onClick={handleNewChat}
            disabled={createConversationMutation.isPending}
            className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 font-medium"
          >
            + New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {conversationsData && conversationsData.length > 0 ? (
            <div className="space-y-1">
              {conversationsData.map((conv) => (
                <div
                  key={conv.id}
                  className={`relative group rounded-lg transition-colors ${
                    currentConversationId === conv.id
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-muted text-foreground'
                  }`}
                >
                  <button
                    onClick={() => handleLoadConversation(conv.id)}
                    className="w-full text-left px-3 py-2 pr-10"
                  >
                    <div className="font-medium text-sm truncate">{conv.title}</div>
                    <div className="text-xs opacity-70 mt-1">
                      {conv.message_count} message{conv.message_count !== 1 ? 's' : ''}
                    </div>
                  </button>
                  <button
                    onClick={(e) => handleDeleteConversation(conv.id, conv.title, e)}
                    disabled={deleteConversationMutation.isPending}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 transition-opacity disabled:opacity-50"
                    title="Delete conversation"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-sm text-muted-foreground p-4">
              No conversations yet
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="border-b border-border px-6 py-4 shrink-0">
          <div className="container mx-auto max-w-4xl flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowSidebar(!showSidebar)}
                className="px-3 py-2 text-sm text-muted-foreground hover:text-foreground border border-border rounded-lg hover:bg-muted transition-colors"
              >
                {showSidebar ? '◀' : '☰'}
              </button>
              <div>
                <h1 className="text-3xl font-bold">Research Chat</h1>
                <p className="text-sm text-muted-foreground">
                  Ask questions about your indexed papers
                </p>
              </div>
            </div>
            {messages.length > 0 && (
              <button
                onClick={handleClearChat}
                className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground border border-border rounded-lg hover:bg-muted transition-colors"
              >
                Clear Chat
              </button>
            )}
          </div>
        </div>

        {/* Messages Container - Scrollable */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="container mx-auto px-6 py-6 max-w-4xl space-y-6 min-h-full">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center text-muted-foreground max-w-md">
                  <p className="text-lg font-medium mb-2">Welcome to Research Chat</p>
                  <p className="text-sm">
                    Search for papers above, index them, then ask me questions.
                    I'll use semantic search to find relevant content and provide detailed answers with source citations.
                  </p>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} group`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-5 py-3 shadow-sm ${
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-card border border-border text-card-foreground'
                    }`}
                  >
                    {/* Branch Navigation (< 1/3 > style) */}
                    {message.sibling_count && message.sibling_count > 1 && (
                      <div className="flex items-center justify-center gap-2 mb-2 text-xs opacity-80">
                        <button
                          onClick={() => handleSwitchBranch(message.id, 'left')}
                          disabled={(message.sibling_index || 0) <= 0 || switchBranchMutation.isPending}
                          className="px-2 py-1 rounded hover:bg-white/20 disabled:opacity-30 disabled:cursor-not-allowed"
                          title="Previous version"
                        >
                          ◀
                        </button>
                        <span>{(message.sibling_index || 0) + 1}/{message.sibling_count}</span>
                        <button
                          onClick={() => handleSwitchBranch(message.id, 'right')}
                          disabled={(message.sibling_index || 0) >= (message.sibling_count - 1) || switchBranchMutation.isPending}
                          className="px-2 py-1 rounded hover:bg-white/20 disabled:opacity-30 disabled:cursor-not-allowed"
                          title="Next version"
                        >
                          ▶
                        </button>
                      </div>
                    )}

                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>

                    {/* Edit button for user messages (only if we have a server ID) */}
                    {message.role === 'user' && !message.id.startsWith('msg-') && (
                      <div className="mt-2 pt-2 border-t border-white/20 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => handleStartEdit(message)}
                          disabled={editMessageMutation.isPending}
                          className="text-xs px-3 py-1.5 bg-white/20 hover:bg-white/30 rounded transition-colors flex items-center gap-1"
                          title="Edit & resubmit"
                        >
                          ✏️ Edit
                        </button>
                      </div>
                    )}

                    {/* Copy with Citations button for assistant messages */}
                    {message.role === 'assistant' && (
                      <div className="mt-3 pt-3 border-t border-border/30">
                        <button
                          onClick={() => copyWithCitations(message)}
                          className="text-xs px-3 py-1.5 bg-muted hover:bg-muted/80 rounded transition-colors flex items-center gap-1"
                          title="Copy answer with source citations"
                        >
                          📋 Copy with Citations
                        </button>
                      </div>
                    )}

                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-border/50">
                        <p className="text-xs font-semibold mb-2 opacity-80">
                          Sources ({message.sources.length}):
                        </p>

                        {/* Citation List */}
                        <div className="space-y-2">
                          {message.sources.map((source, idx) => {
                            const sourceKey = `${message.id}-${idx}`;
                            const isExpanded = expandedSources.has(sourceKey);

                            return (
                              <div key={idx} className="text-xs">
                                <button
                                  onClick={() => toggleSource(message.id, idx)}
                                  className="flex items-start gap-2 w-full text-left hover:opacity-80 transition-opacity"
                                >
                                  <span className="opacity-60 shrink-0">
                                    {isExpanded ? '▼' : '▶'} [{idx + 1}]
                                  </span>
                                  <span className="font-medium">{source.title}</span>
                                </button>

                                {/* Expanded Details */}
                                {isExpanded && (
                                  <div className="bg-background/50 p-2 mt-1 ml-6 rounded">
                                    {source.excerpt && (
                                      <p className="opacity-70 mb-1">{source.excerpt}</p>
                                    )}
                                    <p className="opacity-60">
                                      Relevance: {(source.relevance_score * 100).toFixed(0)}%
                                    </p>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {/* Loading Indicator */}
            {askMutation.isPending && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-lg px-4 py-3 bg-muted">
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse">Thinking...</div>
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce delay-100"></div>
                      <div className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce delay-200"></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Form - Fixed at bottom */}
        <div className="border-t border-border px-6 py-4 bg-background shrink-0">
          {editingMessageId && (
            <div className="container mx-auto max-w-4xl mb-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted px-3 py-2 rounded-lg">
                <span>✏️ Editing message - this will create a new branch</span>
                <button
                  onClick={handleCancelEdit}
                  className="ml-auto text-xs px-2 py-1 bg-background rounded hover:bg-muted-foreground/10"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (editingMessageId) {
                // Submit edit (creates branch)
                editMessageMutation.mutate({
                  messageId: editingMessageId,
                  newContent: question.trim()
                });
              } else {
                handleSubmit(e);
              }
            }}
            className="container mx-auto max-w-4xl flex gap-2"
          >
            <input
              ref={inputRef}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={editingMessageId ? "Edit your message..." : "Ask a question about your papers..."}
              className={`flex-1 px-4 py-3 border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring ${
                editingMessageId ? 'border-amber-500' : 'border-input'
              }`}
              disabled={askMutation.isPending || editMessageMutation.isPending}
            />
            <button
              type="submit"
              disabled={!question.trim() || askMutation.isPending || editMessageMutation.isPending}
              className={`px-6 py-3 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed ${
                editingMessageId
                  ? 'bg-amber-500 text-white hover:bg-amber-600'
                  : 'bg-primary text-primary-foreground hover:bg-primary/90'
              }`}
            >
              {editMessageMutation.isPending ? 'Creating Branch...' : editingMessageId ? 'Resubmit' : 'Ask'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
