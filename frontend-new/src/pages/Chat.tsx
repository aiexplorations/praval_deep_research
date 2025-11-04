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
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, addMessage, clearMessages } = useChatStore();

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
      setCurrentConversationId(data.id);
      clearMessages();
      refetchConversations();
    }
  });

  // Load conversation
  const loadConversationMutation = useMutation({
    mutationFn: (id: string) => apiClient.getConversation(id),
    onSuccess: (data) => {
      console.log('Loaded conversation:', data.id, 'with', data.messages?.length, 'messages');
      setCurrentConversationId(data.id);
      clearMessages();
      // Load messages from conversation
      if (data.messages) {
        data.messages.forEach((msg: any) => {
          addMessage({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            sources: msg.sources,
            timestamp: msg.timestamp
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
        setCurrentConversationId(null);
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
        setCurrentConversationId(convId);
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
    }
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] bg-background">
      {/* Conversation Sidebar */}
      <div className={`${showSidebar ? 'w-64' : 'w-0'} border-r border-border bg-card transition-all duration-300 overflow-hidden flex flex-col`}>
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
      <div className="flex-1 flex flex-col">
        <div className="container mx-auto px-4 py-6 max-w-4xl flex-1 flex flex-col">
          {/* Header */}
          <div className="mb-6 flex items-center justify-between">
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

          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto mb-6 space-y-6">
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
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-5 py-3 shadow-sm ${
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-card border border-border text-card-foreground'
                    }`}
                  >
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>

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

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your papers..."
              className="flex-1 px-4 py-3 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={askMutation.isPending}
            />
            <button
              type="submit"
              disabled={!question.trim() || askMutation.isPending}
              className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              Ask
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
