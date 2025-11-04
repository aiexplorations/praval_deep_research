/**
 * Chat Page - Research Q&A Interface
 *
 * Ask questions about indexed papers with semantic search and source citations.
 */

import { useState, useRef, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../services/api/client';
import { useChatStore } from '../store/useChatStore';
import type { Message, QuestionRequest } from '../types';

export default function Chat() {
  const [question, setQuestion] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, addMessage, clearMessages } = useChatStore();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || askMutation.isPending) return;

    // Add user message
    const userMessage: Message = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content: question.trim(),
      timestamp: new Date().toISOString()
    };
    addMessage(userMessage);

    // Clear input
    setQuestion('');

    // Send question to API
    askMutation.mutate({
      question: userMessage.content,
      include_sources: true
    });
  };

  const handleClearChat = () => {
    if (confirm('Clear all messages?')) {
      clearMessages();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-background">
      <div className="container mx-auto px-4 py-6 max-w-4xl flex-1 flex flex-col">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Research Chat</h1>
            <p className="text-sm text-muted-foreground">
              Ask questions about your indexed papers
            </p>
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
                  className={`max-w-[80%] rounded-lg px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  }`}
                >
                  <div className="whitespace-pre-wrap break-words">{message.content}</div>

                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-border/50">
                      <p className="text-xs font-semibold mb-2 opacity-80">Sources:</p>
                      {message.sources.map((source, idx) => (
                        <div
                          key={idx}
                          className="bg-background/50 p-2 mb-2 rounded text-xs"
                        >
                          <p className="font-medium">{source.title}</p>
                          {source.excerpt && (
                            <p className="opacity-70 mt-1">{source.excerpt}</p>
                          )}
                          <p className="opacity-60 mt-1">
                            Relevance: {(source.relevance_score * 100).toFixed(0)}%
                          </p>
                        </div>
                      ))}
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
  );
}
