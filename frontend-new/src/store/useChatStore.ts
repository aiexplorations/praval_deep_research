/**
 * Chat/Conversation Store
 *
 * Manages chat messages, conversation history, and Q&A state.
 * Uses localStorage persistence to maintain conversation context across sessions.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Message, Conversation } from '../types';

interface ChatState {
  // Current conversation
  currentConversationId: string | null;
  messages: Message[];

  // Conversations list
  conversations: Conversation[];

  // Actions
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  setMessages: (messages: Message[]) => void;
  clearMessages: () => void;
  setCurrentConversation: (id: string | null) => void;
  setConversations: (conversations: Conversation[]) => void;
  addConversation: (conversation: Conversation) => void;
  removeConversation: (id: string) => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      // State
      currentConversationId: null,
      messages: [],
      conversations: [],

      // Actions
      addMessage: (message) =>
        set((state) => ({
          messages: [...state.messages, message]
        })),

      updateMessage: (id, updates) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, ...updates } : msg
          )
        })),

      setMessages: (messages) => set({ messages }),

      clearMessages: () => set({ messages: [] }),

      setCurrentConversation: (id) => set({ currentConversationId: id }),

      setConversations: (conversations) => set({ conversations }),

      addConversation: (conversation) =>
        set((state) => ({
          conversations: [conversation, ...state.conversations]
        })),

      removeConversation: (id) =>
        set((state) => ({
          conversations: state.conversations.filter((c) => c.id !== id),
          currentConversationId:
            state.currentConversationId === id ? null : state.currentConversationId
        }))
    }),
    {
      name: 'chat-storage', // unique name for localStorage key
    }
  )
);

export default useChatStore;
