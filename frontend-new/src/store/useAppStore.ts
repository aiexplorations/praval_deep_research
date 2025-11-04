/**
 * Global Application Store using Zustand
 *
 * Manages global UI state, settings, and cross-cutting concerns.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  // Theme
  theme: 'light' | 'dark' | 'system';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;

  // Connection status
  isConnected: boolean;
  setIsConnected: (connected: boolean) => void;

  // Agent status
  agentStatus: {
    message: string;
    status: 'idle' | 'processing' | 'complete' | 'error';
    details?: string;
  };
  setAgentStatus: (status: AppState['agentStatus']) => void;

  // User settings
  settings: {
    chunkSize: number;
    relevanceThreshold: number;
    voiceEnabled: boolean;
    autoSpeak: boolean;
  };
  updateSettings: (settings: Partial<AppState['settings']>) => void;

  // Current page
  currentPage: string;
  setCurrentPage: (page: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Theme
      theme: 'system',
      setTheme: (theme) => set({ theme }),

      // Connection
      isConnected: false,
      setIsConnected: (isConnected) => set({ isConnected }),

      // Agent status
      agentStatus: {
        message: '',
        status: 'idle'
      },
      setAgentStatus: (agentStatus) => set({ agentStatus }),

      // Settings
      settings: {
        chunkSize: 1000,
        relevanceThreshold: 0.7,
        voiceEnabled: false,
        autoSpeak: false
      },
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings }
        })),

      // Current page
      currentPage: 'dashboard',
      setCurrentPage: (currentPage) => set({ currentPage })
    }),
    {
      name: 'praval-app-storage',
      partialize: (state) => ({
        theme: state.theme,
        settings: state.settings
      })
    }
  )
);

export default useAppStore;
