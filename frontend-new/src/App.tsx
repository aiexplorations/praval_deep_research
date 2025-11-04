/**
 * Praval Deep Research - Main App Component
 *
 * React + TypeScript + Vite application for research paper management.
 * This is Phase 1 scaffolding - core features will be added in subsequent phases.
 */

import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Placeholder pages (to be implemented)
const Dashboard = () => (
  <div className="p-8">
    <h1 className="text-3xl font-bold mb-4">Dashboard</h1>
    <p className="text-muted-foreground">Phase 1 Scaffolding Complete! 🎉</p>
    <p className="mt-4">
      React + TypeScript + Vite + Tailwind CSS + React Query + Zustand
    </p>
    <p className="mt-2 text-sm">
      Voice interface scaffolding is in place (disabled until Phase 7)
    </p>
  </div>
);

const Discover = () => (
  <div className="p-8">
    <h1 className="text-3xl font-bold">Discover Papers</h1>
    <p className="text-muted-foreground">Search and discover research papers from ArXiv</p>
  </div>
);

const Chat = () => (
  <div className="p-8">
    <h1 className="text-3xl font-bold">Research Chat</h1>
    <p className="text-muted-foreground">Ask questions about your indexed papers</p>
  </div>
);

const KnowledgeBase = () => (
  <div className="p-8">
    <h1 className="text-3xl font-bold">Knowledge Base</h1>
    <p className="text-muted-foreground">Manage your indexed papers and collections</p>
  </div>
);

const Settings = () => (
  <div className="p-8">
    <h1 className="text-3xl font-bold">Settings</h1>
    <p className="text-muted-foreground">Configure your preferences</p>
  </div>
);

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1
    }
  }
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background text-foreground">
          <nav className="border-b p-4">
            <div className="container mx-auto flex gap-4">
              <a href="/" className="hover:text-primary">Dashboard</a>
              <a href="/discover" className="hover:text-primary">Discover</a>
              <a href="/chat" className="hover:text-primary">Chat</a>
              <a href="/knowledge-base" className="hover:text-primary">Knowledge Base</a>
              <a href="/settings" className="hover:text-primary">Settings</a>
            </div>
          </nav>

          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/discover" element={<Discover />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/knowledge-base" element={<KnowledgeBase />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
