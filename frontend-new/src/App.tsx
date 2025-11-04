/**
 * Praval Deep Research - Main App Component
 *
 * React + TypeScript + Vite application for research paper management.
 * This is Phase 1 scaffolding - core features will be added in subsequent phases.
 */

import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Import page components
import Discover from './pages/Discover';
import Chat from './pages/Chat';
import KnowledgeBase from './pages/KnowledgeBase';

// Dashboard (placeholder for now)
const Dashboard = () => (
  <div className="container mx-auto px-4 py-8 max-w-4xl">
    <h1 className="text-4xl font-bold mb-4">Praval Deep Research</h1>
    <p className="text-muted-foreground mb-8">
      Phase 2 Migration Complete! Core features are now functional. 🎉
    </p>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
      <div className="p-6 border border-border rounded-lg hover:shadow-md transition-shadow">
        <h2 className="text-xl font-semibold mb-2">🔍 Discover Papers</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Search ArXiv for research papers and index them for Q&A
        </p>
        <Link
          to="/discover"
          className="text-primary hover:underline text-sm font-medium"
        >
          Start Searching →
        </Link>
      </div>

      <div className="p-6 border border-border rounded-lg hover:shadow-md transition-shadow">
        <h2 className="text-xl font-semibold mb-2">💬 Research Chat</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Ask questions about your indexed papers with semantic search
        </p>
        <Link
          to="/chat"
          className="text-primary hover:underline text-sm font-medium"
        >
          Start Chatting →
        </Link>
      </div>

      <div className="p-6 border border-border rounded-lg hover:shadow-md transition-shadow">
        <h2 className="text-xl font-semibold mb-2">📚 Knowledge Base</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Manage your indexed papers and view statistics
        </p>
        <Link
          to="/knowledge-base"
          className="text-primary hover:underline text-sm font-medium"
        >
          View Papers →
        </Link>
      </div>

      <div className="p-6 border border-border rounded-lg hover:shadow-md transition-shadow opacity-50">
        <h2 className="text-xl font-semibold mb-2">⚙️ Settings</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Configure preferences (Coming in Phase 8)
        </p>
      </div>
    </div>

    <div className="p-4 bg-muted rounded-lg text-sm">
      <p className="font-medium mb-2">✨ Features Implemented:</p>
      <ul className="space-y-1 text-muted-foreground">
        <li>✅ Paper search from ArXiv with domain filtering</li>
        <li>✅ Paper selection and indexing</li>
        <li>✅ Q&A chat with source citations</li>
        <li>✅ Knowledge Base table with search and sorting</li>
        <li>✅ Responsive design (mobile-friendly)</li>
        <li>✅ Dark mode ready</li>
        <li>🔜 Collections & Tags (Phase 3)</li>
        <li>🔜 Persistent Chat History (Phase 4)</li>
        <li>🔜 AI Agents Integration (Phase 5)</li>
      </ul>
    </div>
  </div>
);

const Settings = () => (
  <div className="p-8">
    <h1 className="text-3xl font-bold">Settings</h1>
    <p className="text-muted-foreground">Coming in Phase 8 - Advanced Features</p>
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
          <nav className="border-b p-4 bg-card">
            <div className="container mx-auto flex items-center justify-between">
              <div className="flex gap-6">
                <Link
                  to="/"
                  className="font-semibold text-foreground hover:text-primary transition-colors"
                >
                  Praval Research
                </Link>
                <Link
                  to="/discover"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Discover
                </Link>
                <Link
                  to="/chat"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Chat
                </Link>
                <Link
                  to="/knowledge-base"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Knowledge Base
                </Link>
                <Link
                  to="/settings"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  Settings
                </Link>
              </div>
              <div className="text-sm text-muted-foreground">
                v2.0 (React)
              </div>
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
