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
import Settings from './pages/Settings';

const Dashboard = () => (
  <div className="container mx-auto px-4 py-8 max-w-4xl">
    <h1 className="text-4xl font-bold mb-2">Praval Deep Research</h1>
    <p className="text-muted-foreground mb-8">
      AI-powered research assistant for academic papers
    </p>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Link
        to="/discover"
        className="p-6 border border-border rounded-lg hover:shadow-lg transition-all hover:border-primary"
      >
        <h2 className="text-xl font-semibold mb-2">Discover</h2>
        <p className="text-sm text-muted-foreground">
          Search and index ArXiv papers
        </p>
      </Link>

      <Link
        to="/chat"
        className="p-6 border border-border rounded-lg hover:shadow-lg transition-all hover:border-primary"
      >
        <h2 className="text-xl font-semibold mb-2">Chat</h2>
        <p className="text-sm text-muted-foreground">
          Ask questions about your papers
        </p>
      </Link>

      <Link
        to="/knowledge-base"
        className="p-6 border border-border rounded-lg hover:shadow-lg transition-all hover:border-primary"
      >
        <h2 className="text-xl font-semibold mb-2">Knowledge Base</h2>
        <p className="text-sm text-muted-foreground">
          Manage indexed papers
        </p>
      </Link>
    </div>
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
              <div className="flex gap-6 items-center">
                <Link
                  to="/"
                  className="flex items-center gap-3 font-semibold text-foreground hover:text-primary transition-colors"
                >
                  <img src="/praval_deep_research_logo.png" alt="Praval Logo" className="h-10 w-auto" />
                  <span className="text-lg">Praval Deep Research</span>
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

              {/* Praval Branding - Top Right */}
              <a
                href="https://pravalagents.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-muted transition-all group"
                title="Built with Praval - The Modern Agentic Framework"
              >
                <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                  Built with
                </span>
                <img
                  src="/praval_logo.png"
                  alt="Praval Framework"
                  className="h-6 w-auto opacity-80 group-hover:opacity-100 transition-opacity"
                />
              </a>
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
