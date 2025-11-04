import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Discover from './pages/Discover';

// Simple test pages
const Dashboard = () => (
  <div className="min-h-screen bg-background text-foreground p-8">
    <h1 className="text-4xl font-bold mb-4">
      Praval Deep Research
    </h1>
    <p className="text-muted-foreground mb-8">
      Phase 2 Migration - Testing Tailwind CSS
    </p>
    <div className="flex gap-4">
      <Link to="/discover" className="text-primary hover:underline">Discover</Link>
      <Link to="/chat" className="text-primary hover:underline">Chat</Link>
      <Link to="/knowledge-base" className="text-primary hover:underline">Knowledge Base</Link>
    </div>
  </div>
);

// Using real Discover component imported above

const Chat = () => (
  <div style={{ padding: '20px', backgroundColor: 'white', minHeight: '100vh' }}>
    <h1 style={{ color: 'black' }}>Chat Page</h1>
    <Link to="/" style={{ color: 'blue' }}>Back to Home</Link>
  </div>
);

const KnowledgeBase = () => (
  <div style={{ padding: '20px', backgroundColor: 'white', minHeight: '100vh' }}>
    <h1 style={{ color: 'black' }}>Knowledge Base Page</h1>
    <Link to="/" style={{ color: 'blue' }}>Back to Home</Link>
  </div>
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      retry: 1
    }
  }
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/discover" element={<Discover />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/knowledge-base" element={<KnowledgeBase />} />
        </Routes>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
