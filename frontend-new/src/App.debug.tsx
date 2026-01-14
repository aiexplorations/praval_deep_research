// Minimal test app
export default function App() {
  return (
    <div style={{
      backgroundColor: 'white',
      color: 'black',
      padding: '20px',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '32px', fontWeight: 'bold' }}>
        React is Working!
      </h1>
      <p>If you see this, React is rendering correctly.</p>
      <p>The issue is likely with the main App component or its dependencies.</p>
    </div>
  );
}
