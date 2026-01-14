/**
 * SearchModeToggle - Toggle between ArXiv and Knowledge Base search modes
 */

interface SearchModeToggleProps {
  mode: 'arxiv' | 'knowledge_base';
  onChange: (mode: 'arxiv' | 'knowledge_base') => void;
  disabled?: boolean;
}

export default function SearchModeToggle({ mode, onChange, disabled }: SearchModeToggleProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground">Search:</span>
      <div className="flex rounded-lg border border-border overflow-hidden">
        <button
          onClick={() => onChange('arxiv')}
          disabled={disabled}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            mode === 'arxiv'
              ? 'bg-primary text-primary-foreground'
              : 'bg-background text-muted-foreground hover:bg-muted'
          } disabled:opacity-50`}
        >
          ArXiv
        </button>
        <button
          onClick={() => onChange('knowledge_base')}
          disabled={disabled}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            mode === 'knowledge_base'
              ? 'bg-primary text-primary-foreground'
              : 'bg-background text-muted-foreground hover:bg-muted'
          } disabled:opacity-50`}
        >
          Knowledge Base
        </button>
      </div>
    </div>
  );
}
