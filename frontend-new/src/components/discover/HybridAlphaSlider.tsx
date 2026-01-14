/**
 * HybridAlphaSlider - Slider to adjust BM25/Vector search balance
 *
 * alpha = 1.0 -> Pure keyword/BM25 search
 * alpha = 0.5 -> Balanced hybrid (default)
 * alpha = 0.0 -> Pure semantic/vector search
 */

interface HybridAlphaSliderProps {
  alpha: number;
  onChange: (alpha: number) => void;
  disabled?: boolean;
}

export default function HybridAlphaSlider({ alpha, onChange, disabled }: HybridAlphaSliderProps) {
  const getSearchModeLabel = (value: number): string => {
    if (value >= 0.8) return 'Keyword';
    if (value <= 0.2) return 'Semantic';
    return 'Hybrid';
  };

  const getSearchModeDescription = (value: number): string => {
    if (value >= 0.8) return 'Exact term matching (BM25)';
    if (value <= 0.2) return 'Conceptual similarity (Vector)';
    return 'Balanced keyword + semantic';
  };

  return (
    <div className="flex flex-col gap-2 w-full max-w-md">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Search Balance:</span>
        <span className="text-sm font-medium px-2 py-0.5 bg-primary/10 text-primary rounded">
          {getSearchModeLabel(alpha)}
        </span>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground w-16">Semantic</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={alpha}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <span className="text-xs text-muted-foreground w-16 text-right">Keyword</span>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        {getSearchModeDescription(alpha)}
      </p>
    </div>
  );
}
