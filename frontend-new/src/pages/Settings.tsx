/**
 * Settings Page - Configuration for Praval Deep Research
 *
 * Allows users to configure:
 * - LLM Provider (OpenAI, Anthropic, Ollama)
 * - API Keys
 * - Model Selection
 * - Ollama Configuration (for local models)
 */

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

// Check if running in Tauri
const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

interface AppConfig {
  openai_api_key: string | null;
  anthropic_api_key: string | null;
  gemini_api_key: string | null;
  embedding_model: string;
  llm_model: string;
  llm_provider: string;
  ollama_base_url: string;
  ollama_model: string;
  langextract_provider: string;
  langextract_model: string;
  theme: string;
}

const defaultConfig: AppConfig = {
  openai_api_key: null,
  anthropic_api_key: null,
  gemini_api_key: null,
  embedding_model: 'text-embedding-3-small',
  llm_model: 'gpt-4o-mini',
  llm_provider: 'openai',
  ollama_base_url: 'http://localhost:11434',
  ollama_model: 'llama3.2',
  langextract_provider: 'gemini',
  langextract_model: 'gemini-2.5-flash',
  theme: 'system',
};

// Model options per provider
const modelOptions: Record<string, string[]> = {
  openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
  anthropic: ['claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
  ollama: ['llama3.2', 'llama3.1', 'mistral', 'mixtral', 'codellama', 'phi3', 'gemma2:9b', 'qwen2.5'],
  gemini: ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro'],
};

const embeddingOptions = [
  'text-embedding-3-small',
  'text-embedding-3-large',
  'text-embedding-ada-002',
];

export default function Settings() {
  const [config, setConfig] = useState<AppConfig>(defaultConfig);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);

  // Load config on mount
  useEffect(() => {
    loadConfig();
  }, []);

  // Check Ollama connection when provider changes or URL changes
  useEffect(() => {
    if (config.llm_provider === 'ollama' || config.langextract_provider === 'ollama') {
      checkOllamaConnection();
    }
  }, [config.llm_provider, config.langextract_provider, config.ollama_base_url]);

  const loadConfig = async () => {
    setLoading(true);
    try {
      if (isTauri) {
        // Load from Tauri backend
        const { invoke } = await import('@tauri-apps/api/core');
        const savedConfig = await invoke<AppConfig>('get_config');
        setConfig({ ...defaultConfig, ...savedConfig });
      } else {
        // Load from localStorage for web mode
        const saved = localStorage.getItem('praval_config');
        if (saved) {
          setConfig({ ...defaultConfig, ...JSON.parse(saved) });
        }
      }
    } catch (error) {
      console.error('Failed to load config:', error);
      setMessage({ type: 'error', text: 'Failed to load settings' });
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    setMessage(null);
    try {
      if (isTauri) {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke('save_config', { config });
      } else {
        localStorage.setItem('praval_config', JSON.stringify(config));
        // Also update backend via API
        await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        });
      }
      setMessage({ type: 'success', text: 'Settings saved successfully!' });
    } catch (error) {
      console.error('Failed to save config:', error);
      setMessage({ type: 'error', text: 'Failed to save settings' });
    } finally {
      setSaving(false);
    }
  };

  const checkOllamaConnection = async () => {
    setOllamaStatus('checking');
    try {
      const response = await fetch(`${config.ollama_base_url}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok) {
        const data = await response.json();
        const models = data.models?.map((m: { name: string }) => m.name) || [];
        setOllamaModels(models);
        setOllamaStatus('connected');
      } else {
        setOllamaStatus('disconnected');
        setOllamaModels([]);
      }
    } catch {
      setOllamaStatus('disconnected');
      setOllamaModels([]);
    }
  };

  const updateConfig = (key: keyof AppConfig, value: string | null) => {
    setConfig(prev => {
      const updated = { ...prev, [key]: value };

      // Auto-update model when provider changes
      if (key === 'llm_provider') {
        const provider = value as string;
        if (provider === 'ollama' && ollamaModels.length > 0) {
          updated.llm_model = ollamaModels[0];
        } else {
          updated.llm_model = modelOptions[provider]?.[0] || '';
        }
      }
      if (key === 'langextract_provider') {
        const provider = value as string;
        if (provider === 'ollama' && ollamaModels.length > 0) {
          updated.langextract_model = ollamaModels[0];
        } else {
          updated.langextract_model = modelOptions[provider]?.[0] || '';
        }
      }

      return updated;
    });
  };

  const maskApiKey = (key: string | null): string => {
    if (!key) return '';
    if (key.length <= 8) return '••••••••';
    return key.slice(0, 4) + '••••••••' + key.slice(-4);
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground mt-1">
            Configure your LLM providers and API keys
          </p>
        </div>
        <Link
          to="/"
          className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          ← Back to Dashboard
        </Link>
      </div>

      {message && (
        <div
          className={`mb-6 p-4 rounded-lg ${
            message.type === 'success'
              ? 'bg-green-500/10 text-green-500 border border-green-500/20'
              : 'bg-red-500/10 text-red-500 border border-red-500/20'
          }`}
        >
          {message.text}
        </div>
      )}

      <div className="space-y-8">
        {/* LLM Provider Section */}
        <section className="border border-border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">LLM Provider</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Choose the AI provider for chat and research analysis.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {['openai', 'anthropic', 'ollama'].map(provider => (
              <button
                key={provider}
                onClick={() => updateConfig('llm_provider', provider)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  config.llm_provider === provider
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-muted-foreground'
                }`}
              >
                <div className="font-semibold capitalize">{provider}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {provider === 'openai' && 'GPT-4, GPT-4o'}
                  {provider === 'anthropic' && 'Claude 3.5, Claude 4'}
                  {provider === 'ollama' && 'Local models (free)'}
                </div>
              </button>
            ))}
          </div>

          {/* Model Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Model</label>
            <select
              value={config.llm_model}
              onChange={e => updateConfig('llm_model', e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {(config.llm_provider === 'ollama' && ollamaModels.length > 0
                ? ollamaModels
                : modelOptions[config.llm_provider] || []
              ).map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          {/* Ollama Status */}
          {config.llm_provider === 'ollama' && (
            <div className="mt-4 p-4 rounded-lg bg-muted/50">
              <div className="flex items-center gap-2 mb-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    ollamaStatus === 'connected'
                      ? 'bg-green-500'
                      : ollamaStatus === 'checking'
                      ? 'bg-yellow-500 animate-pulse'
                      : 'bg-red-500'
                  }`}
                />
                <span className="text-sm font-medium">
                  {ollamaStatus === 'connected'
                    ? `Ollama Connected (${ollamaModels.length} models)`
                    : ollamaStatus === 'checking'
                    ? 'Checking connection...'
                    : 'Ollama Not Connected'}
                </span>
              </div>

              <div className="mt-3">
                <label className="block text-sm font-medium mb-2">Ollama URL</label>
                <input
                  type="text"
                  value={config.ollama_base_url}
                  onChange={e => updateConfig('ollama_base_url', e.target.value)}
                  placeholder="http://localhost:11434"
                  className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              {ollamaStatus === 'disconnected' && (
                <p className="text-xs text-muted-foreground mt-2">
                  Make sure Ollama is running. Install from{' '}
                  <a
                    href="https://ollama.ai"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    ollama.ai
                  </a>
                </p>
              )}
            </div>
          )}
        </section>

        {/* API Keys Section */}
        <section className="border border-border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">API Keys</h2>
          <p className="text-sm text-muted-foreground mb-4">
            API keys are stored locally and never sent to our servers.
          </p>

          <div className="space-y-4">
            {/* OpenAI API Key */}
            <div>
              <label className="block text-sm font-medium mb-2">
                OpenAI API Key
                {config.llm_provider === 'openai' && (
                  <span className="ml-2 text-xs text-primary">(Required)</span>
                )}
              </label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={config.openai_api_key || ''}
                  onChange={e => updateConfig('openai_api_key', e.target.value || null)}
                  placeholder="sk-..."
                  className="flex-1 px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                />
                {config.openai_api_key && (
                  <button
                    onClick={() => updateConfig('openai_api_key', null)}
                    className="px-3 py-2 text-sm text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
              {config.openai_api_key && (
                <p className="text-xs text-muted-foreground mt-1">
                  Current: {maskApiKey(config.openai_api_key)}
                </p>
              )}
            </div>

            {/* Anthropic API Key */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Anthropic API Key
                {config.llm_provider === 'anthropic' && (
                  <span className="ml-2 text-xs text-primary">(Required)</span>
                )}
              </label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={config.anthropic_api_key || ''}
                  onChange={e => updateConfig('anthropic_api_key', e.target.value || null)}
                  placeholder="sk-ant-..."
                  className="flex-1 px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                />
                {config.anthropic_api_key && (
                  <button
                    onClick={() => updateConfig('anthropic_api_key', null)}
                    className="px-3 py-2 text-sm text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
              {config.anthropic_api_key && (
                <p className="text-xs text-muted-foreground mt-1">
                  Current: {maskApiKey(config.anthropic_api_key)}
                </p>
              )}
            </div>

            {/* Gemini API Key */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Google Gemini API Key
                {config.langextract_provider === 'gemini' && (
                  <span className="ml-2 text-xs text-primary">(Required for PDF extraction)</span>
                )}
              </label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={config.gemini_api_key || ''}
                  onChange={e => updateConfig('gemini_api_key', e.target.value || null)}
                  placeholder="AI..."
                  className="flex-1 px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                />
                {config.gemini_api_key && (
                  <button
                    onClick={() => updateConfig('gemini_api_key', null)}
                    className="px-3 py-2 text-sm text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* PDF Extraction Section */}
        <section className="border border-border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">PDF Extraction (LangExtract)</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Choose the AI provider for extracting structured data from research papers.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {['gemini', 'openai', 'ollama'].map(provider => (
              <button
                key={provider}
                onClick={() => updateConfig('langextract_provider', provider)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  config.langextract_provider === provider
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-muted-foreground'
                }`}
              >
                <div className="font-semibold capitalize">{provider}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {provider === 'gemini' && 'Fast, accurate (recommended)'}
                  {provider === 'openai' && 'GPT-4 Vision'}
                  {provider === 'ollama' && 'Local (requires vision model)'}
                </div>
              </button>
            ))}
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Extraction Model</label>
            <select
              value={config.langextract_model}
              onChange={e => updateConfig('langextract_model', e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {(config.langextract_provider === 'ollama' && ollamaModels.length > 0
                ? ollamaModels
                : modelOptions[config.langextract_provider] || []
              ).map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
        </section>

        {/* Embedding Section */}
        <section className="border border-border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Embeddings</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Configure the embedding model for semantic search.
          </p>

          <div>
            <label className="block text-sm font-medium mb-2">Embedding Model</label>
            <select
              value={config.embedding_model}
              onChange={e => updateConfig('embedding_model', e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {embeddingOptions.map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground mt-2">
              Requires OpenAI API key. text-embedding-3-small is recommended for most use cases.
            </p>
          </div>
        </section>

        {/* Save Button */}
        <div className="flex justify-end gap-4">
          <button
            onClick={loadConfig}
            className="px-6 py-2 rounded-lg border border-border hover:bg-muted transition-colors"
          >
            Reset
          </button>
          <button
            onClick={saveConfig}
            disabled={saving}
            className="px-6 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  );
}
