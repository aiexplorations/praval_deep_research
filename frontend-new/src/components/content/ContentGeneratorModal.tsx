/**
 * Content Generator Modal
 *
 * Allows users to generate shareable content (Twitter threads or blog posts)
 * from research conversations.
 */

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../../services/api/client';
import type { ContentFormat, ContentStyle, ContentGenerationResponse, Tweet } from '../../types';

interface ContentGeneratorModalProps {
  isOpen: boolean;
  onClose: () => void;
  conversationId: string;
}

export default function ContentGeneratorModal({
  isOpen,
  onClose,
  conversationId
}: ContentGeneratorModalProps) {
  // Form state
  const [format, setFormat] = useState<ContentFormat>('twitter');
  const [style, setStyle] = useState<ContentStyle>('academic');
  const [maxTweets, setMaxTweets] = useState(10);
  const [includeToc, setIncludeToc] = useState(true);
  const [customPrompt, setCustomPrompt] = useState('');

  // Generated content
  const [generatedContent, setGeneratedContent] = useState<ContentGenerationResponse | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  // Generate mutation
  const generateMutation = useMutation({
    mutationFn: () =>
      apiClient.generateContent(conversationId, {
        format,
        style,
        maxTweets,
        includeToc,
        customPrompt
      }),
    onSuccess: (data) => {
      setGeneratedContent(data);
    }
  });

  // Copy to clipboard
  const copyToClipboard = async (text: string, index?: number) => {
    try {
      await navigator.clipboard.writeText(text);
      if (index !== undefined) {
        setCopiedIndex(index);
        setTimeout(() => setCopiedIndex(null), 2000);
      }
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Copy all tweets as a thread
  const copyAllTweets = async () => {
    if (!generatedContent?.tweets) return;
    const threadText = generatedContent.tweets
      .map((t) => t.content)
      .join('\n\n---\n\n');
    await copyToClipboard(threadText);
    setCopiedIndex(-1); // -1 indicates "all"
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Copy blog post
  const copyBlogPost = async () => {
    if (!generatedContent?.blog_post) return;
    await copyToClipboard(generatedContent.blog_post.content);
    setCopiedIndex(-1);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Reset on close
  const handleClose = () => {
    setGeneratedContent(null);
    setCopiedIndex(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-background border border-border rounded-lg w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Generate Content</h2>
          <button
            onClick={handleClose}
            className="text-muted-foreground hover:text-foreground p-1"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Format Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Format</label>
            <div className="flex gap-2">
              <button
                onClick={() => setFormat('twitter')}
                className={`flex-1 py-2 px-4 rounded-lg border transition-colors ${
                  format === 'twitter'
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'border-border hover:bg-muted'
                }`}
              >
                <span className="mr-2">𝕏</span> Twitter Thread
              </button>
              <button
                onClick={() => setFormat('blog')}
                className={`flex-1 py-2 px-4 rounded-lg border transition-colors ${
                  format === 'blog'
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'border-border hover:bg-muted'
                }`}
              >
                <span className="mr-2">📝</span> Blog Post
              </button>
            </div>
          </div>

          {/* Style Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Style</label>
            <div className="flex gap-2">
              {(['academic', 'casual', 'narrative'] as ContentStyle[]).map((s) => (
                <button
                  key={s}
                  onClick={() => setStyle(s)}
                  className={`flex-1 py-2 px-3 rounded-lg border text-sm transition-colors capitalize ${
                    style === s
                      ? 'bg-primary text-primary-foreground border-primary'
                      : 'border-border hover:bg-muted'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {style === 'academic' && 'Formal tone, technical depth, methodology focus'}
              {style === 'casual' && 'Accessible language, broader appeal'}
              {style === 'narrative' && 'Storytelling format, engaging flow'}
            </p>
          </div>

          {/* Format-specific options */}
          {format === 'twitter' && (
            <div>
              <label className="block text-sm font-medium mb-2">
                Max Tweets: {maxTweets}
              </label>
              <input
                type="range"
                min={2}
                max={20}
                value={maxTweets}
                onChange={(e) => setMaxTweets(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>2</span>
                <span>20</span>
              </div>
            </div>
          )}

          {format === 'blog' && (
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="includeToc"
                checked={includeToc}
                onChange={(e) => setIncludeToc(e.target.checked)}
                className="rounded border-border"
              />
              <label htmlFor="includeToc" className="text-sm">
                Include table of contents
              </label>
            </div>
          )}

          {/* Custom Instructions */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Custom Instructions <span className="text-muted-foreground font-normal">(optional)</span>
            </label>
            <textarea
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              placeholder="E.g., Focus on practical applications, emphasize the novelty, make it engaging for ML practitioners..."
              className="w-full px-3 py-2 border border-border rounded-lg bg-background text-sm resize-none focus:outline-none focus:ring-2 focus:ring-ring"
              rows={2}
            />
          </div>

          {/* Generate Button */}
          <button
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending}
            className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 transition-colors"
          >
            {generateMutation.isPending ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Generating...
              </span>
            ) : generatedContent ? (
              'Regenerate'
            ) : (
              'Generate'
            )}
          </button>

          {/* Error */}
          {generateMutation.isError && (
            <div className="p-3 bg-destructive/10 text-destructive rounded-lg text-sm">
              Failed to generate content. Please try again.
            </div>
          )}

          {/* Generated Content Preview */}
          {generatedContent && (
            <div className="border border-border rounded-lg overflow-hidden">
              <div className="p-3 bg-muted/50 border-b border-border flex items-center justify-between">
                <span className="text-sm font-medium">
                  {format === 'twitter'
                    ? `Generated ${generatedContent.tweets?.length || 0} tweets`
                    : `${generatedContent.blog_post?.word_count || 0} words`}
                </span>
                <div className="flex gap-2">
                  {format === 'twitter' && (
                    <button
                      onClick={copyAllTweets}
                      className="text-xs px-3 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors"
                    >
                      {copiedIndex === -1 ? 'Copied!' : 'Copy All'}
                    </button>
                  )}
                  {format === 'blog' && (
                    <button
                      onClick={copyBlogPost}
                      className="text-xs px-3 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors"
                    >
                      {copiedIndex === -1 ? 'Copied!' : 'Copy Markdown'}
                    </button>
                  )}
                </div>
              </div>

              {/* Twitter Preview */}
              {format === 'twitter' && generatedContent.tweets && (
                <div className="max-h-80 overflow-y-auto">
                  {generatedContent.tweets.map((tweet: Tweet, index: number) => (
                    <div
                      key={tweet.position}
                      className="p-3 border-b border-border last:border-b-0 hover:bg-muted/30"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-sm flex-1 whitespace-pre-wrap">{tweet.content}</p>
                        <button
                          onClick={() => copyToClipboard(tweet.content, index)}
                          className="text-muted-foreground hover:text-foreground p-1 flex-shrink-0"
                          title="Copy tweet"
                        >
                          {copiedIndex === index ? (
                            <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          ) : (
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          )}
                        </button>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span
                          className={`text-xs ${
                            tweet.char_count > 280 ? 'text-destructive' : 'text-muted-foreground'
                          }`}
                        >
                          {tweet.char_count}/280
                        </span>
                        {tweet.has_citation && (
                          <span className="text-xs text-blue-500">Has citation</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Blog Preview */}
              {format === 'blog' && generatedContent.blog_post && (
                <div className="max-h-80 overflow-y-auto p-4">
                  <h3 className="text-lg font-semibold mb-3">
                    {generatedContent.blog_post.title}
                  </h3>
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <pre className="whitespace-pre-wrap text-sm font-mono bg-muted/50 p-3 rounded overflow-x-auto">
                      {generatedContent.blog_post.content}
                    </pre>
                  </div>
                </div>
              )}

              {/* Citations */}
              {generatedContent.papers_cited.length > 0 && (
                <div className="p-3 bg-muted/30 border-t border-border">
                  <p className="text-xs text-muted-foreground">
                    Papers cited: {generatedContent.papers_cited.join(', ')}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border flex justify-end">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
