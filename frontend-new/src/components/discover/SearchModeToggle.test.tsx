/**
 * Tests for SearchModeToggle component
 */

import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import SearchModeToggle from './SearchModeToggle'

describe('SearchModeToggle', () => {
  it('renders both mode buttons', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="arxiv" onChange={onChange} />)

    expect(screen.getByText('ArXiv')).toBeInTheDocument()
    expect(screen.getByText('Knowledge Base')).toBeInTheDocument()
  })

  it('highlights the active mode', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="arxiv" onChange={onChange} />)

    const arxivButton = screen.getByText('ArXiv')
    const kbButton = screen.getByText('Knowledge Base')

    // ArXiv should have primary styling
    expect(arxivButton.className).toContain('bg-primary')
    // KB should not have primary styling
    expect(kbButton.className).not.toContain('bg-primary')
  })

  it('highlights knowledge_base mode when active', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="knowledge_base" onChange={onChange} />)

    const arxivButton = screen.getByText('ArXiv')
    const kbButton = screen.getByText('Knowledge Base')

    // KB should have primary styling
    expect(kbButton.className).toContain('bg-primary')
    // ArXiv should not have primary styling
    expect(arxivButton.className).not.toContain('bg-primary')
  })

  it('calls onChange when clicking ArXiv button', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="knowledge_base" onChange={onChange} />)

    fireEvent.click(screen.getByText('ArXiv'))

    expect(onChange).toHaveBeenCalledWith('arxiv')
  })

  it('calls onChange when clicking Knowledge Base button', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="arxiv" onChange={onChange} />)

    fireEvent.click(screen.getByText('Knowledge Base'))

    expect(onChange).toHaveBeenCalledWith('knowledge_base')
  })

  it('disables buttons when disabled prop is true', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="arxiv" onChange={onChange} disabled={true} />)

    const arxivButton = screen.getByText('ArXiv')
    const kbButton = screen.getByText('Knowledge Base')

    expect(arxivButton).toBeDisabled()
    expect(kbButton).toBeDisabled()
  })

  it('does not call onChange when disabled', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="arxiv" onChange={onChange} disabled={true} />)

    fireEvent.click(screen.getByText('Knowledge Base'))

    expect(onChange).not.toHaveBeenCalled()
  })

  it('displays the label text', () => {
    const onChange = vi.fn()
    render(<SearchModeToggle mode="arxiv" onChange={onChange} />)

    expect(screen.getByText('Search:')).toBeInTheDocument()
  })
})
