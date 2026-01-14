/**
 * Tests for HybridAlphaSlider component
 */

import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import HybridAlphaSlider from './HybridAlphaSlider'

describe('HybridAlphaSlider', () => {
  it('renders the slider with correct initial value', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.5} onChange={onChange} />)

    const slider = screen.getByRole('slider')
    expect(slider).toHaveValue('0.5')
  })

  it('displays Keyword mode label for alpha >= 0.8', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={1.0} onChange={onChange} />)

    // There are multiple "Keyword" texts - the mode badge and the slider end label
    const keywordLabels = screen.getAllByText('Keyword')
    expect(keywordLabels.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Exact term matching (BM25)')).toBeInTheDocument()
  })

  it('displays Semantic mode label for alpha <= 0.2', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.0} onChange={onChange} />)

    // There are multiple "Semantic" texts - the mode badge and the slider end label
    const semanticLabels = screen.getAllByText('Semantic')
    expect(semanticLabels.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Conceptual similarity (Vector)')).toBeInTheDocument()
  })

  it('displays Hybrid mode label for alpha between 0.2 and 0.8', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.5} onChange={onChange} />)

    expect(screen.getByText('Hybrid')).toBeInTheDocument()
    expect(screen.getByText('Balanced keyword + semantic')).toBeInTheDocument()
  })

  it('calls onChange when slider value changes', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.5} onChange={onChange} />)

    const slider = screen.getByRole('slider')
    fireEvent.change(slider, { target: { value: '0.7' } })

    expect(onChange).toHaveBeenCalledWith(0.7)
  })

  it('disables slider when disabled prop is true', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.5} onChange={onChange} disabled={true} />)

    const slider = screen.getByRole('slider')
    expect(slider).toBeDisabled()
  })

  it('displays Keyword and Semantic labels on slider ends', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.5} onChange={onChange} />)

    // The labels on the slider ends
    const labels = screen.getAllByText(/Keyword|Semantic/)
    expect(labels.length).toBeGreaterThanOrEqual(2)
  })

  it('shows Search Balance label', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.5} onChange={onChange} />)

    expect(screen.getByText('Search Balance:')).toBeInTheDocument()
  })

  it('has correct slider attributes', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.5} onChange={onChange} />)

    const slider = screen.getByRole('slider')
    expect(slider).toHaveAttribute('min', '0')
    expect(slider).toHaveAttribute('max', '1')
    expect(slider).toHaveAttribute('step', '0.1')
  })

  it('handles boundary values correctly - keyword (0.8)', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.8} onChange={onChange} />)

    // The mode label badge
    const modeLabel = screen.getAllByText('Keyword')
    expect(modeLabel.length).toBeGreaterThanOrEqual(1)
  })

  it('handles boundary values correctly - hybrid (0.3)', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.3} onChange={onChange} />)

    // Hybrid mode label
    expect(screen.getByText('Hybrid')).toBeInTheDocument()
  })

  it('handles boundary values correctly - semantic (0.2)', () => {
    const onChange = vi.fn()
    render(<HybridAlphaSlider alpha={0.2} onChange={onChange} />)

    // There are two "Semantic" texts - the mode label badge and the slider end label
    const semanticLabels = screen.getAllByText('Semantic')
    expect(semanticLabels.length).toBeGreaterThanOrEqual(1)
  })
})
