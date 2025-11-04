/**
 * Voice Interface Hook
 *
 * Core abstraction for voice input/output that works with ANY provider.
 * This is scaffolding for Phase 7 - currently returns disabled state.
 *
 * Phases:
 * - Phase 1 (Week 7): Web Speech API (free, instant)
 * - Phase 2 (Month 2): OpenAI Whisper (optional, paid)
 * - Phase 3 (Month 3): whisper.cpp WASM (local, private)
 */

import { useState, useCallback, useEffect } from 'react';
import { voiceConfig } from '../config/voice.config';

export interface VoiceState {
  isListening: boolean;
  isSpeaking: boolean;
  transcript: string;
  interimTranscript: string;
  error: string | null;
  confidence: number;
  isSupported: boolean;
}

export interface VoiceConfig {
  provider?: 'webspeech' | 'openai-whisper' | 'whisper-wasm';
  language?: string;
  continuousListening?: boolean;
  interimResults?: boolean;
  autoSpeak?: boolean;
  onTranscript?: (text: string, isFinal: boolean) => void;
  onError?: (error: Error) => void;
}

export interface UseVoiceInterfaceReturn extends VoiceState {
  startListening: () => Promise<void>;
  stopListening: () => void;
  speak: (text: string) => void;
  cancelSpeaking: () => void;
  reset: () => void;
}

/**
 * Voice interface hook - abstraction layer for all voice providers
 *
 * Usage:
 * ```typescript
 * const voice = useVoiceInterface({
 *   onTranscript: (text, isFinal) => {
 *     if (isFinal) handleQuestion(text);
 *   }
 * });
 *
 * // Start listening
 * await voice.startListening();
 *
 * // Stop listening
 * voice.stopListening();
 *
 * // Speak text
 * voice.speak("Your answer is...");
 * ```
 */
export function useVoiceInterface(options: VoiceConfig = {}): UseVoiceInterfaceReturn {
  const [state, setState] = useState<VoiceState>({
    isListening: false,
    isSpeaking: false,
    transcript: '',
    interimTranscript: '',
    error: null,
    confidence: 0,
    isSupported: false
  });

  // Check if voice is enabled and browser supports it
  useEffect(() => {
    if (!voiceConfig.enabled) {
      setState(prev => ({ ...prev, isSupported: false }));
      return;
    }

    // Check Web Speech API support
    const SpeechRecognition = window.SpeechRecognition || (window as any).webkitSpeechRecognition;
    const isSupported = !!SpeechRecognition && 'speechSynthesis' in window;

    setState(prev => ({ ...prev, isSupported }));
  }, []);

  const startListening = useCallback(async () => {
    if (!voiceConfig.enabled) {
      console.warn('Voice interface is disabled. Enable in voice.config.ts');
      return;
    }

    if (!state.isSupported) {
      const error = new Error('Speech recognition not supported in this browser');
      setState(prev => ({ ...prev, error: error.message }));
      options.onError?.(error);
      return;
    }

    // Implementation will be added in Phase 7
    console.log('Voice listening will be implemented in Phase 7');
  }, [state.isSupported, options]);

  const stopListening = useCallback(() => {
    if (!voiceConfig.enabled) return;

    setState(prev => ({ ...prev, isListening: false }));
    // Implementation will be added in Phase 7
  }, []);

  const speak = useCallback((text: string) => {
    if (!voiceConfig.enabled) {
      console.warn('Voice interface is disabled');
      return;
    }

    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = options.language || voiceConfig.defaults.language;
      utterance.rate = 1.0;
      utterance.pitch = 1.0;

      utterance.onstart = () => {
        setState(prev => ({ ...prev, isSpeaking: true }));
      };

      utterance.onend = () => {
        setState(prev => ({ ...prev, isSpeaking: false }));
      };

      window.speechSynthesis.speak(utterance);
    }
  }, [options.language]);

  const cancelSpeaking = useCallback(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setState(prev => ({ ...prev, isSpeaking: false }));
    }
  }, []);

  const reset = useCallback(() => {
    setState({
      isListening: false,
      isSpeaking: false,
      transcript: '',
      interimTranscript: '',
      error: null,
      confidence: 0,
      isSupported: state.isSupported
    });
  }, [state.isSupported]);

  return {
    ...state,
    startListening,
    stopListening,
    speak,
    cancelSpeaking,
    reset
  };
}

export default useVoiceInterface;
