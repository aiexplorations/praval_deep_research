/**
 * Voice Interface Configuration
 *
 * This configuration controls voice features throughout the application.
 * Voice is currently DISABLED (Phase 7 feature).
 *
 * The scaffolding is in place to enable voice features later without refactoring.
 */

export interface VoiceProvider {
  name: 'webspeech' | 'openai-whisper' | 'whisper-wasm';
  stt: 'available' | 'unavailable';
  tts: 'available' | 'unavailable';
}

export interface VoiceConfig {
  enabled: boolean;

  providers: {
    webSpeech: {
      enabled: boolean;
      stt: boolean;
      tts: boolean;
      default: boolean;
    };
    openaiWhisper: {
      enabled: boolean;
      stt: boolean;
      tts: boolean;
      apiKey?: string;
    };
    localWhisper: {
      enabled: boolean;
      stt: boolean;
      tts: boolean;
      modelPath?: string;
    };
  };

  features: {
    voiceSearch: boolean;
    voiceQA: boolean;
    voiceCommands: boolean;
    voiceNotes: boolean;
    handsFreeMode: boolean;
  };

  defaults: {
    language: string;
    interimResults: boolean;
    continuousListening: boolean;
    autoSpeak: boolean;
    noiseCancellation: boolean;
  };
}

export const voiceConfig: VoiceConfig = {
  // Master switch - DISABLED until Phase 7
  enabled: false,

  providers: {
    webSpeech: {
      enabled: true,
      stt: true,
      tts: true,
      default: true
    },
    openaiWhisper: {
      enabled: false,  // Phase 2 (optional upgrade)
      stt: true,
      tts: true,
      apiKey: import.meta.env.VITE_OPENAI_API_KEY
    },
    localWhisper: {
      enabled: false,  // Phase 3 (local, private)
      stt: true,
      tts: false,
      modelPath: '/models/whisper-tiny.en.bin'
    }
  },

  features: {
    voiceSearch: false,    // Enable in Phase 7
    voiceQA: false,        // Enable in Phase 7
    voiceCommands: false,  // Future feature
    voiceNotes: false,     // Future feature
    handsFreeMode: false   // Future feature
  },

  defaults: {
    language: 'en-US',
    interimResults: true,
    continuousListening: false,
    autoSpeak: false,  // Don't auto-speak answers by default
    noiseCancellation: true
  }
};

export default voiceConfig;
