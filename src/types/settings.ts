import { App } from 'obsidian';

export interface RecentActivitySettings {
  daysBack: number;
  excludeFolders: string[];
  filterJournalSections: boolean;
  dateHeaderFormat: string;
}

export interface ContextGatheringDefaults {
  linkDepth: number;
  maxCharacters: number;
  filterRecentSectionsOnly: boolean;
}

export type ModelProvider = 'openai' | 'ollama' | 'custom';

export interface OpenAugiSettings {
  apiKey: string;
  modelProvider: ModelProvider;
  customBaseUrl: string;
  defaultModel: string;
  customModelOverride: string;
  summaryFolder: string;
  notesFolder: string;
  promptsFolder: string;
  publishedFolder: string;
  useDataviewIfAvailable: boolean;
  enableDistillLogging: boolean;
  recentActivityDefaults: RecentActivitySettings;
  contextGatheringDefaults: ContextGatheringDefaults;
}

export const DEFAULT_SETTINGS: OpenAugiSettings = {
  apiKey: '',
  modelProvider: 'openai',
  customBaseUrl: '',
  defaultModel: 'gpt-5',
  customModelOverride: '',
  summaryFolder: 'OpenAugi/Summaries',
  notesFolder: 'OpenAugi/Notes',
  promptsFolder: 'OpenAugi/Prompts',
  publishedFolder: 'OpenAugi/Published',
  useDataviewIfAvailable: true,
  enableDistillLogging: false,
  recentActivityDefaults: {
    daysBack: 7,
    excludeFolders: ['Templates', 'Archive', 'OpenAugi'],
    filterJournalSections: true,
    dateHeaderFormat: '### YYYY-MM-DD'
  },
  contextGatheringDefaults: {
    linkDepth: 1,
    maxCharacters: 100000,
    filterRecentSectionsOnly: true
  }
}; 