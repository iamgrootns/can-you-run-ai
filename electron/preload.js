const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cyrai', {
  detectHardware: () => ipcRenderer.invoke('hw:detect'),
  netStatus: () => ipcRenderer.invoke('net:status'),
  getTags: () => ipcRenderer.invoke('hf:tags'),
  listModels: (opts) => ipcRenderer.invoke('hf:list', opts),
  analyze: (id) => ipcRenderer.invoke('hf:analyze', id),
  getReadme: (id) => ipcRenderer.invoke('hf:readme', id),
  analyzeBatch: (ids) => ipcRenderer.invoke('hf:analyzeBatch', ids),
  recommendations: (force) => ipcRenderer.invoke('hf:recommendations', force),
  listLocal: () => ipcRenderer.invoke('local:list'),
  listCached: () => ipcRenderer.invoke('local:cached'),
  download: (id) => ipcRenderer.invoke('local:download', id),
  cancelDownload: (id) => ipcRenderer.invoke('local:cancelDownload', id),
  deleteLocal: (id) => ipcRenderer.invoke('local:delete', id),
  getChatHistory: (id) => ipcRenderer.invoke('local:getChatHistory', id),
  saveChatHistory: (id, log) => ipcRenderer.invoke('local:saveChatHistory', id, log),
  openDir: (id) => ipcRenderer.invoke('local:openDir', id),
  openUrl: (url) => ipcRenderer.invoke('shell:open', url),
  onDownloadProgress: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on('download:progress', handler);
    return () => ipcRenderer.removeListener('download:progress', handler);
  },
  loadChatModel: (id, opts) => ipcRenderer.invoke('chat:load', id, opts),
  sendChatMessage: (id, message, opts) => ipcRenderer.invoke('chat:send', { modelId: id, message, opts }),
  stopGeneration: (id) => ipcRenderer.invoke('chat:stop', id),
  unloadChatModel: () => ipcRenderer.invoke('chat:unload'),
  onChatChunk: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on('chat:chunk', handler);
    return () => ipcRenderer.removeListener('chat:chunk', handler);
  },
  generateImage: (id, opts) => ipcRenderer.invoke('sd:generate', { modelId: id, opts }),
  cancelGenerate: (id) => ipcRenderer.invoke('sd:cancel', id),
  onGenerateProgress: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on('sd:progress', handler);
    return () => ipcRenderer.removeListener('sd:progress', handler);
  },
  readDataUrl: (filePath) => ipcRenderer.invoke('local:readDataUrl', filePath),
  pickAudioFile: () => ipcRenderer.invoke('dialog:pickAudio'),
  transcribeAudio: (id, audioPath, opts) => ipcRenderer.invoke('asr:transcribe', { modelId: id, audioPath, opts }),
  cancelTranscribe: (id) => ipcRenderer.invoke('asr:cancel', id),
  onTranscribeProgress: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on('asr:progress', handler);
    return () => ipcRenderer.removeListener('asr:progress', handler);
  },
  speakText: (id, text, opts) => ipcRenderer.invoke('tts:speak', { modelId: id, text, opts }),
  cancelSpeak: (id) => ipcRenderer.invoke('tts:cancel', id),
  onSpeakProgress: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on('tts:progress', handler);
    return () => ipcRenderer.removeListener('tts:progress', handler);
  },
  pickImageFile: () => ipcRenderer.invoke('dialog:pickImage'),
  embedText: (id, text) => ipcRenderer.invoke('embed:text', { modelId: id, text }),
  rerankDocuments: (id, query, documents) => ipcRenderer.invoke('rerank:documents', { modelId: id, query, documents }),
  generateVideo: (id, opts) => ipcRenderer.invoke('video:generate', { modelId: id, opts }),
  cancelVideo: (id) => ipcRenderer.invoke('video:cancel', id),
  onVideoProgress: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on('video:progress', handler);
    return () => ipcRenderer.removeListener('video:progress', handler);
  },
  generateMusic: (id, opts) => ipcRenderer.invoke('music:generate', { modelId: id, opts }),
  cancelMusic: (id) => ipcRenderer.invoke('music:cancel', id),
  onMusicProgress: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on('music:progress', handler);
    return () => ipcRenderer.removeListener('music:progress', handler);
  },
});
