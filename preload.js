const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  saveDialog: () => ipcRenderer.invoke('save-dialog'),
  minimizeWindow: () => ipcRenderer.send('minimize-window'),
  closeWindow: () => ipcRenderer.send('close-window'),
  generateThumbnail: (params) => ipcRenderer.invoke('generate-thumbnail', params),
  generateBlueprint: (params) => ipcRenderer.invoke('generate-blueprint', params)
});