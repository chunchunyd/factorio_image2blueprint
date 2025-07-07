const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  saveDialog: () => ipcRenderer.invoke('save-dialog'),
  minimizeWindow: () => ipcRenderer.send('minimize-window'),
  closeWindow: () => ipcRenderer.send('close-window'),
  getMediaInfo: (filePath) => ipcRenderer.invoke('get-media-info', filePath),
  generateBlueprint: (params) => ipcRenderer.invoke('generate-blueprint', params),
  
  // 进度更新相关 - 简化实现
  setupProgressListener: (callback) => {
    ipcRenderer.on('progress-update', (event, data) => {
      callback(data);
    });
  },
  
  removeProgressListener: () => {
    ipcRenderer.removeAllListeners('progress-update');
  },
  
  openFile: (path) => ipcRenderer.invoke('open-file', path),
  showItemInFolder: (path) => ipcRenderer.invoke('show-item-in-folder', path),
  readFile: (path) => ipcRenderer.invoke('read-file', path)
});