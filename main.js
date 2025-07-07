const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');



let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 1125,
    resizable: false,
    frame: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,   // 禁用 Node.js 集成
      contextIsolation: true    // 启用上下文隔离
    },
    backgroundColor: '#1a1f29',
    icon: path.join(__dirname, 'assets/icon.png'),
    title: 'Factorio 图片转蓝图'
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));
}

// 生成临时文件路径
function getTempFilePath() {
  const tempDir = app.getPath('temp');
  return path.join(tempDir, `factorio-bp-${Date.now()}.txt`);
}


// 获取媒体文件信息
async function getMediaInfo(filePath) {
  return new Promise((resolve, reject) => {
    // 获取可执行文件路径
    let executablePath;
    if (app.isPackaged) {
      executablePath = path.join(process.resourcesPath, 'main.exe');
    } else {
      executablePath = path.join(__dirname, 'main.exe');
    }

    // 执行命令
    const child = spawn(executablePath, [filePath, '--get-info']);

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`获取媒体信息失败: ${stderr}`));
        return;
      }

      try {
        console.log(stdout);
        const info = JSON.parse(stdout.trim());
        resolve(info);
      } catch (e) {
        reject(new Error('解析媒体信息失败'));
      }
    });

    child.on('error', (err) => {
      reject(err);
    });
  });
}


// 执行 Python 脚本 (支持进度更新)
async function executeBlueprintGenerator(params) {
  return new Promise((resolve, reject) => {
    const {
      mediaPath,
      width,
      height,
      skipFps,
      title,
      description,
      electricSet,
      outputPath
    } = params;

    // 检查输出路径是否提供
    if (!outputPath) {
      reject(new Error("输出路径未指定"));
      return;
    }

    const args = [
      mediaPath,
      '--output', outputPath,
      '--title', title,
      '--description', description,
      '--electric-set', electricSet,
      '--size', width, height,
      '--skip-fps', skipFps.toString()
    ];

    let executablePath;
    if (app.isPackaged) {
      executablePath = path.join(process.resourcesPath, 'main.exe');
    } else {
      executablePath = path.join(__dirname, 'main.exe');
    }

    const child = spawn(executablePath, args);

    // 实时输出处理
    child.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        mainWindow.webContents.send('progress-update', output);
      }
    });

    child.stderr.on('data', (data) => {
      const error = data.toString().trim();
      if (error) {
        mainWindow.webContents.send('progress-update', `[ERROR] ${error}`);
      }
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}`));
        return;
      }

      try {
        const stats = fs.statSync(outputPath);
        const fileSizeByte = stats.size;
        
        resolve({
          success: true,
          outputPath: outputPath,
          fileSizeByte: fileSizeByte.toFixed(2)
        });
      } catch (err) {
        reject(err);
      }
    });

    child.on('error', (err) => {
      reject(err);
    });
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// IPC 处理
ipcMain.handle('open-file-dialog', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ 
      name: '媒体文件', 
      extensions: ['jpg', 'png', 'gif', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'] 
    }]
  });
  return result;
});

// 添加获取媒体信息的方法
ipcMain.handle('get-media-info', async (event, filePath) => {
  try {
    const info = await getMediaInfo(filePath);
    return info;
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('save-dialog', async () => {
  const result = await dialog.showSaveDialog({
    title: '保存蓝图文件',
    filters: [{ name: 'Text Files', extensions: ['txt'] }]
  });
  return result;
});

ipcMain.on('minimize-window', () => {
  mainWindow.minimize();
});

ipcMain.on('close-window', () => {
  mainWindow.close();
});

// 处理蓝图生成请求
ipcMain.handle('generate-blueprint', async (event, params) => {
  try {
    const result = await executeBlueprintGenerator(params);
    return { success: true, ...result };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// // 添加进度监听器管理
// const progressListeners = new Map();

// ipcMain.on('register-progress-listener', (event, listenerId) => {
//   const listener = (event, data) => {
//     event.sender.send('progress-update', data);
//   };
  
//   progressListeners.set(listenerId, listener);
//   ipcMain.on('progress-update', listener);
// });

// ipcMain.on('remove-progress-listener', (event, listenerId) => {
//   const listener = progressListeners.get(listenerId);
//   if (listener) {
//     ipcMain.removeListener('progress-update', listener);
//     progressListeners.delete(listenerId);
//   }
// });

// 在IPC处理部分添加
ipcMain.handle('open-file', (event, path) => {
  const { shell } = require('electron');
  return shell.openPath(path);
});

ipcMain.handle('show-item-in-folder', (event, path) => {
  const { shell } = require('electron');
  return shell.showItemInFolder(path);
});

ipcMain.handle('read-file', async (event, path) => {
  try {
    return await fs.promises.readFile(path, 'utf8');
  } catch (error) {
    throw new Error(`读取文件失败: ${error.message}`);
  }
});