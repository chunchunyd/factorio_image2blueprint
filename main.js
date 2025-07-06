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

// 执行 Python 脚本
async function executeBlueprintGenerator(params) {
  return new Promise((resolve, reject) => {
    const {
      imagePath,
      width,
      height,
      title,
      description,
      floorType,
      outputPath
    } = params;

    console.log("Received parameters:");
    console.log("Image Path:", imagePath);
    console.log("Width:", width);
    console.log("Height:", height);
    console.log("Title:", title);
    console.log("Description:", description);
    console.log("Floor Type:", floorType);
    console.log("Output Path:", outputPath);



    // 使用临时文件路径（如果用户没有指定输出路径）
    const finalOutputPath = outputPath || getTempFilePath();
    const isTempFile = !outputPath;

    // 构建命令行参数
    const args = [
      imagePath,
      finalOutputPath,
      '--title', title,
      '--description', description,
      '--floor', floorType,
      '--size', width, height
    ];

    // 获取可执行文件路径（根据打包环境调整）
    let executablePath;
    if (app.isPackaged) {
      // 在生产环境中，可执行文件位于 resources 目录
      executablePath = path.join(process.resourcesPath, 'main.exe');
    } else {
      // 在开发环境中，使用当前目录的可执行文件
      executablePath = path.join(__dirname, 'main.exe');
    }

    console.log("process.resourcesPath:", process.resourcesPath);
    // 确保路径正确
    console.log('Executable path:', executablePath);

    // 执行命令
    const child = spawn(executablePath, args);

    let stdout = '';
    let stderr = '';

    // 收集标准输出
    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    // 收集错误输出
    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // 处理进程结束
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}: ${stderr}`));
        return;
      }

      // 读取输出文件内容
      fs.readFile(finalOutputPath, 'utf8', (err, data) => {
        if (err) {
          reject(err);
          return;
        }

        // 如果是临时文件，读取后立即删除
        if (isTempFile) {
          fs.unlink(finalOutputPath, (unlinkErr) => {
            if (unlinkErr) {
              console.error('Failed to delete temp file:', unlinkErr);
            }
          });
        }

        resolve({
          blueprint: data,
          outputPath: finalOutputPath,
          isTempFile
        });
      });
    });

    // 处理错误
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
    filters: [{ name: 'Images', extensions: ['jpg', 'png', 'gif', 'jpeg'] }]
  });
  return result;
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