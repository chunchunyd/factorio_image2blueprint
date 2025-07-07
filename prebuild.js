const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// 跨平台删除函数（使用 Node.js 原生方法）
function rmRF(dir) {
  if (fs.existsSync(dir)) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
      console.log(`已删除目录: ${dir}`);
    } catch (error) {
      console.error(`删除目录失败: ${dir}`, error);
    }
  }
}

// 跨平台复制函数（使用 Node.js 原生方法）
function copyFile(src, dest) {
  try {
    // 确保目标目录存在
    const destDir = path.dirname(dest);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    
    // 检查源文件是否存在
    if (!fs.existsSync(src)) {
      throw new Error(`源文件不存在: ${src}`);
    }
    
    // 复制文件
    fs.copyFileSync(src, dest);
    console.log(`已复制文件: ${src} -> ${dest}`);
  } catch (error) {
    console.error(`复制文件失败: ${src} -> ${dest}`, error);
    throw error; // 重新抛出错误以停止构建过程
  }
}

try {
  console.log('开始构建过程...');
  
  // 获取当前工作目录
  const projectRoot = process.cwd();

  // 删除工作目录下的main/main.exe
  const oldMainExePath1 = path.join(projectRoot, 'main.exe');
  const oldMainExePath2 = path.join(projectRoot, 'main');
  if (fs.existsSync(oldMainExePath1)) {
    fs.unlinkSync(oldMainExePath1);
  }
  if (fs.existsSync(oldMainExePath2)) {
    fs.unlinkSync(oldMainExePath2);
  }

  
  // 进入 py 目录
  const pyDir = path.join(projectRoot, 'py');
  process.chdir(pyDir);
  
  // 运行 pyinstaller
  console.log('运行 pyinstaller...');
  execSync('pyinstaller --onefile main.py --noconsole', { stdio: 'inherit' });
  
  // 返回项目根目录
  process.chdir(projectRoot);
  
  // 复制文件 - 使用路径解析确保正确性
  console.log('复制生成的文件...');
  
  // 检查源文件是否存在
  const mainExePath = path.join(pyDir, 'dist', 'main.exe');
  if (!fs.existsSync(mainExePath)) {
    throw new Error(`未找到 main.exe 在: ${mainExePath}`);
  }
  
  const rawSignalsPath = path.join(pyDir, 'raw_signals.json');
  if (!fs.existsSync(rawSignalsPath)) {
    console.warn(`警告: 未找到 raw_signals.json 在: ${rawSignalsPath}`);
  } else {
    copyFile(rawSignalsPath, path.join(projectRoot, 'raw_signals.json'));
  }
  
  // 复制 main.exe
  copyFile(mainExePath, path.join(projectRoot, 'main.exe'));
  
  // 清理目录
  console.log('清理临时文件...');
  rmRF(path.join(pyDir, 'dist'));
  rmRF(path.join(pyDir, 'build'));
  
  console.log('预构建步骤完成！');
} catch (error) {
  console.error('构建过程中出错:', error);
  process.exit(1);
}