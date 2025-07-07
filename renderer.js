// 窗口控制功能
document.querySelector('.minimize').addEventListener('click', () => {
    window.electron.minimizeWindow();
});

document.querySelector('.close').addEventListener('click', () => {
    window.electron.closeWindow();
});

// 文件大小阈值
const SIZE_THRESHOLD = 50 * 1024 * 1024; // 50MB

// 页面元素
const mediaPathInput = document.getElementById('mediaPath');
const browseBtn = document.getElementById('browseBtn');
const mediaInfo = document.getElementById('mediaInfo');
const widthInput = document.getElementById('width');
const heightInput = document.getElementById('height');
const keepRatioCheckbox = document.getElementById('keepAspectRatio');
const skipFpsInput = document.getElementById('skipFps');
const titleInput = document.getElementById('title');
const descriptionInput = document.getElementById('description');
const electricSetSelect = document.getElementById('electricSet');
const outputPathInput = document.getElementById('outputPath');
const savePathBtn = document.getElementById('savePathBtn');
const convertBtn = document.getElementById('convertBtn');
const blueprintOutput = document.getElementById('blueprintOutput');
const copyBtn = document.getElementById('copyBtn');
const statusMessage = document.getElementById('statusMessage');

const openFileBtn = document.getElementById('openFileBtn');
const openFolderBtn = document.getElementById('openFolderBtn');

// 在页面加载时初始化按钮状态
openFileBtn.style.display = 'none';

// 变量
let mediaWidth = 0;
let mediaHeight = 0;
let aspectRatio = 1;
let lastOutputPath = '';

// 工具函数
function getFilename(path) {
    return path.split(/[\\/]/).pop();
}

// 处理宽度变化
function handleWidthChange() {
    if (!keepRatioCheckbox.checked || aspectRatio <= 0) return;
    const newWidth = parseFloat(widthInput.value);
    heightInput.value = Math.round(newWidth / aspectRatio);
}

// 处理高度变化
function handleHeightChange() {
    if (!keepRatioCheckbox.checked || aspectRatio <= 0) return;
    const newHeight = parseFloat(heightInput.value);
    widthInput.value = Math.round(newHeight * aspectRatio);
}

// 防抖函数
function debounce(func, delay) {
    let timer;
    return function () {
        clearTimeout(timer);
        timer = setTimeout(() => func.apply(this, arguments), delay);
    };
}

// 绑定事件
widthInput.addEventListener('input', debounce(handleWidthChange, 100));
heightInput.addEventListener('input', debounce(handleHeightChange, 100));

// 当保持比例复选框变化时
keepRatioCheckbox.addEventListener('change', () => {
    if (keepRatioCheckbox.checked && aspectRatio > 0) {
        const currentWidth = parseFloat(widthInput.value) || mediaWidth;
        heightInput.value = Math.round(currentWidth / aspectRatio);
    }
});

// 浏览媒体文件
browseBtn.addEventListener('click', async () => {
    const result = await window.electron.openFileDialog();
    if (!result.canceled && result.filePaths.length > 0) {
        const mediaPath = result.filePaths[0];
        mediaPathInput.value = mediaPath;

        // 显示加载状态
        showStatus("正在获取媒体文件信息...", "processing");
        mediaInfo.textContent = "加载中...";

        try {
            // 获取媒体文件信息
            const info = await window.electron.getMediaInfo(mediaPath);

            if (info.error) {
                showStatus(info.error, "error");
                mediaInfo.textContent = "获取信息失败";
                return;
            }

            // 更新媒体信息显示
            if (info.type === 'video') {
                mediaInfo.innerHTML = `
                    <strong>视频信息:</strong> 
                    ${info.width}×${info.height} 分辨率, 
                    ${info.frame_count} 帧, 
                    ${info.fps.toFixed(2)} FPS
                `;

                // 自动设置标题和描述为视频相关
                titleInput.value = "视频";
                descriptionInput.value = "一个视频";
            } else {
                mediaInfo.innerHTML = `
                    <strong>图片信息:</strong> 
                    ${info.width}×${info.height} 分辨率
                `;

                // 自动设置标题和描述为图片相关
                titleInput.value = "图片";
                descriptionInput.value = "一张图片";
            }

            // 设置尺寸输入框
            mediaWidth = info.width;
            mediaHeight = info.height;
            aspectRatio = mediaWidth / mediaHeight;

            widthInput.value = mediaWidth;
            heightInput.value = mediaHeight;

            showStatus(`已选择: ${getFilename(mediaPath)}`, "success");

        } catch (error) {
            console.error(error);
            showStatus("获取媒体信息失败: " + error.message, "error");
            mediaInfo.textContent = "获取信息失败";
        }
    }
});

// 设置保存路径
savePathBtn.addEventListener('click', () => {
    window.electron.saveDialog().then(result => {
        if (!result.canceled && result.filePath) {
            outputPathInput.value = result.filePath;
            showStatus("已设置保存位置", "success");
        }
    });
});

// 生成蓝图
convertBtn.addEventListener('click', async () => {
    // 获取输入值
    const mediaPath = mediaPathInput.value;
    const width = parseInt(widthInput.value) || 0;
    const height = parseInt(heightInput.value) || 0;
    const skipFps = parseInt(skipFpsInput.value) || 1;
    const title = titleInput.value || "视频";
    const description = descriptionInput.value || "一个视频";
    const electricSet = electricSetSelect.value || "nothing";
    const outputPath = outputPathInput.value;

    // 验证输入
    if (!mediaPath) {
        showStatus("请先选择媒体文件", "error");
        return;
    }

    if (width < 1 || height < 1) {
        showStatus("宽度和高度必须大于0", "error");
        return;
    }

    // 更新按钮状态
    convertBtn.disabled = true;
    convertBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 生成中...';
    showStatus("正在处理媒体文件并生成蓝图...", "processing");

    // 清空输出框并添加初始消息
    blueprintOutput.value = ">>> 开始处理媒体文件...\n";
    blueprintOutput.scrollTop = blueprintOutput.scrollHeight;

    try {
        // 设置进度监听器
        window.electron.setupProgressListener((data) => {
            blueprintOutput.value += data + '\n';
            blueprintOutput.scrollTop = blueprintOutput.scrollHeight;
        });

        // 调用蓝图生成函数
        const result = await window.electron.generateBlueprint({
            mediaPath,
            width: width.toString(),
            height: height.toString(),
            skipFps: skipFps.toString(),
            title,
            description,
            electricSet,
            outputPath
        });

        if (!result.success) {
            throw new Error(result.error);
        }

        // 保存输出路径
        lastOutputPath = result.outputPath;

        // 检查文件大小
        if (result.fileSizeByte > SIZE_THRESHOLD) {
            blueprintOutput.value = `>>> 蓝图生成成功！文件大小 ${result.fileSizeByte}MB，超过显示限制\n`;
            blueprintOutput.value += `>>> 文件已保存至: ${result.outputPath}\n`;
            blueprintOutput.value += '>>> 请使用下方的"打开文件"按钮查看';

            // 显示打开文件按钮
            openFileBtn.style.display = 'inline-block';
            copyBtn.style.display = 'none';
            openFolderBtn.style.display = 'inline-block';

            showStatus(`蓝图生成成功！文件大小 ${result.fileSizeByte}MB`, "success");
        } else {
            // 小文件通过主进程读取内容
            const content = await window.electron.readFile(result.outputPath);
            blueprintOutput.value = content;

            // 显示复制按钮
            openFileBtn.style.display = 'none';
            copyBtn.style.display = 'inline-block';
            openFolderBtn.style.display = 'inline-block';

            showStatus("蓝图生成成功！内容已显示", "success");
        }
    } catch (error) {
        console.error(error);
        showStatus("生成蓝图时出错: " + error.message, "error");
        blueprintOutput.value += `>>> 错误: ${error.message}\n`;

        // 确保移除进度监听器
        window.electron.removeProgressListener();
    } finally {
        // 恢复按钮状态
        convertBtn.disabled = false;
        convertBtn.innerHTML = '<i class="fas fa-bolt"></i> 生成蓝图';
    }
});

// 添加打开文件功能
openFileBtn.addEventListener('click', async () => {
    if (!lastOutputPath) {
        showStatus("没有可打开的文件", "error");
        return;
    }

    try {
        await window.electron.openFile(lastOutputPath);
        showStatus("已打开文件", "success");
    } catch (error) {
        showStatus("打开文件失败: " + error.message, "error");
    }
});

// 添加打开所在文件夹功能
openFolderBtn.addEventListener('click', async () => {
    if (!lastOutputPath) {
        showStatus("没有可打开的文件", "error");
        return;
    }

    try {
        await window.electron.showItemInFolder(lastOutputPath);
        showStatus("已打开所在文件夹", "success");
    } catch (error) {
        showStatus("打开文件夹失败: " + error.message, "error");
    }
});

copyBtn.addEventListener('click', async () => {
    if (!blueprintOutput.value) {
        showStatus("没有蓝图可复制", "error");
        return;
    }

    try {
        await navigator.clipboard.writeText(blueprintOutput.value);
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i> 已复制';
        showStatus("蓝图已复制到剪贴板", "success");
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
        }, 2000);
    } catch (err) {
        console.error("复制失败:", err);
        showStatus("复制失败，请手动复制", "error");
    }
});

// 显示状态消息
function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = "status " + type;
}