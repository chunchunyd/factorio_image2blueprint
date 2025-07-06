// 窗口控制功能
document.querySelector('.minimize').addEventListener('click', () => {
    window.electron.minimizeWindow();
});

document.querySelector('.close').addEventListener('click', () => {
    window.electron.closeWindow();
});


// 页面元素
const imagePathInput = document.getElementById('imagePath');
const browseBtn = document.getElementById('browseBtn');
const imagePreview = document.getElementById('imagePreview');
const previewBox = document.getElementById('previewBox');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const widthInput = document.getElementById('width');
const heightInput = document.getElementById('height');
const keepRatioCheckbox = document.getElementById('keepAspectRatio');
const titleInput = document.getElementById('title');
const descriptionInput = document.getElementById('description');
const floorTypeSelect = document.getElementById('floorType');
const outputPathInput = document.getElementById('outputPath');
const savePathBtn = document.getElementById('savePathBtn');
const convertBtn = document.getElementById('convertBtn');
const blueprintOutput = document.getElementById('blueprintOutput');
const copyBtn = document.getElementById('copyBtn');
const statusMessage = document.getElementById('statusMessage');


// 变量
let imageWidth = 0;
let imageHeight = 0;
let aspectRatio = 1;

// 工具函数
function getFileUrl(path) {
    return `file://${path.replace(/\\/g, '/')}`;
}

function getFilename(path) {
    return path.split(/[\\/]/).pop();
}

// 处理宽度变化
function handleWidthChange() {
    // 超过原尺寸时设置为原尺寸
    if (widthInput.value > imageWidth) {
        widthInput.value = imageWidth;
        showStatus("宽度不能超过原图片宽度", "error");
    }
    if (!keepRatioCheckbox.checked || aspectRatio <= 0) return;

    const newWidth = parseFloat(widthInput.value);
    heightInput.value = Math.round(newWidth / aspectRatio);
}

// 处理高度变化
function handleHeightChange() {
    // 超过原尺寸时设置为原尺寸
    if (heightInput.value > imageHeight) {
        heightInput.value = imageHeight;
        showStatus("高度不能超过原图片高度", "error");
    }
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
        const currentWidth = parseFloat(widthInput.value) || imageWidth;
        heightInput.value = Math.round(currentWidth / aspectRatio);
    }
});

// 浏览图片文件
browseBtn.addEventListener('click', () => {
    window.electron.openFileDialog().then(result => {
        if (!result.canceled && result.filePaths.length > 0) {
            const imagePath = result.filePaths[0];
            imagePathInput.value = imagePath;

            // 显示预览
            previewPlaceholder.style.display = 'none';
            imagePreview.src = imagePath;
            imagePreview.style.display = 'block';

            const img = new Image();
            img.src = getFileUrl(imagePath);

            img.onload = () => {
                imageWidth = img.naturalWidth;
                imageHeight = img.naturalHeight;
                aspectRatio = imageWidth / imageHeight;

                widthInput.value = imageWidth;
                heightInput.value = imageHeight;

                showStatus(`已选择: ${getFilename(imagePath)} (${imageWidth}×${imageHeight})`, "success");
            };


            img.onerror = () => {
                showStatus("图片加载失败", "error");
            };
        }
    });
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
    const imagePath = imagePathInput.value;
    const width = parseInt(widthInput.value) || 0;
    const height = parseInt(heightInput.value) || 0;
    const title = titleInput.value || "图片";
    const description = descriptionInput.value || "一张图片";
    const floorType = floorTypeSelect.value || "nothing";
    const outputPath = outputPathInput.value;

    // 验证输入
    if (!imagePath) {
        showStatus("请先选择图片文件", "error");
        return;
    }

    if (width < 1 && height < 1) {
        showStatus("宽度和高度不能同时为0", "error");
        return;
    }

    // 更新按钮状态
    convertBtn.disabled = true;
    convertBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 生成中...';
    showStatus("正在处理图片并生成蓝图...", "processing");

    try {
        // 调用蓝图生成函数
        const result = await window.electron.generateBlueprint({
            imagePath,
            width: width.toString(),
            height: height.toString(),
            title,
            description,
            floorType,
            outputPath
        });

        if (!result.success) {
            throw new Error(result.error);
        }

        // 显示蓝图
        blueprintOutput.value = result.blueprint;

        // 显示成功消息
        if (result.isTempFile) {
            showStatus("蓝图生成成功！内容已复制到输出框", "success");
        } else {
            showStatus(`蓝图生成成功！已保存到: ${result.outputPath}`, "success");
        }

    } catch (error) {
        console.error(error);
        showStatus("生成蓝图时出错: " + error.message, "error");
    } finally {
        // 恢复按钮状态
        convertBtn.disabled = false;
        convertBtn.innerHTML = '<i class="fas fa-bolt"></i> 生成蓝图';
    }
});

copyBtn.addEventListener('click', async () => {
    if (!blueprintOutput.value) {
        showStatus("没有蓝图可复制", "error");
        return;
    }

    try {
        // 使用 Clipboard API 写入文本
        await navigator.clipboard.writeText(blueprintOutput.value);

        // 显示复制成功消息
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i> 已复制';
        showStatus("蓝图已复制到剪贴板", "success");

        // 2秒后恢复按钮文本
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
