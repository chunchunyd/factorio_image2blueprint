import json
import base64
import zlib
from PIL import Image, ImageFilter, ImageOps
import argparse

# 通电地板类型
FLOOR_TYPE = ["nothing", "space-platform-foundation", "F077ET-refined-concrete", "F077ET-concrete", "F077ET-stone-path"]


def encode_blueprint_json(bp_json: dict) -> str:
    """
    将蓝图 JSON 编码为字符串
    """
    json_str = json.dumps(bp_json)
    compressed_data = zlib.compress(json_str.encode("utf-8"), level=9)
    encoded_data = base64.b64encode(compressed_data).decode("utf-8")
    return "0" + encoded_data


def build_blueprint_from_image(image: Image.Image, title: str, description: str, floor_type: str = "nothing") -> str:
    """
    从 PIL 图像构建蓝图
    """
    bp_json = {
        "blueprint": {
            "description": description,
            "item": "blueprint",
            "icons": [{"signal": {"name": "small-lamp"}, "index": 1}],
            "entities": [],
            "tiles": [],
            "label": title,
            "version": 562949957025792,
        }
    }
    
    # 获取图像尺寸
    width, height = image.size
    
    # 添加通电地板
    if floor_type and floor_type in FLOOR_TYPE and floor_type != "nothing":
        for y in range(height):
            for x in range(width):
                bp_json["blueprint"]["tiles"].append({
                    "name": floor_type, 
                    "position": [x, y]
                })
    
    # 添加小灯（使用图像像素颜色）
    pixels = image.load()  # 获取像素访问对象
    entity_number = 1
    
    for y in range(height):
        for x in range(width):
            # 获取像素RGB值
            r, g, b = pixels[x, y][:3]  # 兼容RGBA格式
            
            bp_json["blueprint"]["entities"].append({
                "entity_number": entity_number,
                "name": "small-lamp",
                "position": {"x": x + 0.5, "y": y + 0.5},
                "color": {
                    "r": r / 255.0,
                    "g": g / 255.0,
                    "b": b / 255.0,
                    "a": 1,
                },
                "always_on": True,
            })
            entity_number += 1

    return encode_blueprint_json(bp_json)


def downsample_for_pixel_art(input_path, target_size):
    """降采样"""
    img = Image.open(input_path)
    
    
    # 新增：应用 EXIF 方向标签（解决 JPG 旋转问题）
    img = ImageOps.exif_transpose(img)

    # 等比例缩放
    if target_size[0] == 0:
        target_size = (int(target_size[1] * img.width / img.height), target_size[1])
    elif target_size[1] == 0:
        target_size = (target_size[0], int(target_size[0] * img.height / img.width))

    # 第一步：用Lanczos+锐化保留细节
    if max(img.size) > 4 * max(target_size):
        intermediate_size = (img.width // 2, img.height // 2)
        img = img.resize(intermediate_size, Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

    # 第二步：最近邻插值降至目标分辨率
    img = img.resize(target_size, Image.NEAREST)

    return img.convert("RGB")  # 确保返回 RGB 模式


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Convert image to Factorio blueprint.")
    parser.add_argument("input", type=str, help="Input image file path.")
    parser.add_argument("output", type=str, help="Output blueprint file path.")
    parser.add_argument("--title", type=str, default="Converted Blueprint", help="Blueprint title.")
    parser.add_argument("--description", type=str, default="Converted from image.", help="Blueprint description.")
    parser.add_argument("--floor", type=str, choices=FLOOR_TYPE, 
                        help="Floor type for blueprint tiles. Options: " + ", ".join(FLOOR_TYPE), 
                        default="nothing")
    parser.add_argument("--size", type=int, nargs=2, default=(256, 0), 
                        help="Target size for pixel art (width height). Use 0 for auto scaling.")
    
    args = parser.parse_args()
    
    # 处理图像
    image = downsample_for_pixel_art(args.input, tuple(args.size))
    
    # 构建蓝图
    blueprint = build_blueprint_from_image(
        image, 
        args.title, 
        args.description, 
        args.floor
    )
    
    # 写入输出文件
    with open(args.output, "w") as f:
        f.write(blueprint)
    
    print(f"Blueprint saved to {args.output}")
    print(f"Final image size: {image.size[0]}x{image.size[1]} pixels")
    print(f"Total lamps: {image.size[0] * image.size[1]}")