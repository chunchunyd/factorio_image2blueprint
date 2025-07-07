import cv2
import numpy as np
import json
import base64
import zlib
from PIL import Image, ImageFilter, ImageOps
import argparse
import os

# 供电方法
ELECTRIC_SET = [
    "nothing",
    "space-platform-foundation",
    "F077ET-refined-concrete",
    "F077ET-concrete",
    "F077ET-stone-path",
    "substation1",
    "substation5",
]



def get_image_info(image_path: str) -> tuple[int, int]:
    "获取图片的宽、高"
    image = Image.open(image_path)
    width, height = image.size
    return width, height


# 获取视频信息
def get_video_info(video_path: str) -> tuple[int, int, int, int]:
    "获取视频的帧率、宽、高、帧数"
    input_video = cv2.VideoCapture(video_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    input_video.release()
    return round(fps), int(width), int(height), int(frame_count)


# 视频->指定分辨率图片
def video_to_images(video_path: str, target_size: tuple[int, int]) -> tuple[list[np.ndarray], int]:
    "将视频转换为图片序列"
    input_video = cv2.VideoCapture(video_path)
    images = []

    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        images.append(frame)

    fps = round(input_video.get(cv2.CAP_PROP_FPS))

    input_video.release()
    return images, fps


def encode_blueprint(bp_json: dict) -> str:
    """
    将蓝图 JSON 编码为字符串
    """
    # 将蓝图 JSON 转换为字符串
    json_str = json.dumps(bp_json)
    # 使用 zlib 压缩字符串
    compressed_data = zlib.compress(json_str.encode("utf-8"), level=9)
    # 使用 base64 编码压缩后的数据
    encoded_data = base64.b64encode(compressed_data).decode("utf-8")
    # 在编码后的数据前添加一个 "0" 字符
    return "0" + encoded_data


### 图片生成蓝图部分 -------------------------------------------------------------
def build_blueprint_from_image(image: Image.Image, title: str, description: str, electric_set: str = "nothing") -> str:
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
    if electric_set and electric_set != "nothing":
        if electric_set in ELECTRIC_SET[1:5]:
            for y in range(height):
                for x in range(width):
                    bp_json["blueprint"]["tiles"].append({"name": electric_set, "position": [x, y]})
        elif electric_set in ELECTRIC_SET[5:]:
            raise NotImplementedError("Substation not implemented")
    # 添加小灯（使用图像像素颜色）
    pixels = image.load()  # 获取像素访问对象
    entity_number = 1

    for y in range(height):
        for x in range(width):
            # 获取像素RGB值
            r, g, b = pixels[x, y][:3]  # 兼容RGBA格式

            bp_json["blueprint"]["entities"].append(
                {
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
                }
            )
            entity_number += 1

    return encode_blueprint(bp_json)


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


### 视频生成蓝图部分 -------------------------------------------------------------
quality_rank = ["normal", "uncommon", "rare", "epic", "legendary"]


class PixelLamp:
    "像素灯"

    def __init__(self, x: int, y: int, entity_number: int, signal: dict = {"type": "virtual", "name": "signal-white"}):
        "参数是左上角坐标，self.x和self.y是中心坐标"
        self.entity_number = entity_number
        self.x = x + 0.5
        self.y = y + 0.5
        self.signal = signal
        self.controlled = True

    def add_signal(self, signal: dict):
        self.signal = signal

    def to_dict(self):
        if self.controlled:
            return {
                "entity_number": self.entity_number,
                "name": "small-lamp",
                "position": {"x": self.x, "y": self.y},
                "control_behavior": {"use_colors": True, "rgb_signal": self.signal, "color_mode": 2},
                "always_on": True,
            }
        else:
            return {
                "entity_number": self.entity_number,
                "name": "small-lamp",
                "position": {"x": self.x, "y": self.y},
                "color": {"r": 1, "g": 1, "b": 1},
                "always_on": True,
            }


class Substation:
    "广域配电站"

    def __init__(self, x: int, y: int, entity_number: int, quality: int = 0):
        self.entity_number = entity_number
        self.x = x + 1
        self.y = y + 1
        self.quality = quality
        self.range = [18, 20, 22, 24, 28]

    def to_dict(self):
        return {
            "entity_number": self.entity_number,
            "name": "substation",
            "position": {"x": self.x, "y": self.y},
        }


class ConstantCombinator:
    "常量运算器"

    def __init__(self, x: int, y: int, entity_number: int):
        self.entity_number = entity_number
        self.x = x + 0.5
        self.y = y + 0.5
        self.signals = []

    def add_signal(self, signal: dict, value: int = 1):
        self.signals.append(
            {
                "index": len(self.signals) + 1,
                "name": signal["name"],
                "quality": signal["quality"],
                "comparator": "=",
                "count": value,
            }
        )
        if "type" in signal and signal["type"] != "item":
            self.signals[-1]["type"] = signal["type"]

    def to_dict(self):
        return {
            "entity_number": self.entity_number,
            "name": "constant-combinator",
            "position": {"x": self.x, "y": self.y},
            "direction": 4,
            "control_behavior": {
                "sections": {
                    "sections": [
                        {
                            "index": 1,
                            "filters": self.signals,
                        }
                    ]
                }
            },
        }


class DeciderCombinator:
    "判断运算器"

    def __init__(self, x: int, y: int, entity_number: int, direction: int = 0):
        self.entity_number = entity_number
        if direction not in [0, 4, 8, 12]:
            raise ValueError("direction只取0, 4, 8, 12, 分别为北、东、南、西")
        self.direction = direction
        if direction in [0, 8]:
            self.x = x + 0.5
            self.y = y + 1
        else:
            self.x = x + 1
            self.y = y + 0.5
        self.conditions = []
        self.outputs = []

    def add_condition(
        self, first_signal: dict, second_signal: dict, constant: int, comparator: str = "=", and_last: bool = False
    ):
        new_condition = {}
        if first_signal:
            new_condition["first_signal"] = first_signal
        if second_signal:
            new_condition["second_signal"] = second_signal
        elif constant:
            new_condition["constant"] = constant
        new_condition["comparator"] = comparator
        if and_last:
            new_condition["compare_type"] = "and"  # and_last为True时，表示与上一个条件进行and运算，否则为or运算
        self.conditions.append(new_condition)

    def add_output(
        self, signal: dict, value: int = 0, copy_count_from_red: bool = False, copy_count_from_green: bool = False
    ):
        output = {"signal": signal}
        if not copy_count_from_red and not copy_count_from_green:
            output["constant"] = value
            output["copy_count_from_input"] = False
        elif copy_count_from_red and copy_count_from_green:
            return
        else:
            output["networks"] = {"red": copy_count_from_red, "green": copy_count_from_green}
        self.outputs.append(output)

    def add_outputs_with_value(self, signals: list[dict], values: list[int] = None):
        self.outputs.extend(
            [
                {"signal": signal, "copy_count_from_input": False, "constant": value}
                for signal, value in zip(signals, values)
            ]
        )

    def to_dict(self):
        return {
            "entity_number": self.entity_number,
            "name": "decider-combinator",
            "position": {"x": self.x, "y": self.y},
            "direction": self.direction,
            "control_behavior": {
                "decider_conditions": {
                    "conditions": self.conditions,
                    "outputs": self.outputs,
                }
            },
        }


class DisplaySection:
    "显示面板"

    def __init__(self, x: int, y: int, width: int, height: int, entity_number_start: int):
        "x,y为左上角坐标，index为显示面板编号, entity_number_start为实体编号起始值"
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.entity_number_start = entity_number_start
        self.lamps: list[PixelLamp] = []
        self.deciders: list[DeciderCombinator] = []
        self.wires = []
        self.add_lamps()

    def add_lamps(self):
        for i in range(self.width):
            for j in range(self.height):
                entity_number = self.entity_number_start + i * self.height + j
                self.lamps.append(
                    PixelLamp(
                        self.x + i,
                        self.y + j,
                        entity_number,
                    )
                )
                if i == 0 and j == 0:
                    continue
                elif j == 0:  # 添加横向连接线
                    self.wires.append([self.lamps[-1].entity_number, 2, self.lamps[-1 - self.height].entity_number, 2])
                else:  # 添加竖向连接线
                    self.wires.append([self.lamps[-1].entity_number, 2, self.lamps[-2].entity_number, 2])

    def arrage_signals_for_lamps(self, signals: list[dict]):
        for idx, lamp in enumerate(self.lamps):
            lamp.add_signal(signals[idx])

    def add_combinators(self, frame_count: int):
        num_y = frame_count // self.width
        for i in range(frame_count):
            x = i % self.width
            y = 2 * (i // self.width)
            self.deciders.append(
                DeciderCombinator(
                    self.x + x,
                    self.y + self.height + y,
                    self.entity_number_start + len(self.lamps) + i,
                    0,
                )
            )
            if i == 0:  # 连接最近的lamp
                self.wires.append([self.deciders[-1].entity_number, 4, self.lamps[self.height - 1].entity_number, 2])
            elif x == 0:  # 连接上一行
                self.wires.append([self.deciders[-1].entity_number, 1, self.deciders[-1 - self.width].entity_number, 1])
                self.wires.append([self.deciders[-1].entity_number, 4, self.deciders[-1 - self.width].entity_number, 4])
            else:  # 连接上一列
                self.wires.append([self.deciders[-1].entity_number, 1, self.deciders[-2].entity_number, 1])
                self.wires.append([self.deciders[-1].entity_number, 4, self.deciders[-2].entity_number, 4])

    def to_list(self):
        return [lamp.to_dict() for lamp in self.lamps] + [decider.to_dict() for decider in self.deciders]


class Frames2Blueprint:
    def __init__(
        self,
        images: list[np.ndarray],
        fps: int,
        width: int,
        height: int,
        electric_set: str = "nothing",
        title: str = "video player",
        description: str = "video player",
        signal_mask: list[str] = [],
    ):
        "signal_mask: 删除不想要的信号名，添加有些mod可能导致比原版少一些信号"
        self.images = images
        self.fps = fps
        self.width = width
        self.height = height
        self.delta_timetick = 60 / fps
        
        if electric_set not in ELECTRIC_SET:
            raise ValueError(f"electric_set {electric_set} not in {ELECTRIC_SET}")
        
        # todo
        if electric_set.startswith("substation"):
            raise NotImplementedError("Substation not implemented")
        
        self.electric_set = electric_set

        self.signal_mask = signal_mask

        self.blueprint = {
            "blueprint": {
                "icons": [{"signal": {"name": "display-panel"}, "index": 1}],
                "entities": [],
                "wires": [],
                "tiles": [],
                "item": "blueprint",
                "label": title,
                "description": description,
                "version": 562949957025792,
            }
        }
        self.all_signals = []
        self.timer_signal = None

        self.section_width = None
        self.section_size = None
        self.section_count = None
        self.padding = None

        self.display_sections: list[DisplaySection] = []
        self.timer: tuple[ConstantCombinator, DeciderCombinator] = ()  # 一个常量 + 一个判断
        self.wires = []

        self.current_entity_number = 1

        self.preprocess_images()

    def preprocess_images(self):
        if self.fps > 60 and self.fps < 80:
            self.fps = 60  # 比60多不算太多的话，就按60帧处理
        elif self.fps > 80:
            cut_rate = (self.fps - 1) // 60 + 1
            self.images = self.images[::cut_rate]
            self.fps /= cut_rate
        self.delta_timetick = 60 / self.fps

    def init_signals(self, timer_signal: dict):
        raw_signal_path = os.path.join(resources_directory, 'raw_signals.json')

        with open(raw_signal_path, "r") as f:
            raw_signals = json.load(f)

        self.all_signals = [
            {
                "type": catagory,
                "name": name,
                "quality": quality,
            }
            for quality in quality_rank
            for catagory, signals in raw_signals.items()
            for name in signals
            if not (name in self.signal_mask or (name == timer_signal["name"] and quality == timer_signal["quality"]))
        ]

        self.timer_signal = timer_signal

    def set_up_timer(self):
        time_period = self.delta_timetick * len(self.images)

        constant_combinator = ConstantCombinator(-1, self.height, self.current_entity_number)
        constant_combinator.add_signal(self.timer_signal)

        decider_combinator = DeciderCombinator(-1, self.height + 1, self.current_entity_number + 1, 0)
        decider_combinator.add_condition(self.timer_signal, None, time_period, "<")
        decider_combinator.add_output(self.timer_signal, 0, True, False)

        self.timer = (constant_combinator, decider_combinator)

        self.wires.append([self.timer[0].entity_number, 1, self.timer[1].entity_number, 1])
        self.wires.append([self.timer[1].entity_number, 1, self.timer[1].entity_number, 3])  # 判断器的输入输出端相连

        self.current_entity_number += 2

    def cut_into_sections(self):
        self.section_width = len(self.all_signals) // self.height
        if self.section_width > self.width:
            self.section_width = self.width
        self.section_size = self.section_width * self.height
        self.all_signals = self.all_signals[: self.section_size]
        self.section_count = (self.width - 1) // self.section_width + 1
        self.padding = (self.section_count * self.section_width - self.width) // 2
        for i in range(self.section_count):
            display_section = DisplaySection(
                0 + i * self.section_width, 0, self.section_width, self.height, self.current_entity_number
            )
            display_section.arrage_signals_for_lamps(self.all_signals)
            display_section.add_combinators(len(self.images))
            self.display_sections.append(display_section)

            self.current_entity_number += len(display_section.lamps) + len(display_section.deciders)

            if i == 0:  # 与计时器相连
                self.wires.append(
                    [self.timer[1].entity_number, 1, self.display_sections[-1].deciders[0].entity_number, 1]
                )
            else:  # 与上一列相连
                self.wires.append(
                    [
                        self.display_sections[-2].deciders[self.section_width - 1].entity_number,
                        1,
                        self.display_sections[-1].deciders[0].entity_number,
                        1,
                    ]
                )

            self.wires.extend(display_section.wires)

    # def get_specific_combinator(self, index: int):

    def set_combinator_signals(self):
        # 预计算信号映射表（每个section使用完整的信号集）
        # signal_map = {}
        # for section_idx in range(self.section_count):
        #     start_idx = section_idx * self.section_size
        #     end_idx = start_idx + self.section_size
        #     signal_map[section_idx] = self.all_signals[start_idx:end_idx]

        # 设置条件（保持不变）
        for section_idx, section in enumerate(self.display_sections):
            time_ptr = 0
            for combinator_idx, combinator in enumerate(section.deciders):
                if combinator_idx % 10 == 0 and combinator_idx != 0:
                    print(
                        f"Adding conditions: {section_idx:02d}/{self.section_count - 1:02d} -> {combinator_idx:02d}/{len(section.deciders) - 1:02d}",
                        end="\r",
                    )
                combinator.add_condition(self.timer_signal, None, time_ptr, ">=")
                time_ptr += self.delta_timetick
                combinator.add_condition(self.timer_signal, None, time_ptr, "<", True)

        # 向量化RGB转换函数
        def vectorized_rgb_conversion(image):
            # 注意：OpenCV图像是BGR格式
            r = image[:, :, 2].astype(np.uint32) << 16
            g = image[:, :, 1].astype(np.uint32) << 8
            b = image[:, :, 0].astype(np.uint32)
            return r | g | b

        # 预计算每个section的有效区域
        section_regions = []
        for section_idx in range(self.section_count):
            start_x = section_idx * self.section_width - self.padding
            end_x = start_x + self.section_width
            valid_start = max(start_x, 0)
            valid_end = min(end_x, self.width)
            section_regions.append((start_x, end_x, valid_start, valid_end))

        # 批量处理所有帧
        for frame_idx, image in enumerate(self.images):
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx+1}/{len(self.images)}")

            # 向量化转换整幅图像为RGB值
            rgb_image = vectorized_rgb_conversion(image)

            # 处理每个section
            for section_idx, section in enumerate(self.display_sections):
                start_x, end_x, valid_start, valid_end = section_regions[section_idx]
                combinator = section.deciders[frame_idx]

                # 创建该section的输出数组（初始为0 - 黑色）
                outputs = np.zeros(self.section_size, dtype=np.uint32)

                # 计算有效区域的大小
                valid_width = valid_end - valid_start
                if valid_width > 0:
                    # 计算在section内的起始位置
                    section_start = valid_start - start_x

                    # 提取图像的有效区域
                    image_region = rgb_image[
                        :, valid_start:valid_end
                    ]  # 注意: opencv的格式为 (height, width, channels), 即(y, x, bgr)

                    # 将有效区域复制到输出数组中
                    for col in range(valid_width):
                        start_idx = (section_start + col) * self.height
                        end_idx = start_idx + self.height
                        outputs[start_idx:end_idx] = image_region[:, col]

                # # 批量创建输出信号
                # if section_idx == 0:  # 第一列
                #     signals = self.all_signals[- valid_width * self.height :]
                # elif section_idx == self.section_count - 1:  # 最后一列
                #     signals = self.all_signals[: valid_width * self.height ]
                # else:  # 中间列
                #     signals = self.all_signals
                l_edge = section_start * self.height
                r_edge = (section_start + valid_width) * self.height
                combinator.add_outputs_with_value(self.all_signals[l_edge:r_edge], outputs[l_edge:r_edge].tolist())
    def set_unused_lamps_white(self):
        for index, lamp in enumerate(self.display_sections[0].lamps):
            if index >= self.padding * self.height:
                break
            lamp.controlled = False
        for index, lamp in enumerate(self.display_sections[-1].lamps):
            rpadding = self.section_count * self.section_width - self.width - self.padding
            if index < self.section_size - rpadding * self.height:
                continue
            lamp.controlled = False
            
    def set_up_electricity(self):
        if self.electric_set == "nothing":
            return
        elif self.electric_set in ELECTRIC_SET[1:5]:
            start_x, start_y = -1, -1
            end_x = self.width + 1
            end_y = self.display_sections[0].deciders[-1].y + 2
            self.blueprint["blueprint"]["tiles"] = [
                {
                    "name": self.electric_set,
                    "position": {"x": x, "y": y}
                }
                for x in range(start_x, end_x)
                for y in range(start_y, end_y)
            ]
        elif self.electric_set in ELECTRIC_SET[5:]:
            raise NotImplementedError("Substation not implemented")
        else:
            raise ValueError("Invalid electric set")

    def generate_blueprint(self):
        self.set_up_timer()
        print("Timer set up")
        self.cut_into_sections()
        print("Sections cut")
        self.set_combinator_signals()
        print("Signals set")
        self.set_unused_lamps_white()
        print("Lamps set")
        self.set_up_electricity()
        print("Electricity set up")

        self.blueprint["blueprint"]["entities"] = [entity.to_dict() for entity in self.timer]
        for section in self.display_sections:
            self.blueprint["blueprint"]["entities"].extend(section.to_list())
        self.blueprint["blueprint"]["wires"] = self.wires
        print("Blueprint generated")

    def save_blueprint(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.blueprint, f, indent=4)

    def save_blueprint_code(self, file_path: str):
        print("Ecoding blueprint")
        bp_code = encode_blueprint(self.blueprint)

        with open(file_path, "w") as f:
            f.write(bp_code)


# 处理视频入口
def handle_video(
    video_path: str,
    target_path: str,
    skip_fps: int,
    target_size: tuple[int, int],
    title: str,
    description: str,
    electric_set: str,
    store_origin: bool,
):
    images, fps = video_to_images(video_path, target_size)
    images = images[::skip_fps]
    fps = fps // skip_fps
    signal_mask = [f"parameter-{idx}" for idx in range(10)]
    frames_to_blueprint = Frames2Blueprint(images, fps, target_size[0], target_size[1], electric_set, title, description, signal_mask)
    frames_to_blueprint.init_signals({"type": "virtual", "name": "signal-0", "quality": "normal"})
    frames_to_blueprint.generate_blueprint()
    if store_origin:
        frames_to_blueprint.save_blueprint(target_path + "_origin.json")
    frames_to_blueprint.save_blueprint_code(target_path)


# 处理图片入口
def handle_image(
    image_path: str, target_path: str, target_size: tuple[int, int], title: str, description: str, electric_set: str
):
    image = downsample_for_pixel_art(image_path, target_size)
    blueprint = build_blueprint_from_image(image, title, description, electric_set)
    with open(target_path, "w") as f:
        f.write(blueprint)


# 分流入口
def main():
    parser = argparse.ArgumentParser(description="Convert video/image to blueprint.")
    parser.add_argument("input", type=str, help="Input video or image file.")
    parser.add_argument("--output", type=str, help="Output blueprint file.")
    parser.add_argument("--title", type=str, default="Video", help="Title of the blueprint.")
    parser.add_argument("--description", type=str, default="", help="Description of the blueprint.")
    parser.add_argument(
        "--electric-set",
        type=str,
        choices=ELECTRIC_SET,
        default="nothing",
        help=f"Electric set of the blueprint. ",
    )
    parser.add_argument("--size", type=int, nargs=2, default=(128, 128), help="Target size of the blueprint.")
    parser.add_argument("--skip-fps", type=int, default=1, help="Skip x frames before loading one.")
    parser.add_argument("--get-info", action="store_true", help="Only get video/image info.")
    parser.add_argument("--store-origin", action="store_true", help="Store original blueprint.")
    parser.add_argument("--resources-dir", type=str, default=".", help="Directory to store resources.")
    args = parser.parse_args()

    input_path: str = args.input
    output_path: str = args.output
    title: str = args.title
    description: str = args.description
    electric_set: str = args.electric_set
    target_size: tuple[int, int] = args.size
    skip_fps: int = args.skip_fps
    get_info: bool = args.get_info
    store_origin: bool = args.store_origin
    global resources_directory 
    resources_directory = args.resources_dir

    # cv2支持的视频后缀
    video_suffixes = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"] + [".gif"]
    # pillow支持的照片后缀
    image_suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]

    if get_info:
        if input_path.lower().endswith(tuple(video_suffixes)):
            video_info = get_video_info(input_path) # 返回[fps, width, height, frame_count]
            print(json.dumps({
                "type": "video",
                "fps": video_info[0],
                "width": video_info[1],
                "height": video_info[2],
                "frame_count": video_info[3]
            }))
        elif input_path.lower().endswith(tuple(image_suffixes)):
            image_info = get_image_info(input_path) # 返回[width, height]
            print(json.dumps({
                "type": "image",
                "width": image_info[0],
                "height": image_info[1]
            }))
        else:
            raise ValueError(f"Unsupported file format: {input_path.split('.')[-1]}")    
    else:
        if input_path.lower().endswith(tuple(video_suffixes)):
            handle_video(input_path, output_path, skip_fps, target_size, title, description, electric_set, store_origin)
        elif input_path.lower().endswith(tuple(image_suffixes)):
            handle_image(input_path, output_path, target_size, title, description, electric_set)
        else:
            raise ValueError(f"Unsupported file format: {input_path.split('.')[-1]}")  


if __name__ == "__main__":
    main()
