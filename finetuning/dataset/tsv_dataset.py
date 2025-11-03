import copy
from base64 import b64decode
from collections.abc import Sequence
from io import BytesIO
from typing import Dict, List

import numpy as np
import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset

from engine.registry import BUILDER
from utils.box_utils import xywh2xwxy
from utils.constants import *
from utils.rope2d import get_rope_index_2, get_rope_index_25


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
    system_message="You are a helpful assistant.",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class GroundingTSVDataset(Dataset):
    """Detection TSV Dataset for training Det2.0

    Args:

    """

    def __init__(
        self,
        img_tsv_file: str,
        ann_tsv_file: str,
        ann_lineidx_file: str,
        tokenizer,
        data_args,
        image_min_pixels,
        image_max_pixels,
        max_num_samples=None,
        task_fn=None,
        system_message="You are a helpful assistant.",
        ratio_range=[0.0, 1.0],
        ori_box_format="xyxy",
        dataset_name=None,
        if_convert_raw_data=True,
        max_length=4096,
    ):
        super(GroundingTSVDataset, self).__init__()
        self.data = []
        f = open(ann_lineidx_file)
        for line in f:
            self.data.append(int(line.strip()))
        self.data = self.data[
            int(len(self.data) * ratio_range[0]) : int(len(self.data) * ratio_range[1])
        ]
        if max_num_samples is not None:
            # shuffle the data and select the first max_num_samples
            np.random.shuffle(self.data)
            self.data = self.data[:max_num_samples]

        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        self.id2name = None

        self.img_handle = None
        self.ann_handle = None
        self.img_tsv_file = img_tsv_file
        self.ann_tsv_file = ann_tsv_file

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = image_max_pixels
        self.data_args.image_processor.min_pixels = image_min_pixels
        self.data_args.image_processor.size["longest_edge"] = image_max_pixels
        self.data_args.image_processor.size["shortest_edge"] = image_min_pixels

        self.system_message = system_message
        self.ori_box_format = ori_box_format
        self.dataset_name = dataset_name
        self.if_convert_raw_data = if_convert_raw_data
        self.max_length = max_length
        if task_fn is not None:
            self.task_fn = BUILDER.build(task_fn)
        else:
            self.task_fn = None

    def __len__(self):
        return len(self.data)

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
        elif isinstance(image_file, Image.Image):
            image = image_file
        else:
            raise ValueError("image_file should be a string or PIL Image.")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def load_image_and_anno(self, idx):
        ann_line_idx = self.data[idx]

        if self.ann_handle is None:
            self.ann_handle = open(self.ann_tsv_file)
        self.ann_handle.seek(ann_line_idx)
        img_line_idx, ann = self.ann_handle.readline().strip().split("\t")

        img_line_idx = int(img_line_idx)
        if self.img_handle is None:
            self.img_handle = open(self.img_tsv_file)
        self.img_handle.seek(img_line_idx)
        img = self.img_handle.readline().strip().split("\t")[1]
        if img.startswith("b'"):
            img = img[1:-1]
        img = BytesIO(b64decode(img))
        image = Image.open(img).convert("RGB")
        data_dict = json.loads(ann)
        return image, data_dict

    def convert_raw_data(self, image_pil, data_dict):
        """Convert raw data from the tsv format to a unified format."""
        boxes = [anno["bbox"] for anno in data_dict["boxes"]]
        labels = [
            anno.get("phrase", anno.get("caption", None)) for anno in data_dict["boxes"]
        ]
        # convert box to xyxy
        if self.ori_box_format == "xywh":
            if len(boxes) > 0:
                boxes = xywh2xwxy(boxes)
        data_dict = dict(
            boxes=np.array(boxes),
            labels=labels,
            size=(image_pil.height, image_pil.width),
        )
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            image_pil, data_dict = self.load_image_and_anno(i)
        except Exception as e:
            print(f"error image {self.dataset_name} id: {i}. {e}")
            return self.__getitem__((i + 1) % len(self.data))
        w, h = image_pil.size
        if w < 28 or h < 28:
            print(f"image size is too small {w}x{h}, skip this image.")
            return self.__getitem__((i + 1) % len(self.data))
        # convert data format
        data_dict = self.convert_raw_data(image_pil, data_dict)
        if data_dict is None:
            return self.__getitem__((i + 1) % len(self.data))

        if "boxes" in data_dict and len(data_dict["boxes"]) == 0:
            print(f"no boxes in {self.dataset_name} id: {i}.")
            return self.__getitem__((i + 1) % len(self.data))

        if image_pil is not None:  # image only data
            ori_width, ori_height = image_pil.size
            image, grid_thw = self.process_image_unified(image_pil)
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]

            # convert sources based on the task_fn
            if self.task_fn is not None:
                # pass image here for the need of converting coordinates
                data_dict = self.task_fn(
                    {"id2name": self.id2name, "annotations": data_dict},
                    ori_width,
                    ori_height,
                )

            sources = copy.deepcopy([e["conversations"] for e in [data_dict]])
            data_dict = preprocess_qwen_2_visual(
                sources,
                self.tokenizer,
                grid_thw=grid_thw_merged,
                visual_type="image",
                system_message=self.system_message,
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )

            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )

            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources,
                self.tokenizer,
                grid_thw=grid_thw_merged,
                system_message=self.system_message,
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        if data_dict["input_ids"].size(0) > self.max_length:
            print(
                f"input_ids is too long {data_dict['input_ids'].size(0)}, dataset name {self.dataset_name} skip this image."
            )
            return self.__getitem__((i + 1) % len(self.data))

        return data_dict
