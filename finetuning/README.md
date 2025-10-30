# Rex-Omni Finetuning Code

We provide code for both SFT and GRPO finetuning of the Rex-Omni model.

## Table of Contents

- [Rex-Omni Finetuning Code](#rex-omni-finetuning-code)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Stage 1: SFT Finetuning](#stage-1-sft-finetuning)
    - [1.1. Data Format](#11-data-format)
    - [1.2. Download Toy Data](#12-download-toy-data)
      - [1.2.1 Grounding Data Format Example](#121-grounding-data-format-example)
      - [1.2.2 Visualize the Toy Dataset](#122-visualize-the-toy-dataset)
      - [1.2.3 Convert Custom Data to TSV Format](#123-convert-custom-data-to-tsv-format)
    - [1.3. Launch Training](#13-launch-training)
  - [Stage 2: GRPO Finetuning](#stage-2-grpo-finetuning)
    - [2.1 Download Toy Data](#21-download-toy-data)
    - [2.2 Launch Training](#22-launch-training)
    - [2.3 Convert Checkpoint to Huggingface Version](#23-convert-checkpoint-to-huggingface-version)
    - [2.4 Reward Function](#24-reward-function)

---

## Installation

```bash
# Install Rex-Omni, skip if you have already installed it
conda create -n rexomni -m python=3.10
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/IDEA-Research/Rex-Omni.git
cd Rex-Omni
pip install -v -e .

# Install required dependencies
cd finetuning
pip install -v -e .

```

---

## Stage 1: SFT Finetuning

Stage 1 SFT finetuning uses supervised learning to finetune the model with annotated data.

### 1.1. Data Format

This project uses **TSV (Tab-Separated Values) format** to store datasets. Each dataset requires three files:

1. **Image TSV file** (`*.images.tsv`): Stores base64-encoded images
2. **Annotation TSV file** (`*.annotations.tsv`): Stores annotation data
3. **Line index file** (`*.annotations.tsv.lineidx`): Stores byte offsets for each annotation line in the file

Here is an example code to parse the dataset:

```python
import json
import os
from base64 import b64decode
from io import BytesIO

import numpy as np
from torch.utils.data import Dataset

class TSVBase(Dataset):
    """Base class for TSV dataset. This class is used to load image and annotations from TSV file.

    Args:
        img_tsv_file (str): The path to the image TSV file.
        ann_tsv_file (str): The path to the annotation TSV file.
        ann_lineidx_file (str): The path to the annotation lineidx file.
        num_workers (int): The number of workers.
        data_ratio (float, optional): The ratio of data to use. Defaults to 1.0.
        filter_empty (bool): If filter the samples without annotations. When training, set it to True.
        dataset_type (str): The data source.
    """

    def __init__(
        self,
        img_tsv_file: str,
        ann_tsv_file: str,
        ann_lineidx_file: str,
    ):

        self.data = []
        f = open(ann_lineidx_file)
        for line in tqdm(f):
            self.data.append(int(line.strip()))

        self.img_handle = None
        self.ann_handle = None
        self.img_tsv_file = img_tsv_file
        self.ann_tsv_file = ann_tsv_file

        self.preparer = None
        self.captionbuilder = None
        self._transforms = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

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
        img = Image.open(img).convert("RGB")
        target = json.loads(ann)
        return img, target
```

### 1.2. Download Toy Data

We have released a toy grounding dataset with 1000 samples for finetuning example. You can download it at [Huggingface](https://huggingface.co/datasets/Mountchicken/Rex-Omni-Finetune-ToyData)

#### 1.2.1 Grounding Data Format Example

The Grounding task is used to train the model for region localization (phrase grounding). Below is the format specification for annotation data:

**Annotation TSV file format** (`*.annotations.tsv`):
- Line format: `{image_line_idx}\t{annotation_json}`
- `image_line_idx`: Line index of the corresponding image in the images.tsv file
- `annotation_json`: JSON-formatted annotation data

**Annotation JSON data structure**:
```json
{
  "boxes": [
    {
      "bbox": [x0, y0, x1, y1],  // Bounding box coordinates in xyxy format
      "phrase": "object description",  // the category of description for this box
    },
    ...
  ]
}
```

#### 1.2.2 Visualize the Toy Dataset

```python
python tools/vis_tsv_dataset.py \
  --img_tsv_file Mountchicken/Rex-Omni-Finetune-ToyData/SFT_Grounding_data.images.tsv \
  --ann_tsv_file Mountchicken/Rex-Omni-Finetune-ToyData/SFT_Grounding_data.annotations.tsv \
  --ann_lineidx_file Mountchicken/Rex-Omni-Finetune-ToyData/SFT_Grounding_data.annotations.tsv.lineidx \
  --output_dir Mountchicken/Rex-Omni-Finetune-ToyData/vis \
  --num_samples 20 \
```

#### 1.2.3 Convert Custom Data to TSV Format

We also provide a script to convert custom data in JSON format to TSV format.

**Usage:**

```bash
python tools/convert_json_data_to_tsv.py \
  --json_file /path/to/annotations.json \
  --image_root_path /path/to/images \
  --save_image_tsv_path output/images.tsv \
  --save_ann_tsv_path output/annotations.tsv \
  --save_ann_lineidx_path output/annotations.tsv.lineidx
```

**JSON Format:**

The input JSON file should contain one JSON object per line. Each line must have two keys: `image_name` and `annotation`.

- `image_name`: The relative path or filename of the image (will be joined with `image_root_path`)
- `annotation`: The annotation data (can be any valid JSON object, will be stored as-is in the TSV file)

**Example JSON file (`annotations.json`):**

```json
{"image_name": "image1.jpg", "annotation": {"boxes": [{"bbox": [10, 20, 100, 200], "phrase": "person"}]}}
{"image_name": "subdir/image2.jpg", "annotation": {"boxes": [{"bbox": [50, 50, 150, 250], "phrase": "car"}]}}
{"image_name": "image3.png", "annotation": {"boxes": [{"bbox": [0, 0, 200, 300], "phrase": "dog"}]}}
```

**Parameters:**

- `--json_file`: Path to the input JSON file (one JSON object per line)
- `--image_root_path`: Root directory where images are stored
- `--save_image_tsv_path`: Output path for the image TSV file
- `--save_ann_tsv_path`: Output path for the annotation TSV file
- `--save_ann_lineidx_path`: Output path for the annotation lineidx file

The script will:
1. Read JSON file line by line
2. Load images from `image_root_path/image_name`
3. Convert images to base64-encoded JPEG format
4. Generate TSV files with proper line indices

### 1.3. Launch Training

Use the provided script to launch training:

```bash
bash scripts/sft.sh
```

**Main parameter descriptions**:
- `--config`: Specify the configuration file path (contains dataset configuration)
- `--deepspeed`: DeepSpeed configuration file for memory optimization
- `--output_dir`: Model output directory
- `--per_device_train_batch_size`: Training batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Main learning rate
- `--mm_projector_lr`: Multimodal projector learning rate
- `--vision_tower_lr`: Vision encoder learning rate

---

## Stage 2: GRPO Finetuning

Stage 2 GRPO finetuning uses reinforcement learning to further optimize model performance through reward functions. We mainly adopt the code from [Easy-R1](https://github.com/hiyouga/EasyR1).

### 2.1 Download Toy Data

We use the same toy data as the SFT stage.

### 2.2 Launch Training

```bash
bash scripts/grpo.sh
```

**Main parameter descriptions**:
- `--data.config_path`: Specify the configuration file path (contains dataset configuration)

### 2.3 Convert Checkpoint to Huggingface Version
python tools/merge_rl_checkpoints_to_hg_version.py --local_dir PATH_TO_CHECKPOINT_AT_ACTOR_DIR

--local_dir: The path to the checkpoint at the actor directory.

### 2.4 Reward Function

We implement the following reward functions in `verl/configs/reward_func.py`:
- Box IoU
- Point in Box
- Point in Mask
