
<div align=center>
  <img src="assets/logo.png" width=600 >
</div>

<h1 align="center">Detect Anything via Next Point Prediction</h1>

<div align=center>

<p align="center">
  <a href="https://rex-omni.github.io/">
    <img
      src="https://img.shields.io/badge/RexOmni-Website-BADFDB?style=flat-square&logo=deno&logoColor=violet&color=BADFDB"
      alt="RexThinker Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2510.12798">
    <img
      src="https://img.shields.io/badge/RexOmni-Paper-Red%25red?logo=arxiv&logoColor=red&color=yellow"
      alt="RexThinker Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/IDEA-Research/Rex-Omni-AWQ">
    <img 
        src="https://img.shields.io/badge/RexOmni_AWQ-Weight-orange?logo=huggingface&logoColor=yellow" 
        alt="RexThinker weight on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/IDEA-Research/Rex-Omni">
    <img 
        src="https://img.shields.io/badge/RexOmni-Weight-orange?logo=huggingface&logoColor=yellow" 
        alt="RexThinker weight on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/spaces/Mountchicken/Rex-Omni">
    <img
      src="https://img.shields.io/badge/RexOmni-Demo-orange?logo=huggingface&logoColor=yellow" 
      alt="RexThinker Demo on Hugging Face"
    />
  </a>
  
</p>

</div>

> Rex-Omni is a 3B-parameter Multimodal Large Language Model (MLLM) that redefines object detection and a wide range of other visual perception tasks as a simple next-token prediction problem.

<p align="center"><img src="assets/teaser.png" width="95%"></p>


# News 🎉
- [2025-10-31] We release the AWQ quantized version of Rex-Omni, which saves 50% of the storage space. [Rex-Omni-AWQ](https://huggingface.co/IDEA-Research/Rex-Omni-AWQ)
- [2025-10-29] Fine-tuning code is now [available](finetuning/README.md).
- [2025-10-17] Evaluation code and dataset is now [available](evaluation/README.md).
- [2025-10-15] Rex-Omni is released.

# Table of Contents

- [News 🎉](#news-)
- [Table of Contents](#table-of-contents)
  - [TODO LIST 📝](#todo-list-)
  - [1. Installation ⛳️](#1-installation-️)
  - [2. Quick Start: Using Rex-Omni for Detection](#2-quick-start-using-rex-omni-for-detection)
      - [Initialization parameters (RexOmniWrapper)](#initialization-parameters-rexomniwrapper)
      - [Inference parameters (rex.inference)](#inference-parameters-rexinference)
  - [3. Cookbooks](#3-cookbooks)
  - [4. Applications of Rex-Omni](#4-applications-of-rex-omni)
  - [5. Gradio Demo](#5-gradio-demo)
    - [Quick Start](#quick-start)
    - [Available Options](#available-options)
  - [6. Evaluation](#6-evaluation)
  - [7. Fine-tuning Rex-Omni](#7-fine-tuning-rex-omni)
  - [8. LICENSE](#8-license)
  - [9. Citation](#9-citation)


## TODO LIST 📝
- [x] Add Evaluation Code
- [x] Add Fine-tuning Code
- [x] Add Quantilized Rex-Omni

## 1. Installation ⛳️

```bash
git clone https://github.com/IDEA-Research/Rex-Omni.git
cd Rex-Omni
conda create -n rexomni python=3.10 -y
conda activate rexomni
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Test Installation
```bash
CUDA_VISIBLE_DEVICES=1 python tutorials/detection_example/detection_example.py
```

If the installation is successful, you will find a visualization of the detection results at `tutorials/detection_example/test_images/cafe_visualize.jpg`

## 2. Quick Start: Using Rex-Omni for Detection
Below is a minimal example showing how to run object detection using the `rex_omni` package.

```python
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize

# 1) Initialize the wrapper (model loads internally)
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",   # HF repo or local path
    backend="transformers",                # or "vllm" for high-throughput inference
    # Inference/generation controls (applied across backends)
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

# If you are using the AWQ quantized version of Rex-Omni, you can use the following code:
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni-AWQ",
    backend="vllm",
    quantization="awq",
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

# 2) Prepare input
image = Image.open("tutorials/detection_example/test_images/cafe.jpg").convert("RGB")
categories = [
    "man", "woman", "yellow flower", "sofa", "robot-shope light",
    "blanket", "microwave", "laptop", "cup", "white chair", "lamp",
]

# 3) Run detection
results = rex.inference(images=image, task="detection", categories=categories)
result = results[0]

# 4) Visualize
vis = RexOmniVisualize(
    image=image,
    predictions=result["extracted_predictions"],
    font_size=20,
    draw_width=5,
    show_labels=True,
)
vis.save("tutorials/detection_example/test_images/cafe_visualize.jpg")
```

#### Initialization parameters (RexOmniWrapper)
- **model_path**: Hugging Face repo ID or a local checkpoint directory for the Rexe-Omni model.
- **backend**: "transformers" or "vllm".
  - **transformers**: easy to use, good baseline latency.
  - **vllm**: high-throughput, low-latency inference. Requires the `vllm` package and a compatible environment.
- **max_tokens**: Maximum number of tokens to generate for each output.
- **temperature**: Sampling temperature; higher values increase randomness (0.0 = deterministic/greedy).
- **top_p**: Nucleus sampling parameter; model samples from the smallest set of tokens with cumulative probability ≥ top_p.
- **top_k**: Top-k sampling; restricts sampling to the k most likely tokens.
- **repetition_penalty**: Penalizes repeated tokens; >1.0 discourages repetition.
- Optional advanced settings (supported via kwargs when constructing the wrapper):
  - Transformers: `torch_dtype`, `attn_implementation`, `device_map`, `trust_remote_code`, etc.
  - VLLM: `tokenizer_mode`, `limit_mm_per_prompt`, `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`, `trust_remote_code`, etc.

#### Inference parameters (rex.inference)
- **images**: A single `PIL.Image.Image` or a list of images for batch inference.
- **task**: One of `"detection"`, `"pointing"`, `"visual_prompting"`, `"keypoint"`, `"ocr_box"`, `"ocr_polygon"`, `"gui_grounding"`, `"gui_pointing"`.
- **categories**: List of category names/phrases to detect or extract, e.g., `["person", "cup"]`. Used to build task prompts.
- **keypoint_type": Type of keypoints for keypoint detection task. Options: "person", "hand", "animal"
- **visual_prompt_boxes**: Reference bounding boxes for visual prompting task. Format: [[x0, y0, x1, y1], ...] in absolute coordinates

Returns a list of dictionaries (one per input image). Each dictionary includes:
- **raw_output**: The raw text generated by the LLM.
- **extracted_predictions**: Structured predictions parsed from the raw output, grouped by category.
  - For detection: `{category: [{"type": "box", "coords": [x0,y0,x1,y1]}, ...], ...}`
  - For pointing:  `{category: [{"type": "point", "coords": [x0,y0]}, ...], ...}`
  - For polygon: `{category: [{"type": "polygon", "coords": [x0,y0, ...]}, ...], ...}`
  - For keypointing: Structured Json

Tips:
- For best performance with VLLM, set `backend="vllm"` and tune `gpu_memory_utilization` and `tensor_parallel_size` according to your GPUs.

## 3. Cookbooks

We provide comprehensive tutorials for each supported task. Each tutorial includes both standalone Python scripts and interactive Jupyter notebooks.

|       Task       |                                                                Applications                                                               |   Demo |                  Python Example                    |                     Notebook                     |
|:----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------:|:------------------------------------------------:|:------------------------------------------------:|
| Detection |                               `object detection`                                 | <img src="assets/cafe_visualize.jpg" width="240"/>  | [code](tutorials/detection_example/detection_example.py)   | [notebook](tutorials/detection_example/_full_notebook.ipynb) |
|                  |                         `object referring`                      | <img src="assets/boys_visualize.jpg" width="240"/>   | [code](tutorials/detection_example/referring_example.py)      |       [notebook](tutorials/detection_example/_full_notebook.ipynb)                                           |
|                  | `gui grounding` | <img src="assets/gui_visualize.jpg" width="240"/> | [code](tutorials/detection_example/gui_grounding_example.py)  |       [notebook](tutorials/detection_example/_full_notebook.ipynb)                                           |
|                  |                    `layout grounding`    |  <img src="assets/layout_visualize.jpg" width="240"/>        | [code](tutorials/detection_example/layout_grouding_examle.py) |       [notebook](tutorials/detection_example/_full_notebook.ipynb)                                             |
| Pointing |                               `object pointing`            |       <img src="assets/object_pointing_visualize.jpg" width="240"/>           |   [code](tutorials/pointing_example/object_pointing_example.py)   | [notebook](tutorials/pointing_example/_full_notebook.ipynb) |
|                  |                         `gui pointing`    |      <img src="assets/gui_pointing_visualize.jpg" width="240"/>              | [code](tutorials/pointing_example/gui_pointing_example.py)      |       [notebook](tutorials/pointing_example/_full_notebook.ipynb)                                           |
|                  | `affordance pointing` | <img src="assets/affordance_pointing_visualize.jpg" width="240"/> | [code](tutorials/pointing_example/affordance_pointing_example.py)  |       [notebook](tutorials/pointing_example/_full_notebook.ipynb)                                           |
| Visual prompting | `visual prompting` | <img src="assets/pigeons_visualize.jpg" width="240"/> | [code](tutorials/visual_prompting_example/visual_prompt_example.py) | [notebook](tutorials/visual_prompting_example/_full_tutorial.ipynb) |
| OCR | `ocr word box` | <img src="assets/ocr_word_box_visualize.jpg" width="240"/> | [code](tutorials/ocr_example/ocr_word_box_example.py) | [notebook](tutorials/ocr_example/_full_tutorial.ipynb) |
|                  | `ocr textline box` | <img src="assets/ocr_textline_box_visualize.jpg" width="240"/> | [code](tutorials/ocr_example/ocr_textline_box_example.py) |       [notebook](tutorials/ocr_example/_full_tutorial.ipynb)                                           |
|                  | `ocr polygon` | <img src="assets/ocr_polygon_visualize.jpg" width="240"/> | [code](tutorials/ocr_example/ocr_polygon_example.py) |       [notebook](tutorials/ocr_example/_full_tutorial.ipynb)                                           |
| Keypointing | `person keypointing` | <img src="assets/person_keypointing_visualize.jpg" width="240"/> | [code](tutorials/keypointing_example/person_keypointing_example.py) | [notebook](tutorials/keypointing_example/_full_tutorial.ipynb)|
|             | `animal keypointing`   |     <img src="assets/animal_keypointing_visualize.jpg" width="240"/>                     |  [code](tutorials/keypointing_example/animal_keypointing_example.py)                                                |       [notebook](tutorials/keypointing_example/_full_tutorial.ipynb)                                           |
| Other | `batch inference` |  | [code](tutorials/other_example/batch_inference.py) ||

## 4. Applications of Rex-Omni

Rex-Omni's unified detection framework enables seamless integration with other vision models.

| Application | Description | Demo | Documentation |
|:------------|:------------|:----:|:-------------:|
| **Rex-Omni + SAM** | Combine language-driven detection with pixel-perfect segmentation. Rex-Omni detects objects → SAM generates precise masks | <img src="assets/rexomni_sam.jpg" width="500"/> | [README](applications/_1_rexomni_sam/README.md) |
| **Grounding Data Engine** | Automatically generate phrase grounding annotations from image captions using spaCy and Rex-Omni. | <img src="assets/cafe_grounding.jpg" width="500"/> | [README](applications/_2_automatic_grounding_data_engine/README.md) |


## 5. Gradio Demo

![img](assets/gradio.png)

We provide an interactive Gradio demo that allows you to test all Rex-Omni capabilities through a web interface.

### Quick Start
```bash
# Launch the demo
CUDA_VISIBLE_DEVICES=0 python app.py --model_path IDEA-Research/Rex-Omni

# With custom settings
CUDA_VISIBLE_DEVICES=0 python app.py \
    --model_path IDEA-Research/Rex-Omni \
    --backend vllm \
    --server_name 0.0.0.0 \
    --server_port 7890
```

### Available Options
- `--model_path`: Model path or HuggingFace repo ID (default: "IDEA-Research/Rex-Omni")
- `--backend`: Backend to use - "transformers" or "vllm" (default: "transformers")
- `--server_name`: Server host address (default: "192.168.81.138")
- `--server_port`: Server port (default: 5211)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top_p`: Nucleus sampling parameter (default: 0.05)
- `--max_tokens`: Maximum tokens to generate (default: 2048)

## 6. Evaluation
Please refer to [Evaluation](evaluation/README.md) for more details.

## 7. Fine-tuning Rex-Omni
Please refer to [Fine-tuning Rex-Omni](finetuning/README.md) for more details.

## 8. LICENSE

Rex-Omni is licensed under the [IDEA License 1.0](LICENSE), Copyright (c) IDEA. All Rights Reserved. This model is based on Qwen, which is licensed under the [Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE), Copyright (c) Alibaba Cloud. All Rights Reserved.


## 9. Citation
Rex-Omni comes from a series of prior works. If you’re interested, you can take a look.

- [RexThinker](https://arxiv.org/abs/2506.04034)
- [RexSeek](https://arxiv.org/abs/2503.08507)
- [ChatRex](https://arxiv.org/abs/2411.18363)
- [DINO-X](https://arxiv.org/abs/2411.14347)
- [Grounidng DINO 1.5](https://arxiv.org/abs/2405.10300)
- [T-Rex2](https://link.springer.com/chapter/10.1007/978-3-031-73414-4_3)
- [T-Rex](https://arxiv.org/abs/2311.13596)


```text
@misc{jiang2025detectpointprediction,
      title={Detect Anything via Next Point Prediction}, 
      author={Qing Jiang and Junan Huo and Xingyu Chen and Yuda Xiong and Zhaoyang Zeng and Yihao Chen and Tianhe Ren and Junzhi Yu and Lei Zhang},
      year={2025},
      eprint={2510.12798},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12798}, 
}


```