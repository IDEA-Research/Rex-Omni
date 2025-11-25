#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import re
from typing import Any, Dict, List

import gradio as gr
import numpy as np
import torch
from gradio_image_prompter import ImagePrompter
from PIL import Image

from rex_omni import RexOmniVisualize, RexOmniWrapper, TaskType
from rex_omni.tasks import KEYPOINT_CONFIGS, TASK_CONFIGS, get_task_config


def parse_args():
    parser = argparse.ArgumentParser(description="Rex Omni Gradio Demo")
    parser.add_argument(
        "--model_path",
        default="IDEA-Research/Rex-Omni",
        help="Model path or HuggingFace repo ID",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend to use for inference",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=6121)
    args = parser.parse_args()
    return args


# Task configurations with detailed descriptions
DEMO_TASK_CONFIGS = {
    "Detection": {
        "task_type": TaskType.DETECTION,
        "description": "Detect objects and return bounding boxes",
        "example_categories": "person",
        "supports_visual_prompt": False,
        "supports_ocr_config": False,
    },
    "Pointing": {
        "task_type": TaskType.POINTING,
        "description": "Point to objects and return point coordinates",
        "example_categories": "person",
        "supports_visual_prompt": False,
        "supports_ocr_config": False,
    },
    "Visual Prompting": {
        "task_type": TaskType.VISUAL_PROMPTING,
        "description": "Ground visual examples to find similar objects",
        "example_categories": "",
        "supports_visual_prompt": True,
        "supports_ocr_config": False,
    },
    "Keypoint": {
        "task_type": TaskType.KEYPOINT,
        "description": "Detect keypoints with skeleton visualization",
        "example_categories": "person, hand, animal",
        "supports_visual_prompt": False,
        "supports_ocr_config": False,
    },
    "OCR": {
        "task_type": None,  # Will be determined by OCR config
        "description": "Optical Character Recognition with customizable output format",
        "example_categories": "text, word",
        "supports_visual_prompt": False,
        "supports_ocr_config": True,
    },
}

# OCR configuration options
OCR_OUTPUT_FORMATS = {
    "Box": {
        "task_type": TaskType.OCR_BOX,
        "description": "Detect text with bounding boxes",
    },
    "Polygon": {
        "task_type": TaskType.OCR_POLYGON,
        "description": "Detect text with polygon boundaries",
    },
}

OCR_GRANULARITY_LEVELS = {
    "Word Level": {"categories": "word", "description": "Detect individual words"},
    "Text Line Level": {"categories": "text line", "description": "Detect text lines"},
}

# Example configurations
EXAMPLE_CONFIGS = [
    {
        "name": "Detection: Cafe Scene",
        "image_path": "tutorials/detection_example/test_images/cafe.jpg",
        "task": "Detection",
        "categories": "man, woman, yellow flower, sofa, robot-shape light, blanket, microwave, laptop, cup, white chair, lamp",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "Detection",
    },
    {
        "name": "Referring: Boys Playing",
        "image_path": "tutorials/detection_example/test_images/boys.jpg",
        "task": "Detection",
        "categories": "boys holding microphone, boy playing piano, the four guitars on the wall",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "Referring",
    },
    {
        "name": "GUI Grounding: Boys Playing",
        "image_path": "tutorials/detection_example/test_images/gui.png",
        "task": "Detection",
        "categories": "more information of song 'Photograph'",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "GUI Grounding",
    },
    {
        "name": "Object Pointing: Point to boxes",
        "image_path": "tutorials/pointing_example/test_images/boxes.jpg",
        "task": "Pointing",
        "categories": "open boxes, closed boxes",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "Point to boxes in the image",
    },
    {
        "name": "Affordance Pointing",
        "image_path": "tutorials/pointing_example/test_images/cup.png",
        "task": "Pointing",
        "categories": "where I can hold the green cup",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "Affordance Pointing",
    },
    {
        "name": "Keypoint: Person",
        "image_path": "tutorials/keypointing_example/test_images/person.png",
        "task": "Keypoint",
        "categories": "person",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "Detect human keypoints and pose estimation",
    },
    {
        "name": "Keypoint: Animal",
        "image_path": "tutorials/keypointing_example/test_images/animal.png",
        "task": "Keypoint",
        "categories": "animal",
        "keypoint_type": "animal",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "Detect animal keypoints and pose structure",
    },
    {
        "name": "OCR: Box and Word",
        "image_path": "tutorials/ocr_example/test_images/ocr.png",
        "task": "OCR",
        "categories": "text",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": None,
        "description": "OCR: Box and Word",
    },
    {
        "name": "OCR: Box and Text Line",
        "image_path": "tutorials/ocr_example/test_images/ocr.png",
        "task": "OCR",
        "categories": "text",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Text Line Level",
        "visual_prompt_boxes": None,
        "description": "OCR: Box and Text Line",
    },
    {
        "name": "OCR: Polygon and Text Line",
        "image_path": "tutorials/ocr_example/test_images/ocr.png",
        "task": "OCR",
        "categories": "text",
        "keypoint_type": "person",
        "ocr_output_format": "Polygon",
        "ocr_granularity": "Text Line Level",
        "visual_prompt_boxes": None,
        "description": "OCR: Polygon and Text Line",
    },
    {
        "name": "Visual Prompting: Pigeons",
        "image_path": "tutorials/visual_prompting_example/test_images/pigeons.jpeg",
        "task": "Visual Prompting",
        "categories": "pigeon",
        "keypoint_type": "person",
        "ocr_output_format": "Box",
        "ocr_granularity": "Word Level",
        "visual_prompt_boxes": [[644, 1210, 842, 1361], [1180, 1066, 1227, 1160]],
        "description": "Find similar pigeons using visual prompting examples",
    },
]


def parse_visual_prompt(points: List) -> List[List[float]]:
    """Parse visual prompt points to bounding boxes"""
    boxes = []
    for point in points:
        if point[2] == 2 and point[-1] == 3:  # Rectangle
            x1, y1, _, x2, y2, _ = point
            boxes.append([x1, y1, x2, y2])
        elif point[2] == 1 and point[-1] == 4:  # Positive point
            x, y, _, _, _, _ = point
            half_width = 10
            x1 = max(0, x - half_width)
            y1 = max(0, y - half_width)
            x2 = x + half_width
            y2 = y + half_width
            boxes.append([x1, y1, x2, y2])
    return boxes


def convert_boxes_to_visual_prompt_format(
    boxes: List[List[float]], image_width: int, image_height: int
) -> str:
    """Convert bounding boxes to visual prompt format for the model"""
    if not boxes:
        return ""

    # Convert to normalized bins (0-999)
    visual_prompts = []
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box

        # Normalize and convert to bins
        x0_norm = max(0.0, min(1.0, x0 / image_width))
        y0_norm = max(0.0, min(1.0, y0 / image_height))
        x1_norm = max(0.0, min(1.0, x1 / image_width))
        y1_norm = max(0.0, min(1.0, y1 / image_height))

        x0_bin = int(x0_norm * 999)
        y0_bin = int(y0_norm * 999)
        x1_bin = int(x1_norm * 999)
        y1_bin = int(y1_norm * 999)

        visual_prompt = f"<{x0_bin}><{y0_bin}><{x1_bin}><{y1_bin}>"
        visual_prompts.append(visual_prompt)

    return ", ".join(visual_prompts)


def get_task_prompt(
    task_name: str,
    categories: str,
    keypoint_type: str = "",
    visual_prompt_boxes: List = None,
    image_width: int = 0,
    image_height: int = 0,
    ocr_output_format: str = "Box",
    ocr_granularity: str = "Word Level",
) -> str:
    """Generate the actual prompt that will be sent to the model"""
    if task_name not in DEMO_TASK_CONFIGS:
        return "Invalid task selected."

    demo_config = DEMO_TASK_CONFIGS[task_name]

    if task_name == "Visual Prompting":
        task_config = get_task_config(TaskType.VISUAL_PROMPTING)
        if visual_prompt_boxes and len(visual_prompt_boxes) > 0:
            visual_prompt_str = convert_boxes_to_visual_prompt_format(
                visual_prompt_boxes, image_width, image_height
            )
            return task_config.prompt_template.replace(
                "{visual_prompt}", visual_prompt_str
            )
        else:
            return "Please draw bounding boxes on the image to provide visual examples."

    elif task_name == "Keypoint":
        task_config = get_task_config(TaskType.KEYPOINT)
        if keypoint_type and keypoint_type in KEYPOINT_CONFIGS:
            keypoints_list = KEYPOINT_CONFIGS[keypoint_type]
            keypoints_str = ", ".join(keypoints_list)
            prompt = task_config.prompt_template.replace("{categories}", keypoint_type)
            prompt = prompt.replace("{keypoints}", keypoints_str)
            return prompt
        else:
            return "Please select a keypoint type first."

    elif task_name == "OCR":
        # Get OCR task type based on output format
        ocr_task_type = OCR_OUTPUT_FORMATS[ocr_output_format]["task_type"]
        task_config = get_task_config(ocr_task_type)

        # Get categories based on granularity level
        ocr_categories = OCR_GRANULARITY_LEVELS[ocr_granularity]["categories"]

        # Replace categories in prompt template
        return task_config.prompt_template.replace("{categories}", ocr_categories)

    else:
        # For other tasks, use the task config from tasks.py
        task_type = demo_config["task_type"]
        task_config = get_task_config(task_type)

        # Replace {categories} placeholder
        if categories.strip():
            return task_config.prompt_template.replace(
                "{categories}", categories.strip()
            )
        else:
            return task_config.prompt_template.replace("{categories}", "objects")


def run_inference(
    image,
    task_selection,
    categories,
    keypoint_type,
    visual_prompt_data,
    ocr_output_format,
    ocr_granularity,
    font_size,
    draw_width,
    show_labels,
    custom_color,
):
    """Run inference using Rex Omni"""
    if image is None:
        return None, "Please upload an image first."

    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_width, image_height = image.size

        # Parse visual prompts if needed
        visual_prompt_boxes = []
        if task_selection == "Visual Prompting":
            # Check if we have predefined visual prompt boxes from examples
            if hasattr(image, "_example_visual_prompts"):
                visual_prompt_boxes = image._example_visual_prompts
            elif visual_prompt_data is not None and "points" in visual_prompt_data:
                visual_prompt_boxes = parse_visual_prompt(visual_prompt_data["points"])

        # Determine task type and categories based on task selection
        if task_selection == "OCR":
            # For OCR, use the selected output format to determine task type
            task_type = OCR_OUTPUT_FORMATS[ocr_output_format]["task_type"]
            task_key = task_type.value
            # Use granularity level to determine categories
            categories_list = [OCR_GRANULARITY_LEVELS[ocr_granularity]["categories"]]
        elif task_selection == "Visual Prompting":
            # For visual prompting, we don't need explicit categories
            task_key = "visual_prompting"
            categories_list = ["object"]

            # Check if visual prompt boxes are provided
            if not visual_prompt_boxes:
                return (
                    None,
                    "Please draw bounding boxes on the image to provide visual examples for Visual Prompting task.",
                )
        elif task_selection == "Keypoint":
            task_key = "keypoint"
            categories_list = [keypoint_type] if keypoint_type else ["person"]
        else:
            # For other tasks, get task type from demo config
            demo_config = DEMO_TASK_CONFIGS[task_selection]
            task_type = demo_config["task_type"]
            task_key = task_type.value

            # Split categories by comma and clean up
            categories_list = [
                cat.strip() for cat in categories.split(",") if cat.strip()
            ]
            if not categories_list:
                categories_list = ["object"]

        # Run inference
        if task_selection == "Visual Prompting":
            results = rex_model.inference(
                images=image,
                task=task_key,
                categories=categories_list,
                visual_prompt_boxes=visual_prompt_boxes,
            )
        elif task_selection == "Keypoint":
            results = rex_model.inference(
                images=image,
                task=task_key,
                categories=categories_list,
                keypoint_type=keypoint_type if keypoint_type else "person",
            )
        else:
            results = rex_model.inference(
                images=image, task=task_key, categories=categories_list
            )

        result = results[0]

        # Check if inference was successful
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error occurred during inference")
            return None, f"Inference failed: {error_msg}"

        # Get predictions and raw output
        predictions = result["extracted_predictions"]
        raw_output = result["raw_output"]

        # Create visualization
        try:
            vis_image = RexOmniVisualize(
                image=image,
                predictions=predictions,
                font_size=font_size,
                draw_width=draw_width,
                show_labels=show_labels,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return vis_image, raw_output
        except Exception as e:
            return image, f"Visualization failed: {str(e)}\n\nRaw output:\n{raw_output}"

    except Exception as e:
        return None, f"Error during inference: {str(e)}"


def update_interface(task_selection):
    """Update interface based on task selection"""
    config = DEMO_TASK_CONFIGS.get(task_selection, {})

    if task_selection == "Visual Prompting":
        return (
            gr.update(visible=False),  # categories
            gr.update(visible=False),  # keypoint_type
            gr.update(visible=True),  # visual_prompt_tab
            gr.update(visible=False),  # ocr_config_group
            gr.update(value=config.get("description", "")),  # task_description
        )
    elif task_selection == "Keypoint":
        return (
            gr.update(visible=False),  # categories
            gr.update(visible=True),  # keypoint_type
            gr.update(visible=False),  # visual_prompt_tab
            gr.update(visible=False),  # ocr_config_group
            gr.update(value=config.get("description", "")),  # task_description
        )
    elif task_selection == "OCR":
        return (
            gr.update(visible=False),  # categories
            gr.update(visible=False),  # keypoint_type
            gr.update(visible=False),  # visual_prompt_tab
            gr.update(visible=True),  # ocr_config_group
            gr.update(value=config.get("description", "")),  # task_description
        )
    else:
        return (
            gr.update(
                visible=True, placeholder=config.get("example_categories", "")
            ),  # categories
            gr.update(visible=False),  # keypoint_type
            gr.update(visible=False),  # visual_prompt_tab
            gr.update(visible=False),  # ocr_config_group
            gr.update(value=config.get("description", "")),  # task_description
        )


def load_example_image(image_path, visual_prompt_boxes=None):
    """Load example image from tutorials directory"""
    if image_path is None:
        return None

    try:
        import os

        from PIL import Image

        # Construct full path
        full_path = os.path.join(os.path.dirname(__file__), image_path)
        if os.path.exists(full_path):
            image = Image.open(full_path).convert("RGB")

            # Attach visual prompt boxes if provided (for Visual Prompting examples)
            if visual_prompt_boxes:
                image._example_visual_prompts = visual_prompt_boxes

            return image
        else:
            print(f"Warning: Example image not found at {full_path}")
            return None
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None


def prepare_gallery_data():
    """Prepare gallery data for examples"""
    gallery_images = []
    gallery_captions = []

    for config in EXAMPLE_CONFIGS:
        # Load example image
        image = load_example_image(config["image_path"], config["visual_prompt_boxes"])
        if image:
            gallery_images.append(image)
            gallery_captions.append(f"{config['name']}\n{config['description']}")

    return gallery_images, gallery_captions


def update_example_selection(selected_index):
    """Update all interface elements based on gallery selection"""
    if selected_index is None or selected_index >= len(EXAMPLE_CONFIGS):
        return [gr.update() for _ in range(7)]  # Return no updates if invalid selection

    config = EXAMPLE_CONFIGS[selected_index]

    # Load example image if available
    example_image = None
    if config["image_path"]:
        example_image = load_example_image(
            config["image_path"], config["visual_prompt_boxes"]
        )

    return (
        example_image,  # input_image
        config["task"],  # task_selection
        config["categories"],  # categories
        config["keypoint_type"],  # keypoint_type
        config["ocr_output_format"],  # ocr_output_format
        config["ocr_granularity"],  # ocr_granularity
        gr.update(
            value=DEMO_TASK_CONFIGS[config["task"]]["description"]
        ),  # task_description
    )


def update_prompt_preview(
    task_selection,
    categories,
    keypoint_type,
    visual_prompt_data,
    ocr_output_format,
    ocr_granularity,
):
    """Update the prompt preview"""
    if visual_prompt_data is None:
        visual_prompt_data = {}

    # Parse visual prompts
    visual_prompt_boxes = []
    if "points" in visual_prompt_data:
        visual_prompt_boxes = parse_visual_prompt(visual_prompt_data["points"])

    # Generate prompt preview
    prompt = get_task_prompt(
        task_selection,
        categories,
        keypoint_type,
        visual_prompt_boxes,
        800,  # dummy image dimensions for preview
        600,
        ocr_output_format=ocr_output_format,
        ocr_granularity=ocr_granularity,
    )

    return prompt


def create_demo():
    """Create the Gradio demo interface"""

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue"),
        title="Rex Omni Demo",
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .prompt-preview {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 0.75rem;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
        }
        .preserve-aspect-ratio img {
            object-fit: contain !important;
            max-height: 400px !important;
            width: auto !important;
        }
        .preserve-aspect-ratio canvas {
            object-fit: contain !important;
            max-height: 400px !important;
            width: auto !important;
        }
        """,
    ) as demo:

        gr.Markdown("# Rex Omni: Detect Anything Demo")
        gr.Markdown("Upload an image and select a task to see Rex Omni in action!")

        with gr.Row():
            # Left Column - Input Controls
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Task Configuration")

                # Task Selection
                task_selection = gr.Dropdown(
                    label="Select Task",
                    choices=list(DEMO_TASK_CONFIGS.keys()),
                    value="Detection",
                    info="Choose the vision task to perform",
                )

                # Task Description
                task_description = gr.Textbox(
                    label="Task Description",
                    value=DEMO_TASK_CONFIGS["Detection"]["description"],
                    interactive=False,
                    lines=2,
                )

                # Text Prompt Section
                with gr.Group():
                    gr.Markdown("### 💬 Text Prompt Configuration")

                    categories = gr.Textbox(
                        label="Categories",
                        value="person, car, dog",
                        placeholder="person, car, dog",
                        info="Enter object categories separated by commas",
                        visible=True,
                    )

                    keypoint_type = gr.Dropdown(
                        label="Keypoint Type",
                        choices=list(KEYPOINT_CONFIGS.keys()),
                        value="person",
                        visible=False,
                        info="Select the type of keypoints to detect",
                    )

                    # OCR Configuration Section
                    ocr_config_group = gr.Group(visible=False)
                    with ocr_config_group:
                        gr.Markdown("### 📄 OCR Configuration")

                        ocr_output_format = gr.Radio(
                            label="Output Format",
                            choices=list(OCR_OUTPUT_FORMATS.keys()),
                            value="Box",
                            info="Choose between bounding box or polygon output format",
                        )

                        ocr_granularity = gr.Radio(
                            label="Granularity Level",
                            choices=list(OCR_GRANULARITY_LEVELS.keys()),
                            value="Word Level",
                            info="Choose between word-level or text-line-level detection",
                        )

                # Visual Prompt Section
                visual_prompt_tab = gr.Group(visible=False)
                with visual_prompt_tab:
                    gr.Markdown("### 🎯 Visual Prompt Configuration")
                    gr.Markdown(
                        "Draw bounding boxes on the image to provide visual examples"
                    )

                # Prompt Preview
                gr.Markdown("### 🔍 Generated Prompt Preview")
                prompt_preview = gr.Textbox(
                    label="Actual Prompt",
                    value="Detect person, car, dog.",
                    interactive=False,
                    lines=3,
                    elem_classes=["prompt-preview"],
                )

                # Visualization Controls
                with gr.Accordion("🎨 Visualization Settings", open=False):
                    font_size = gr.Slider(
                        label="Font Size", value=20, minimum=10, maximum=50, step=1
                    )
                    draw_width = gr.Slider(
                        label="Line Width", value=5, minimum=1, maximum=20, step=1
                    )
                    show_labels = gr.Checkbox(label="Show Labels", value=True)
                    custom_color = gr.Textbox(
                        label="Custom Colors (Hex)",
                        placeholder="#FF0000,#00FF00,#0000FF",
                        info="Comma-separated hex colors for different categories",
                    )

            # Right Column - Image and Results
            with gr.Column(scale=2):
                with gr.Row():
                    # Input Image
                    with gr.Column():
                        input_image = gr.Image(
                            label="📷 Input Image", type="numpy", height=400
                        )

                        # Visual Prompt Interface (only visible for Visual Prompting task)
                        visual_prompter = ImagePrompter(
                            label="🎯 Visual Prompt Interface",
                            width=420,
                            height=315,  # 4:3 aspect ratio (420 * 3/4 = 315)
                            visible=False,
                            elem_classes=["preserve-aspect-ratio"],
                        )

                    # Output Visualization
                    with gr.Column():
                        output_image = gr.Image(
                            label="🎨 Visualization Result", height=400
                        )

                # Run Button
                run_button = gr.Button("🚀 Run Inference", variant="primary", size="lg")

                # Model Output
                model_output = gr.Textbox(
                    label="🤖 Model Raw Output",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True,
                )

        # Example Gallery Section
        with gr.Row():
            gr.Markdown("## 🖼️ Example Gallery")

        with gr.Row():
            gallery_images, gallery_captions = prepare_gallery_data()
            example_gallery = gr.Gallery(
                value=list(zip(gallery_images, gallery_captions)),
                label="Click on an example to load it",
                show_label=True,
                elem_id="example_gallery",
                columns=4,
                rows=2,
                height="auto",
                allow_preview=True,
            )

        # Event Handlers

        # Update interface when gallery example is selected
        def handle_gallery_select(evt: gr.SelectData):
            return update_example_selection(evt.index)

        example_gallery.select(
            fn=handle_gallery_select,
            outputs=[
                input_image,
                task_selection,
                categories,
                keypoint_type,
                ocr_output_format,
                ocr_granularity,
                task_description,
            ],
        )

        # Update interface when task changes
        task_selection.change(
            fn=update_interface,
            inputs=[task_selection],
            outputs=[
                categories,
                keypoint_type,
                visual_prompt_tab,
                ocr_config_group,
                task_description,
            ],
        )

        # Update prompt preview when any input changes
        for component in [
            task_selection,
            categories,
            keypoint_type,
            ocr_output_format,
            ocr_granularity,
        ]:
            component.change(
                fn=update_prompt_preview,
                inputs=[
                    task_selection,
                    categories,
                    keypoint_type,
                    visual_prompter,
                    ocr_output_format,
                    ocr_granularity,
                ],
                outputs=[prompt_preview],
            )

        # Show/hide visual prompter based on task
        def toggle_visual_prompter(task_selection):
            if task_selection == "Visual Prompting":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)

        task_selection.change(
            fn=toggle_visual_prompter,
            inputs=[task_selection],
            outputs=[input_image, visual_prompter],
        )

        # Run inference with dynamic image selection
        def run_inference_wrapper(
            input_image,
            visual_prompter_data,
            task_selection,
            categories,
            keypoint_type,
            ocr_output_format,
            ocr_granularity,
            font_size,
            draw_width,
            show_labels,
            custom_color,
        ):
            # For Visual Prompting task, use the visual prompter image
            if task_selection == "Visual Prompting":
                if visual_prompter_data is not None and "image" in visual_prompter_data:
                    image_to_use = visual_prompter_data["image"]
                else:
                    return (
                        None,
                        "Please upload an image in the Visual Prompt Interface for Visual Prompting task.",
                    )
            else:
                image_to_use = input_image

            return run_inference(
                image_to_use,
                task_selection,
                categories,
                keypoint_type,
                visual_prompter_data,
                ocr_output_format,
                ocr_granularity,
                font_size,
                draw_width,
                show_labels,
                custom_color,
            )

        run_button.click(
            fn=run_inference_wrapper,
            inputs=[
                input_image,
                visual_prompter,
                task_selection,
                categories,
                keypoint_type,
                ocr_output_format,
                ocr_granularity,
                font_size,
                draw_width,
                show_labels,
                custom_color,
            ],
            outputs=[output_image, model_output],
        )

    return demo


if __name__ == "__main__":
    args = parse_args()

    print("🚀 Initializing Rex Omni model...")
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")

    # Initialize Rex Omni model
    rex_model = RexOmniWrapper(
        model_path=args.model_path,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    print("✅ Model initialized successfully!")

    # Create and launch demo
    demo = create_demo()

    print(f"🌐 Launching demo at http://{args.server_name}:{args.server_port}")
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        debug=True,
        inbrowser=True,
    )
