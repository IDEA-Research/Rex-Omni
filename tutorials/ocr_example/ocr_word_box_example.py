#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR word-level detection example using Rex Omni (box format)
"""

import matplotlib.pyplot as plt
import torch
from PIL import Image

from rex_omni import RexOmniVisualize, RexOmniWrapper


def main():
    # Model path - replace with your actual model path
    model_path = "/comp_robot/jiangqing/projects/2023/research/R1/QwenSFTOfficial/open_source/IDEA-Research/Rex-Omni"

    print("🚀 Initializing Rex Omni model...")

    # Create wrapper with custom parameters
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",  # Choose "transformers" or "vllm"
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    # Load image
    image_path = (
        "tutorials/ocr_example/test_images/ocr.png"  # Replace with your image path
    )
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Image loaded successfully!")
    print(f"📏 Image size: {image.size}")

    # OCR word-level detection in box format
    categories = ["word"]

    print("🔍 Performing word-level OCR detection...")
    results = rex_model.inference(images=image, task="ocr_box", categories=categories)

    # Process results
    result = results[0]
    if result["success"]:
        predictions = result["extracted_predictions"]
        vis_image = RexOmniVisualize(
            image=image,
            predictions=predictions,
            font_size=20,
            draw_width=5,
            show_labels=True,
        )

        # Save visualization
        output_path = "tutorials/ocr_example/test_images/ocr_word_box_visualize.jpg"
        vis_image.save(output_path)
        print(f"✅ Word-level OCR visualization saved to: {output_path}")

        # Display results
        print(f"📊 Detected {len(predictions)} words")
        for i, pred in enumerate(predictions):
            print(
                f"  Word {i+1}: {pred.get('text', 'N/A')} - Confidence: {pred.get('confidence', 'N/A')}"
            )

    else:
        print(f"❌ Inference failed: {result['error']}")


if __name__ == "__main__":
    main()
