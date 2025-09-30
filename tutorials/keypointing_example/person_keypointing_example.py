"""
Person keypointing example using Rex Omni
"""

from PIL import Image


from rex_omni import RexOmniVisualize, RexOmniWrapper


def main():
    # Model path - replace with your actual model path
    model_path = "IDEA-Research/Rex-Omni"

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
    image_path = "tutorials/keypointing_example/test_images/person.png"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Image loaded successfully!")
    print(f"📏 Image size: {image.size}")

    # Person keypointing
    print("👤 Performing person keypointing...")
    results = rex_model.inference(
        images=image, task="keypoint", keypoint_type="person", categories=["person"]
    )

    # Process results
    result = results[0]
    if result["success"]:
        predictions = result["extracted_predictions"]
        vis_image = RexOmniVisualize(
            image=image,
            predictions=predictions,
            font_size=6,
            draw_width=6,
            show_labels=True,
        )

        # Save visualization
        output_path = (
            "tutorials/keypointing_example/test_images/person_keypointing_visualize.jpg"
        )
        vis_image.save(output_path)
        print(f"✅ Person keypointing visualization saved to: {output_path}")

        # Display results
        print(f"📊 Detected {len(predictions)} person keypoints")
        for i, pred in enumerate(predictions):
            print(
                f"  Keypoint {i+1}: {pred.get('text', 'N/A')} - Confidence: {pred.get('confidence', 'N/A')}"
            )

    else:
        print(f"❌ Inference failed: {result['error']}")


if __name__ == "__main__":
    main()
