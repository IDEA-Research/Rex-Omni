#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main wrapper class for Rex Omni
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize

from .parser import convert_boxes_to_normalized_bins, parse_prediction
from .tasks import TASK_CONFIGS, TaskType, get_keypoint_config, get_task_config


class RexOmniWrapper:
    """
    High-level wrapper for Rex-Omni
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        system_prompt: str = "You are a helpful assistant",
        min_pixels: int = 16 * 28 * 28,
        max_pixels: int = 2560 * 28 * 28,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 1,
        repetition_penalty: float = 1.05,
        skip_special_tokens: bool = False,
        stop: Optional[List[str]] = None,
        quantization: str = None,
        **kwargs,
    ):
        """
        Initialize the wrapper

        Args:
            model_path: Path to the model directory
            backend: Backend type ("transformers" or "vllm")
            system_prompt: System prompt for the model
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            skip_special_tokens: Whether to skip special tokens in output
            stop: Stop sequences for generation
            quantization: Quantization type
            **kwargs: Additional arguments for model initialization
        """
        self.model_path = model_path
        self.backend = backend.lower()
        self.system_prompt = system_prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Store generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.skip_special_tokens = skip_special_tokens
        self.stop = stop or ["<|im_end|>"]
        self.quantization = quantization

        # Initialize model and processor
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        """Initialize model and processor based on backend type"""
        print(f"Initializing {self.backend} backend...")

        if self.backend == "vllm":
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            # Initialize VLLM model
            self.model = LLM(
                model=self.model_path,
                tokenizer=self.model_path,
                tokenizer_mode=kwargs.get("tokenizer_mode", "slow"),
                limit_mm_per_prompt=kwargs.get(
                    "limit_mm_per_prompt", {"image": 10, "video": 10}
                ),
                max_model_len=kwargs.get("max_model_len", 4096),
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.8),
                tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
                trust_remote_code=kwargs.get("trust_remote_code", True),
                quantization=self.quantization,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "tokenizer_mode",
                        "limit_mm_per_prompt",
                        "max_model_len",
                        "gpu_memory_utilization",
                        "tensor_parallel_size",
                        "trust_remote_code",
                    ]
                },
            )

            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Set padding side to left for batch inference with Flash Attention
            self.processor.tokenizer.padding_side = "left"

            # Set up sampling parameters
            self.sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                temperature=self.temperature,
                skip_special_tokens=self.skip_special_tokens,
                stop=self.stop,
            )

            self.model_type = "vllm"

        elif self.backend == "transformers":
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            # Initialize transformers model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                attn_implementation=kwargs.get(
                    "attn_implementation", "flash_attention_2"
                ),
                device_map=kwargs.get("device_map", "auto"),
                trust_remote_code=kwargs.get("trust_remote_code", True),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "torch_dtype",
                        "attn_implementation",
                        "device_map",
                        "trust_remote_code",
                    ]
                },
            )

            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                use_fast=False,
            )

            # Set padding side to left for batch inference with Flash Attention
            self.processor.tokenizer.padding_side = "left"

            self.model_type = "transformers"

        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. Choose 'transformers' or 'vllm'."
            )

    def inference(
        self,
        images: Union[Image.Image, List[Image.Image]],
        task: Union[str, TaskType, List[Union[str, TaskType]]],
        categories: Optional[Union[str, List[str], List[List[str]]]] = None,
        keypoint_type: Optional[Union[str, List[str]]] = None,
        visual_prompt_boxes: Optional[
            Union[List[List[float]], List[List[List[float]]]]
        ] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Perform batch inference on images for various vision tasks.

        Args:
            images: Input image(s) in PIL.Image format. Can be single image or list of images.
            task: Task type(s). Can be single task or list of tasks for batch processing.
                Available options:
                - "detection": Object detection with bounding boxes
                - "pointing": Point to objects with coordinates
                - "visual_prompting": Find similar objects based on reference boxes
                - "keypoint": Detect keypoints for persons/hands/animals
                - "ocr_box": Detect and recognize text in bounding boxes
                - "ocr_polygon": Detect and recognize text in polygons
                - "gui_grounding": Detect gui element and return in box format
                - "gui_pointing": Point to gui element and return in point format
            categories: Object categories to detect/locate. Can be:
                - Single string: "person"
                - List of strings: ["person", "car"] (applied to all images)
                - List of lists: [["person"], ["car", "dog"]] (per-image categories)
            keypoint_type: Type of keypoints for keypoint detection task.
                Can be single string or list of strings for batch processing.
                Options: "person", "hand", "animal"
            visual_prompt_boxes: Reference bounding boxes for visual prompting task.
                Can be single list or list of lists for batch processing.
                Format: [[x0, y0, x1, y1], ...] or [[[x0, y0, x1, y1], ...], ...]
            **kwargs: Additional arguments (reserved for future use)

        Returns:
            List of prediction dictionaries, one for each input image. Each dictionary contains:
            - success (bool): Whether inference succeeded
            - extracted_predictions (dict): Parsed predictions by category
            - raw_output (str): Raw model output text
            - inference_time (float): Total inference time in seconds
            - num_output_tokens (int): Number of generated tokens
            - num_prompt_tokens (int): Number of input tokens
            - tokens_per_second (float): Generation speed
            - image_size (tuple): Input image dimensions (width, height)
            - task (str): Task type used
            - prompt (str): Generated prompt sent to model

        Examples:
            # Single image object detection
            results = model.inference(
                images=image,
                task="detection",
                categories=["person", "car", "dog"]
            )

            # Batch processing with same task and categories
            results = model.inference(
                images=[img1, img2, img3],
                task="detection",
                categories=["person", "car"]
            )

            # Batch processing with different tasks per image
            results = model.inference(
                images=[img1, img2, img3],
                task=["detection", "pointing", "keypoint"],
                categories=[["person", "car"], ["dog"], ["person"]],
                keypoint_type=[None, None, "person"]
            )

            # Batch keypoint detection with different types
            results = model.inference(
                images=[img1, img2],
                task=["keypoint", "keypoint"],
                categories=[["person"], ["hand"]],
                keypoint_type=["person", "hand"]
            )

            # Batch visual prompting
            results = model.inference(
                images=[img1, img2],
                task="visual_prompting",
                visual_prompt_boxes=[
                    [[100, 100, 200, 200]],
                    [[50, 50, 150, 150], [300, 300, 400, 400]]
                ]
            )

            # Mixed batch processing
            results = model.inference(
                images=[img1, img2, img3],
                task=["detection", "ocr_box", "pointing"],
                categories=[["person", "car"], ["text"], ["dog"]]
            )
        """
        # Convert single image to list
        if isinstance(images, Image.Image):
            images = [images]

        batch_size = len(images)

        # Normalize inputs to batch format
        tasks, categories_list, keypoint_types, visual_prompt_boxes_list = (
            self._normalize_batch_inputs(
                task, categories, keypoint_type, visual_prompt_boxes, batch_size
            )
        )

        # Perform batch inference
        return self._inference_batch(
            images=images,
            tasks=tasks,
            categories_list=categories_list,
            keypoint_types=keypoint_types,
            visual_prompt_boxes_list=visual_prompt_boxes_list,
            **kwargs,
        )

    def _normalize_batch_inputs(
        self,
        task: Union[str, TaskType, List[Union[str, TaskType]]],
        categories: Optional[Union[str, List[str], List[List[str]]]],
        keypoint_type: Optional[Union[str, List[str]]],
        visual_prompt_boxes: Optional[
            Union[List[List[float]], List[List[List[float]]]]
        ],
        batch_size: int,
    ) -> Tuple[
        List[TaskType],
        List[Optional[List[str]]],
        List[Optional[str]],
        List[Optional[List[List[float]]]],
    ]:
        """Normalize all inputs to batch format"""

        # Normalize tasks
        if isinstance(task, (str, TaskType)):
            # Single task for all images
            if isinstance(task, str):
                task = TaskType(task.lower())
            tasks = [task] * batch_size
        else:
            # List of tasks
            tasks = []
            for t in task:
                if isinstance(t, str):
                    tasks.append(TaskType(t.lower()))
                else:
                    tasks.append(t)

            if len(tasks) != batch_size:
                raise ValueError(
                    f"Number of tasks ({len(tasks)}) must match number of images ({batch_size})"
                )

        # Normalize categories
        if categories is None:
            categories_list = [None] * batch_size
        elif isinstance(categories, str):
            # Single string for all images
            categories_list = [[categories]] * batch_size
        elif isinstance(categories, list):
            if len(categories) == 0:
                categories_list = [None] * batch_size
            elif isinstance(categories[0], str):
                # List of strings for all images
                categories_list = [categories] * batch_size
            else:
                # List of lists (per-image categories)
                categories_list = categories
                if len(categories_list) != batch_size:
                    raise ValueError(
                        f"Number of category lists ({len(categories_list)}) must match number of images ({batch_size})"
                    )
        else:
            categories_list = [None] * batch_size

        # Normalize keypoint_type
        if keypoint_type is None:
            keypoint_types = [None] * batch_size
        elif isinstance(keypoint_type, str):
            # Single keypoint type for all images
            keypoint_types = [keypoint_type] * batch_size
        else:
            # List of keypoint types
            keypoint_types = keypoint_type
            if len(keypoint_types) != batch_size:
                raise ValueError(
                    f"Number of keypoint types ({len(keypoint_types)}) must match number of images ({batch_size})"
                )

        # Normalize visual_prompt_boxes
        if visual_prompt_boxes is None:
            visual_prompt_boxes_list = [None] * batch_size
        elif isinstance(visual_prompt_boxes, list):
            if len(visual_prompt_boxes) == 0:
                visual_prompt_boxes_list = [None] * batch_size
            elif isinstance(visual_prompt_boxes[0], (int, float)):
                # Single box for all images: [x0, y0, x1, y1]
                visual_prompt_boxes_list = [[visual_prompt_boxes]] * batch_size
            elif isinstance(visual_prompt_boxes[0], list):
                if len(visual_prompt_boxes[0]) == 4 and isinstance(
                    visual_prompt_boxes[0][0], (int, float)
                ):
                    # List of boxes for all images: [[x0, y0, x1, y1], ...]
                    visual_prompt_boxes_list = [visual_prompt_boxes] * batch_size
                else:
                    # List of lists of boxes (per-image boxes): [[[x0, y0, x1, y1], ...], ...]
                    visual_prompt_boxes_list = visual_prompt_boxes
                    if len(visual_prompt_boxes_list) != batch_size:
                        raise ValueError(
                            f"Number of visual prompt box lists ({len(visual_prompt_boxes_list)}) must match number of images ({batch_size})"
                        )
            else:
                visual_prompt_boxes_list = [None] * batch_size
        else:
            visual_prompt_boxes_list = [None] * batch_size

        return tasks, categories_list, keypoint_types, visual_prompt_boxes_list

    def _inference_batch(
        self,
        images: List[Image.Image],
        tasks: List[TaskType],
        categories_list: List[Optional[List[str]]],
        keypoint_types: List[Optional[str]],
        visual_prompt_boxes_list: List[Optional[List[List[float]]]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Perform true batch inference"""

        start_time = time.time()
        batch_size = len(images)

        # Prepare batch data
        batch_messages = []
        batch_prompts = []
        batch_image_sizes = []

        for i in range(batch_size):
            image = images[i]
            task = tasks[i]
            categories = categories_list[i]
            keypoint_type = keypoint_types[i]
            visual_prompt_boxes = visual_prompt_boxes_list[i]

            # Get image dimensions
            w, h = image.size
            batch_image_sizes.append((w, h))

            # Generate prompt
            prompt = self._generate_prompt(
                task=task,
                categories=categories,
                keypoint_type=keypoint_type,
                visual_prompt_boxes=visual_prompt_boxes,
                image_width=w,
                image_height=h,
            )
            batch_prompts.append(prompt)

            # Calculate resized dimensions
            resized_height, resized_width = smart_resize(
                h,
                w,
                28,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Prepare messages
            if self.model_type == "transformers":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                                "resized_height": resized_height,
                                "resized_width": resized_width,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                                "min_pixels": self.min_pixels,
                                "max_pixels": self.max_pixels,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

            batch_messages.append(messages)

        # Perform batch generation
        if self.model_type == "vllm":
            batch_outputs, batch_generation_info = self._generate_vllm_batch(
                batch_messages
            )
        else:
            batch_outputs, batch_generation_info = self._generate_transformers_batch(
                batch_messages, images
            )

        # Parse results
        results = []
        total_time = time.time() - start_time

        for i in range(batch_size):
            raw_output = batch_outputs[i]
            generation_info = batch_generation_info[i]
            w, h = batch_image_sizes[i]
            task = tasks[i]
            prompt = batch_prompts[i]

            # Parse predictions
            extracted_predictions = parse_prediction(
                text=raw_output,
                w=w,
                h=h,
                task_type=task.value,
            )

            # Calculate resized dimensions for result
            resized_height, resized_width = smart_resize(
                h,
                w,
                28,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            result = {
                "success": True,
                "image_size": (w, h),
                "resized_size": (resized_width, resized_height),
                "task": task.value,
                "prompt": prompt,
                "raw_output": raw_output,
                "extracted_predictions": extracted_predictions,
                "inference_time": total_time,  # Total batch time
                **generation_info,
            }
            results.append(result)

        return results

    def _inference_single(
        self,
        image: Image.Image,
        task: TaskType,
        categories: Optional[Union[str, List[str]]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform inference on a single image"""

        start_time = time.time()

        # Get image dimensions
        w, h = image.size

        # Generate prompt based on task
        final_prompt = self._generate_prompt(
            task=task,
            categories=categories,
            keypoint_type=keypoint_type,
            visual_prompt_boxes=visual_prompt_boxes,
            image_width=w,
            image_height=h,
        )

        # Calculate resized dimensions using smart_resize
        resized_height, resized_width = smart_resize(
            h,
            w,
            28,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Prepare messages
        if self.model_type == "transformers":
            # For transformers, use resized_height and resized_width
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "resized_height": resized_height,
                            "resized_width": resized_width,
                        },
                        {"type": "text", "text": final_prompt},
                    ],
                },
            ]
        else:
            # For VLLM, use min_pixels and max_pixels
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": final_prompt},
                    ],
                },
            ]

        # Generate response
        if self.model_type == "vllm":
            raw_output, generation_info = self._generate_vllm(messages)
        else:
            raw_output, generation_info = self._generate_transformers(messages)

        # Parse predictions
        extracted_predictions = parse_prediction(
            text=raw_output,
            w=w,
            h=h,
            task_type=task.value,
        )

        # Calculate timing
        total_time = time.time() - start_time

        return {
            "success": True,
            "image_size": (w, h),
            "resized_size": (resized_width, resized_height),
            "task": task.value,
            "prompt": final_prompt,
            "raw_output": raw_output,
            "extracted_predictions": extracted_predictions,
            "inference_time": total_time,
            **generation_info,
        }

    def _generate_prompt(
        self,
        task: TaskType,
        categories: Optional[Union[str, List[str]]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        image_width: int = None,
        image_height: int = None,
    ) -> str:
        """Generate prompt based on task configuration"""

        task_config = get_task_config(task)

        if task == TaskType.VISUAL_PROMPTING:
            if visual_prompt_boxes is None:
                raise ValueError(
                    "Visual prompt boxes are required for visual prompting task"
                )

            # Convert boxes to normalized bins format
            word_mapped_boxes = convert_boxes_to_normalized_bins(
                visual_prompt_boxes, image_width, image_height
            )
            visual_prompt_dict = {"object_1": word_mapped_boxes}
            visual_prompt_json = json.dumps(visual_prompt_dict)

            return task_config.prompt_template.format(visual_prompt=visual_prompt_json)

        elif task == TaskType.KEYPOINT:
            if categories is None:
                raise ValueError("Categories are required for keypoint task")
            if keypoint_type is None:
                raise ValueError("Keypoint type is required for keypoint task")

            keypoints_list = get_keypoint_config(keypoint_type)
            if keypoints_list is None:
                raise ValueError(f"Unknown keypoint type: {keypoint_type}")

            keypoints_str = ", ".join(keypoints_list)
            categories_str = (
                ", ".join(categories) if isinstance(categories, list) else categories
            )

            return task_config.prompt_template.format(
                categories=categories_str, keypoints=keypoints_str
            )

        else:
            # Standard tasks (detection, pointing, OCR, etc.)
            if task_config.requires_categories and categories is None:
                raise ValueError(f"Categories are required for {task.value} task")

            if categories is not None:
                categories_str = (
                    ", ".join(categories)
                    if isinstance(categories, list)
                    else categories
                )
                return task_config.prompt_template.format(categories=categories_str)
            else:
                return task_config.prompt_template.format(categories="objects")

    def _generate_vllm(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Generate using VLLM model"""

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {"image": image_inputs}
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        # Generate
        generation_start = time.time()
        outputs = self.model.generate(
            [llm_inputs], sampling_params=self.sampling_params
        )
        generation_time = time.time() - generation_start

        generated_text = outputs[0].outputs[0].text

        # Extract token information
        output_tokens = outputs[0].outputs[0].token_ids
        num_output_tokens = len(output_tokens) if output_tokens else 0

        prompt_token_ids = outputs[0].prompt_token_ids
        num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0

        tokens_per_second = (
            num_output_tokens / generation_time if generation_time > 0 else 0
        )

        return generated_text, {
            "num_output_tokens": num_output_tokens,
            "num_prompt_tokens": num_prompt_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }

    def _generate_vllm_batch(
        self, batch_messages: List[List[Dict]]
    ) -> Tuple[List[str], List[Dict]]:
        """Generate using VLLM model for batch processing"""

        # Process all messages
        batch_inputs = []
        for messages in batch_messages:
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {"image": image_inputs}
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            batch_inputs.append(llm_inputs)

        # Generate for entire batch
        generation_start = time.time()
        outputs = self.model.generate(
            batch_inputs, sampling_params=self.sampling_params
        )
        generation_time = time.time() - generation_start

        # Extract results
        batch_outputs = []
        batch_generation_info = []

        for output in outputs:
            generated_text = output.outputs[0].text
            batch_outputs.append(generated_text)

            # Extract token information
            output_tokens = output.outputs[0].token_ids
            num_output_tokens = len(output_tokens) if output_tokens else 0

            prompt_token_ids = output.prompt_token_ids
            num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0

            tokens_per_second = (
                num_output_tokens / generation_time if generation_time > 0 else 0
            )

            generation_info = {
                "num_output_tokens": num_output_tokens,
                "num_prompt_tokens": num_prompt_tokens,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
            }
            batch_generation_info.append(generation_info)

        return batch_outputs, batch_generation_info

    def _generate_transformers(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Generate using Transformers model"""

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        generation_start = time.time()
        inputs = self.processor(
            text=[text],
            images=[messages[1]["content"][0]["image"]],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,  # Enable sampling if temperature > 0
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - generation_start

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )[0]

        num_output_tokens = len(generated_ids_trimmed[0])
        num_prompt_tokens = len(inputs.input_ids[0])
        tokens_per_second = (
            num_output_tokens / generation_time if generation_time > 0 else 0
        )

        return output_text, {
            "num_output_tokens": num_output_tokens,
            "num_prompt_tokens": num_prompt_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }

    def _generate_transformers_batch(
        self, batch_messages: List[List[Dict]], batch_images: List[Image.Image]
    ) -> Tuple[List[str], List[Dict]]:
        """Generate using Transformers model for batch processing"""

        # Prepare batch inputs
        batch_texts = []
        for messages in batch_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        # Process inputs for batch
        generation_start = time.time()
        inputs = self.processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Generate for entire batch
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - generation_start

        # Decode batch results
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        batch_outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        # Prepare generation info for each item
        batch_generation_info = []
        for i, output_ids in enumerate(generated_ids_trimmed):
            num_output_tokens = len(output_ids)
            num_prompt_tokens = len(inputs.input_ids[i])
            tokens_per_second = (
                num_output_tokens / generation_time if generation_time > 0 else 0
            )

            generation_info = {
                "num_output_tokens": num_output_tokens,
                "num_prompt_tokens": num_prompt_tokens,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
            }
            batch_generation_info.append(generation_info)

        return batch_outputs, batch_generation_info

    def get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks"""
        return [task.value for task in TaskType]

    def get_task_info(self, task: Union[str, TaskType]) -> Dict[str, Any]:
        """Get information about a specific task"""
        if isinstance(task, str):
            task = TaskType(task.lower())

        config = get_task_config(task)
        return {
            "name": config.name,
            "description": config.description,
            "output_format": config.output_format,
            "requires_categories": config.requires_categories,
            "requires_visual_prompt": config.requires_visual_prompt,
            "requires_keypoint_type": config.requires_keypoint_type,
            "prompt_template": config.prompt_template,
        }
