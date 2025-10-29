import random
from typing import Dict, List


class GroundingTaskFn(object):
    """This is for detection dataset tsv training

    Args:
        task_prompts (list[str]): A list of prompts to random choose from.
        image_min_pixels (int): The minimal number of pixels for the resized image.
        image_max_pixels (int): The maximal number of pixels for the resized image.
        extra_categories (List[str], optional): A list of all possible category names. Used for negative sampling.
            If None, only positive examples will be used. Default: None.

    Returns:
        - dict: Dict with the following keys:
            "conversations" (List): [
                {
                    "from": "human",
                    "value": "<image>\nCan you detect the dog, cat in this image? Answer the question in json format."
                },
                {
                    "from": "gpt",
                    "value": "<|object_ref_start|>dog<|object_ref_end|><|box_start|>x0y0x1y1, x0y0x1y1<|box_end|>, <|object_ref_start|>cat<|object_ref_end|><|box_start|>None<|box_end|>"
                },
            ]
            # ! Note: The coordinates are now normalized to [0, 999] bins. If coord_to_word_map is provided,
            # ! the coordinates will be mapped to corresponding word tokens. Otherwise, they remain as integers.
            # ! For negative examples, the box coordinates will be "None".
    """

    def __init__(
        self,
        task_prompts,
        image_min_pixels,
        image_max_pixels,
        extra_categories: List = None,
        **kwargs,
    ):
        self.min_pixels = image_min_pixels
        self.max_pixels = image_max_pixels
        self.task_prompts = task_prompts
        self.extra_categories = extra_categories

        if extra_categories is not None:
            # modify category name
            self.extra_categories = [
                self.modify_cateogry_name(category_name)
                for category_name in extra_categories
            ]

    def modify_cateogry_name(self, category_name: str):
        """Process the category name to be more readable.

        Args:
            region_map (Dict): Region map from the input.
        """
        try:
            if "/" in category_name:
                # If the category name contains '/', replace it with '_'
                category_name = category_name.split("/")[0]
            category_name = category_name.replace("_", " ").replace(",", "")
        except Exception as e:
            raise ValueError(f"Error modifying category name: {category_name}")
        return category_name

    def convert_boxes_from_absolute_to_normalized_bins(
        self, gt_boxes, ori_width, ori_height
    ):
        """Convert boxes from absolute coordinates to normalized bins (0-999) and map to words.

        Args:
            gt_boxes: List of boxes in absolute coordinates
            ori_width: Original image width
            ori_height: Original image height

        Returns:
            List of boxes with coordinates mapped to words
        """
        # Step 1: Convert to normalized bins
        normalized_gt_boxes = []
        for box in gt_boxes:
            # Normalize coordinates to [0, 1] range
            x0, y0, x1, y1 = box
            x0_norm = x0 / ori_width
            x1_norm = x1 / ori_width
            y0_norm = y0 / ori_height
            y1_norm = y1 / ori_height

            # Clip to [0, 1] range
            x0_norm = max(0.0, min(1.0, x0_norm))
            x1_norm = max(0.0, min(1.0, x1_norm))
            y0_norm = max(0.0, min(1.0, y0_norm))
            y1_norm = max(0.0, min(1.0, y1_norm))

            # Convert to bins [0, 999]
            x0_bin = int(x0_norm * 999)
            y0_bin = int(y0_norm * 999)
            x1_bin = int(x1_norm * 999)
            y1_bin = int(y1_norm * 999)

            # Ensure bins are in valid range [0, 999]
            x0_bin = max(0, min(999, x0_bin))
            y0_bin = max(0, min(999, y0_bin))
            x1_bin = max(0, min(999, x1_bin))
            y1_bin = max(0, min(999, y1_bin))

            normalized_gt_boxes.append([x0_bin, y0_bin, x1_bin, y1_bin])

        # Step 2: Sort boxes based on x0
        normalized_gt_boxes.sort(key=lambda box: box[0])

        # Step 3: Map to words using coord_to_word_map
        word_mapped_boxes = []
        for box in normalized_gt_boxes:
            x0_bin, y0_bin, x1_bin, y1_bin = box
            # check if x1 > x0 and y1 > y0
            if x1_bin < x0_bin or y1_bin < y0_bin:
                print(
                    f"x1_bin <= x0_bin or y1_bin <= y0_bin: {x1_bin} <= {x0_bin} or {y1_bin} <= {y0_bin}"
                )
                print(f"box: {box}")
                print(f"normalized_gt_boxes: {normalized_gt_boxes}")
                print(f"ori_width: {ori_width}, ori_height: {ori_height}")
            word_mapped_boxes.append(
                "".join(
                    [
                        f"<{x0_bin}>",
                        f"<{y0_bin}>",
                        f"<{x1_bin}>",
                        f"<{y1_bin}>",
                    ]
                )
            )

        return word_mapped_boxes

    def compose_answer(self, qa_pair):
        """
        Compose the answer for the question.
        """
        answer = []
        for category_name, bboxes in qa_pair.items():
            # Check if this is a negative example (no boxes)
            if bboxes is None or len(bboxes) == 0:
                # Negative example - object not found in image
                answer.append(
                    f"<|object_ref_start|>{category_name}<|object_ref_end|><|box_start|>None<|box_end|>"
                )
            else:
                # Positive example - format bboxes for output
                bbox_strings = []
                for bbox in bboxes:
                    # Join the four coordinate words directly without spaces
                    bbox_strings.append(bbox)
                bboxes_formatted = ",".join(bbox_strings)

                answer.append(
                    f"<|object_ref_start|>{category_name}<|object_ref_end|><|box_start|>{bboxes_formatted}<|box_end|>"
                )
        return ", ".join(answer)

    def step1_compose_qa_pair(self, annotations, ori_width, ori_height):
        """
        Compose a dict for qa pair. The key is the cateogry name and the value is a list of bbox coordinates after resized.

        Args:
            annotations (Dict): Annotation dict with the following keys:
                {
                    "boxes": List[List[float]], a list of bbox coordinates in xyxy format
                    "labels": List[int], a list of category ids
                    "size": Tuple[int, int], the original size of the image
                }

        Returns:
            Dict: A dict for qa pair. The key is the cateogry name and the value is a list of bbox coordinates after resized.
        """
        # Get positive categories from current image
        positive_categories = set()
        qa_pair = {}

        for box, label in zip(annotations["boxes"], annotations["labels"]):
            category_name = self.modify_cateogry_name(label)

            positive_categories.add(category_name)
            if category_name not in qa_pair:
                qa_pair[category_name] = []
            qa_pair[category_name].append(box)

        # Add negative categories if extra_categories is provided
        if self.extra_categories is not None:
            # Get negative categories (categories not in current image)
            negative_categories = set(self.extra_categories) - positive_categories

            # Add all negative categories with None boxes
            for cat in negative_categories:
                qa_pair[cat] = None

        # Convert positive boxes to normalized bins
        for category_name, bboxes in qa_pair.items():
            if bboxes is not None:  # Only process positive examples
                qa_pair[category_name] = (
                    self.convert_boxes_from_absolute_to_normalized_bins(
                        bboxes, ori_width, ori_height
                    )
                )

        # last step, shuffle qa_pair
        qa_pair = dict(sorted(qa_pair.items(), key=lambda x: random.random()))

        return qa_pair

    def __call__(self, example, ori_width, ori_height):
        """
        example (dict): Example from a detection tsv dataset.
            {
                "image_lineidx" (int): The line index of the image in the tsv file.
                "annotations" (Dict): Annotation dict with the following keys:
                    {
                        "boxes": List[List[float]], a list of bbox coordinates in xyxy format
                        "labels": List[int], a list of category ids
                        "size": Tuple[int, int], the original size of the image
                    }
            },
        ori_width (int): The original width of the image.
        ori_height (int): The original height of the image.
        """
        # step1 build qa pair
        qa_pair = self.step1_compose_qa_pair(
            example["annotations"], ori_width, ori_height
        )

        # step2 build conversation
        question = random.choice(self.task_prompts)
        conversations = []
        questioned_categories = list(qa_pair.keys())
        question = question.replace("[OBJ]", ", ".join(questioned_categories))

        # conversation from human
        conversation_from_human = {}
        conversation_from_human["from"] = "human"
        conversation_from_human["value"] = f"<image>\n{question}"
        conversations.append(conversation_from_human)
        # conversation from gpt
        conversation_from_gpt = {}
        conversation_from_gpt["from"] = "gpt"
        answer = self.compose_answer(qa_pair)
        conversation_from_gpt["value"] = answer
        conversations.append(conversation_from_gpt)
        example["conversations"] = conversations
        return example
