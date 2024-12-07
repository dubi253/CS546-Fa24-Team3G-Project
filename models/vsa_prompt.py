# Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Engines
# Github source: https://github.com/cnzzx/VSA-dev
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on LLaVA, GroundingDINO, and MindSearch code bases
# https://github.com/haotian-liu/LLaVA
# https://github.com/IDEA-Research/GroundingDINO
# https://github.com/InternLM/MindSearch
# --------------------------------------------------------'
COCO_CLASSES =[
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush',
]


CAPTION_FORMAT = """
Please describe the {label} in detail in order to answer the user's question: \"{text}\". 
Please only output detailed description of every organ and part of the image provided without other additional comments on the image and question.
"""

CORRELATE_FORMAT = """
Please correct the description \"{caption}\" of the given object briefly.
Please refer to the descriptions of other objects in the same scene: {other_captions}
"""

QA_FORMAT = """
Please answer the user\'s question: \"{text}\" according to the given image in English. \
The following information might be related to the image and please selectively refer to them: {contexts}
"""


def get_caption_prompt(label, text):
    prompt = CAPTION_FORMAT.replace('{label}', label)
    prompt = prompt.replace('{text}', text)
    return prompt


def get_correlate_prompt(caption, other_captions):
    prompt = CORRELATE_FORMAT.replace('{caption}', caption)
    prompt = prompt.replace('{other_captions}', ' '.join(other_captions))
    return prompt

import logging
def get_qa_prompt(text, contexts):
    prompt = QA_FORMAT.replace('{text}', text)
    prompt = prompt.replace('{contexts}', ' '.join(contexts))
    return prompt
