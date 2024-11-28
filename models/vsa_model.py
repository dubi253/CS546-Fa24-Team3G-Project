# Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Engines
# Github source: https://github.com/cnzzx/VSA-dev
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on LLaVA and MindSearch code bases
# https://github.com/haotian-liu/LLaVA
# https://github.com/IDEA-Research/GroundingDINO
# https://github.com/InternLM/MindSearch
# --------------------------------------------------------

import os
import copy

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from .vsa_prompt import COCO_CLASSES, get_caption_prompt, get_correlate_prompt, get_qa_prompt

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from datetime import datetime
from lagent.actions import ActionExecutor, BingBrowser
from lagent.llms import INTERNLM2_META, LMDeployServer
from lagent.schema import AgentReturn, AgentStatusCode
from lagent.schema import AgentStatusCode
from .search_agent.mindsearch_agent import (
    MindSearchAgent, SimpleSearchAgent, MindSearchProtocol
)
from .search_agent.mindsearch_prompt import (
    FINAL_RESPONSE_CN, FINAL_RESPONSE_EN, GRAPH_PROMPT_CN, GRAPH_PROMPT_EN,
    searcher_context_template_cn, searcher_context_template_en,
    searcher_input_template_cn, searcher_input_template_en,
    searcher_system_prompt_cn, searcher_system_prompt_en
)

from typing import List, Union

SEARCH_MODEL_NAMES = {
    'internlm2_5-7b-chat': 'internlm2'
}


def render_bboxes(in_image: Image.Image, bboxes: np.ndarray, labels: List[str]):
    out_image = copy.deepcopy(in_image)
    draw = ImageDraw.Draw(out_image)
    font = ImageFont.truetype(font = 'assets/Arial.ttf', size = min(in_image.width, in_image.height) // 30)
    line_width = min(in_image.width, in_image.height) // 100
    for i in range(bboxes.shape[0]):
        draw.rectangle((bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]), outline=(0, 255, 0), width=line_width)
        bbox = draw.textbbox((bboxes[i, 0], bboxes[i, 1]), '[Area {}] '.format(i), font=font)
        draw.rectangle(bbox, fill='white')
        draw.text((bboxes[i, 0], bboxes[i, 1]), '[Area {}] '.format(i), fill='black', font=font)
    if bboxes.shape[0] == 0:
        draw.rectangle((0, 0, in_image.width, in_image.height), outline=(0, 255, 0), width=line_width)
        bbox = draw.textbbox((0, 0), '[Area {}] '.format(0), font=font)
        draw.rectangle(bbox, fill='white')
        draw.text((0, 0), '[Area {}] '.format(0), fill='black', font=font)
    return out_image


class VisualGrounder:
    def __init__(
        self,
        model_path: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:1",
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device)
        self.device = device
        self.default_classes = COCO_CLASSES
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        print('Visual Grounder initialized.')
            
    def __call__(
        self,
        in_image: Image.Image,
        classes: Union[List[str], None] = None,
    ):
        # Save image.
        in_image.save('temp/in_image.jpg')
        
        # Preparation.
        if classes is None:
            classes = self.default_classes
        
        text = ". ".join(classes)
        inputs = self.processor(images=in_image, text=text, return_tensors="pt").to(self.device)

        # Grounding.
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Postprocess
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold = self.box_threshold,
            text_threshold = self.text_threshold,
            target_sizes=[in_image.size[::-1]]
        )
        bboxes = results[0]['boxes'].cpu().numpy()
        labels = results[0]['labels']

        print(results)
        
        # Visualization.
        out_image = render_bboxes(in_image, bboxes, labels)
        out_image.save('temp/ground_bbox.jpg')
        
        return bboxes, labels, out_image


class VLM:
    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.6-vicuna-7b",
        device: str = "cuda:2",
        load_8bit: bool = False,
        load_4bit: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 2000,
    ):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, load_8bit, load_4bit, device=device
        )
        self.device = device

        if "llama-2" in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
        
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        print('VLM initialized.')

    def __call__(
        self,
        image: Image.Image,
        text: str,
    ):
        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
            else:
                text = DEFAULT_IMAGE_TOKEN + '\n' + text
            image = None
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images = image_tensor,
                image_sizes = [image_size],
                do_sample = True if self.temperature > 0 else False,
                temperature = self.temperature,
                max_new_tokens = self.max_new_tokens,
                streamer = None,
                use_cache = True)
        outputs = self.tokenizer.decode(output_ids[0]).strip()
        outputs = outputs.replace('<s>', '').replace('</s>', '').replace('"', "'")

        return outputs


class WebSearcher:
    def __init__(
        self,
        model_path: str = 'InternLM/internlm2_5-7b-chat',
        lang: str = 'en',
        top_p: float = 0.8,
        top_k: int = 1,
        temperature: float = 0,
        max_new_tokens: int = 8192,
        repetition_penalty: float = 1.02,
        max_turn: int = 10,
    ):
        model_name = get_model_name_from_path(model_path)
        if model_name in SEARCH_MODEL_NAMES:
            model_name = SEARCH_MODEL_NAMES[model_name]
        else:
            raise Exception('Unsupported model for web searcher.')
        
        self.lang = lang
        llm = LMDeployServer(
            path = model_path,
            model_name = model_name,
            server_name='127.0.0.1',
            meta_template = INTERNLM2_META,
            top_p = top_p,
            top_k = top_k,
            temperature = temperature,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            stop_words = ['<|im_end|>']
        )
        self.agent = MindSearchAgent(
            llm = llm,
            protocol = MindSearchProtocol(
                meta_prompt = datetime.now().strftime('The current date is %Y-%m-%d.'),
                interpreter_prompt = GRAPH_PROMPT_CN if lang == 'cn' else GRAPH_PROMPT_EN,
                response_prompt = FINAL_RESPONSE_CN if lang == 'cn' else FINAL_RESPONSE_EN
            ),
            searcher_cfg=dict(
                llm = llm,
                plugin_executor = ActionExecutor(
                    BingBrowser(searcher_type='DuckDuckGoSearch', topk=6)
                ),
                protocol = MindSearchProtocol(
                    meta_prompt=datetime.now().strftime('The current date is %Y-%m-%d.'),
                    plugin_prompt=searcher_system_prompt_cn if lang == 'cn' else searcher_system_prompt_en,
                ),
                template = dict(
                    input=searcher_input_template_cn if lang == 'cn' else searcher_input_template_en,
                    context=searcher_context_template_cn if lang == 'cn' else searcher_context_template_en)
                ),
            max_turn = max_turn
        )
        
        print('Web Searcher initialized.', flush=True)
    
    def __call__(
        self,
        queries: List[str]
    ):
        results = []
        for qid, query in enumerate(queries):
            result = None
            for agent_return in self.agent.stream_chat(query):
                if isinstance(agent_return, AgentReturn):
                    if agent_return.state == AgentStatusCode.END:
                        result = agent_return.response
            assert result is not None
            with open('temp/search_result_{}.txt'.format(qid), 'w', encoding='utf-8') as wf:
                wf.write(result)
            results.append(result)
        return results


class VisionSearchAssistant:
    """
    Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Engines
    
    This class implements all variants of Vision Search Assistant:
    
    * search_model: Vision Search Assistant use this model for dealing with the search process,
    it corresponds to the $\mathcal{F}_{llm}(cdot)$ in the paper. You can choose the model
    according to your preference.
    * ground_model: The vision foundation model used in the open-vocab detection process,
    it's relevant to the specific contents of the classes in the image.
    * vlm_model: The main vision-language model we used in our paper is LLaVA-1.6 baseline,
    It can be further improved by using advanced models. And it corresponds to 
    the $\mathcal{F}_{vlm}(cdot)$ in the paper.

    """
    def __init__(
        self,
        search_model: str = "internlm/internlm2_5-7b-chat",
        ground_model: str = "IDEA-Research/grounding-dino-base",
        ground_device: str = "cuda",
        vlm_model: str = "microsoft/llava-med-v1.5-mistral-7b",
        vlm_device: str = "cuda",
        vlm_load_4bit: bool = False,
        vlm_load_8bit: bool = False,
    ):
        self.searcher = WebSearcher(
            model_path = search_model
        )
        self.grounder = VisualGrounder(
            model_path = ground_model,
            device = ground_device,
        )
        self.vlm = VLM(
            model_path = vlm_model,
            device = vlm_device,
            load_4bit = vlm_load_4bit,
            load_8bit = vlm_load_8bit
        )
        self.use_correlate = True
        
        print('Vision Search Assistant initialized.', flush=True)
    
    def __call__(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text: str,
        ground_classes: Union[List[str], None] = None
    ):
        # Create and clear the temporary directory.
        if not os.access('temp', os.F_OK):
            os.makedirs('temp')
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
        
        with open('temp/text.txt', 'w', encoding='utf-8') as wf:
            wf.write(text)

        # Load Image
        if isinstance(image, str):
            in_image = Image.open(image)
        elif isinstance(image, Image.Image):
            in_image = image
        elif isinstance(image, np.ndarray):
            in_image = Image.fromarray(image.astype(np.uint8))
        else:
            raise Exception('Unsupported input image format.')

        # Visual Grounding
        bboxes, labels, out_image = self.grounder(in_image, classes = ground_classes)

        det_images = []
        for bid, bbox in enumerate(bboxes):
            crop_box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            det_image = in_image.crop(crop_box)
            det_image.save('temp/debug_bbox_image_{}.jpg'.format(bid))
            det_images.append(det_image)
        
        if len(det_images) == 0:  # No object detected, use the full image.
            det_images.append(in_image)
            labels.append('image')
        
        # Visual Captioning
        captions = []
        for det_image, label in zip(det_images, labels):
            inp = get_caption_prompt(label, text)
            caption = self.vlm(det_image, inp)
            captions.append(caption)
        
        for cid, caption in enumerate(captions):
            with open('temp/caption_{}.txt'.format(cid), 'w', encoding='utf-8') as wf:
                wf.write(caption)
        
        # Visual Correlation
        if len(captions) >= 2 and self.use_correlate:
            queries = []
            for mid, det_image in enumerate(det_images):
                caption = captions[mid]
                other_captions = []
                for cid in range(len(captions)):
                    if cid == mid:
                        continue
                    other_captions.append(captions[cid])
                inp = get_correlate_prompt(caption, other_captions)
                query = self.vlm(det_image, inp)
                queries.append(query)
        else:
            queries = captions
        
        for qid, query in enumerate(queries):
            with open('temp/query_{}.txt'.format(qid), 'w', encoding='utf-8') as wf:
                wf.write(query)
        
        queries = [text + " " + query for query in queries]

        # Web Searching
        contexts = self.searcher(queries)

        # QA
        TOKEN_LIMIT = 3500
        max_length_per_context = TOKEN_LIMIT // len(contexts)
        for cid, context in enumerate(contexts):
            contexts[cid] = (queries[cid] + context)[:max_length_per_context]
        
        inp = get_qa_prompt(text, contexts)
        answer = self.vlm(in_image, inp)

        with open('temp/answer.txt', 'w', encoding='utf-8') as wf:
            wf.write(answer)
        print(answer)
        
        return answer
    
    def app_run(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text: str,
        ground_classes: List[str] = COCO_CLASSES
    ):
        # Create and clear the temporary directory.
        if not os.access('temp', os.F_OK):
            os.makedirs('temp')
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
        
        with open('temp/text.txt', 'w', encoding='utf-8') as wf:
            wf.write(text)

        # Load Image
        if isinstance(image, str):
            in_image = Image.open(image)
        elif isinstance(image, Image.Image):
            in_image = image
        elif isinstance(image, np.ndarray):
            in_image = Image.fromarray(image.astype(np.uint8))
        else:
            raise Exception('Unsupported input image format.')

        # Visual Grounding
        bboxes, labels, out_image = self.grounder(in_image, classes = ground_classes)
        yield out_image, 'ground'

        det_images = []
        for bid, bbox in enumerate(bboxes):
            crop_box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            det_image = in_image.crop(crop_box)
            det_image.save('temp/debug_bbox_image_{}.jpg'.format(bid))
            det_images.append(det_image)
        
        if len(det_images) == 0:  # No object detected, use the full image.
            det_images.append(in_image)
            labels.append('image')
        
        # Visual Captioning
        captions = []
        for det_image, label in zip(det_images, labels):
            inp = get_caption_prompt(label, text)
            caption = self.vlm(det_image, inp)
            captions.append(caption)
        
        for cid, caption in enumerate(captions):
            with open('temp/caption_{}.txt'.format(cid), 'w', encoding='utf-8') as wf:
                wf.write(caption)
        
        # Visual Correlation
        if len(captions) >= 2 and self.use_correlate:
            queries = []
            for mid, det_image in enumerate(det_images):
                caption = captions[mid]
                other_captions = []
                for cid in range(len(captions)):
                    if cid == mid:
                        continue
                    other_captions.append(captions[cid])
                inp = get_correlate_prompt(caption, other_captions)
                query = self.vlm(det_image, inp)
                queries.append(query)
        else:
            queries = captions
        
        for qid, query in enumerate(queries):
            with open('temp/query_{}.txt'.format(qid), 'w', encoding='utf-8') as wf:
                wf.write(query)
        yield queries, 'query'

        queries = [text + " " + query for query in queries]

        # Web Searching
        contexts = self.searcher(queries)
        yield contexts, 'search'

        # QA
        TOKEN_LIMIT = 3500
        max_length_per_context = TOKEN_LIMIT // len(contexts)
        for cid, context in enumerate(contexts):
            contexts[cid] = (queries[cid] + context)[:max_length_per_context]
        
        inp = get_qa_prompt(text, contexts)
        answer = self.vlm(in_image, inp)

        with open('temp/answer.txt', 'w', encoding='utf-8') as wf:
            wf.write(answer)
        print(answer)
        
        yield answer, 'answer'