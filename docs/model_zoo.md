# Model Zoo
In the following, we list the models acquired in Vision Search Assistant. It's expected that those models will be automatically downloaded to your device when you run the code for the first time. If you encounter any problems, you can manually download them.
## Grounding Model
We use [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) as the gounding model. 

| Model | Box AP on COCO | Weights |
| :-: | :-: | :-: |
| GroundingDINO-Tiny | 48.4 | [Huggingface](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| GroundingDINO-Base | 56.7 | [Huggingface](https://huggingface.co/IDEA-Research/grounding-dino-base) |

## Vision Language Model
We use [LLaVA-v1.6](https://github.com/haotian-liu/LLaVA) as the core Vision Language Model.

| Version | LLM | LLaVA-Bench-Wild | Weights |
| :-: | :-: | :-: | :-: |
| LLaVA-1.6 | Vicuna-7B | 81.6 | [Huggingface](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
| LLaVA-1.6 | Vicuna-13B | 87.3 | [Huggingface](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) |
| LLaVA-1.6 | Mistral-7B | 83.2 | [Huggingface](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b) |
| LLaVA-1.6 | Hermes-Yi-34B | 89.6 | [Huggingface](https://huggingface.co/liuhaotian/llava-v1.6-34b) |

## Searching Model
We use [InternLM](https://github.com/InternLM/InternLM) as the searching model.

| Model | CMMLU | Weights |
| :-: | :-: | :-: |
| InternLM2.5-1.8B-Chat | - | [Huggingface](https://huggingface.co/internlm/internlm2_5-1_8b-chat) |
| InternLM2.5-7B-Chat | 78.0 | [Huggingface](https://huggingface.co/internlm/internlm2_5-7b-chat) |
| InternLM2.5-20B-Chat | - | [Huggingface](https://huggingface.co/internlm/internlm2_5-20b-chat) |