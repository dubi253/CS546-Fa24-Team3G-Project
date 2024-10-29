import argparse
from PIL import Image
from models.vsa_model import VisionSearchAssistant


def process_args(args):
    if args.vlm_load_4bit and args.vlm_load_8bit:
        raise Exception('Cannot set both "vlm_load_4bit" and "vlm_load_8bit" to true.')
    if args.ground_classes is not None:
        args.ground_classes = args.ground_classes.split(', ')
    return args


def main(args):
    args = process_args(args)
    vsa = VisionSearchAssistant(
        search_model = args.search_model,
        ground_model = args.ground_model,
        ground_device = args.ground_device,
        vlm_model = args.vlm_model,
        vlm_device = args.vlm_device,
        vlm_load_4bit = args.vlm_load_4bit,
        vlm_load_8bit = args.vlm_load_8bit
    )
    while True:
        image = input('[image path] (enter "exit" to quit): ')
        if image == 'exit':
            break
        try:
            _ = Image.open(image)
        except:
            print('Image not found.')
            continue
        text = input('[question] (enter "exit" to quit): ')
        if text == 'exit':
            break
        vsa(image, text, ground_classes = args.ground_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-model", type=str, default="internlm/internlm2_5-7b-chat")
    parser.add_argument("--ground-model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--vlm-model", type=str, default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--ground-device", type=str, default="cuda:1")
    parser.add_argument("--ground-classes", type=str, default=None)
    parser.add_argument("--vlm-device", type=str, default="cuda:2")
    parser.add_argument("--vlm-load-4bit", action='store_true')
    parser.add_argument("--vlm-load-8bit", action='store_true')
    args = parser.parse_args()
    main(args)