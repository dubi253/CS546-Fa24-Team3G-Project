import gradio as gr
# from models.vsa_model import VisionSearchAssistant
from models.vsa_prompt import COCO_CLASSES

# Create a mock VSA
class VisionSearchAssistant:
    def app_run(self, image, text):
        # Simulate output
        yield ["Mock Query 1", "Mock Query 2"], "query"
        yield ["Mock Search Context 1", "Mock Search Context 2"], "search"
        yield "Mock Answer", "answer"

SAMPLES = {
    "images/iclr.jpg": ("What prize did this paper win in 2024?", ", ".join(COCO_CLASSES)),
    "images/tesla.jpg": ("What's the income of this company?", "car"),
    "images/xiaomi.jpg": ("Provide information about the new products of this brand.", ", ".join(COCO_CLASSES)),
    "images/leshi.jpg": ("Provide information about new products of this brand of potato chips in 2024.", ", ".join(COCO_CLASSES)),
}
SAMPLE_IMAGES = list(SAMPLES.keys())
SAMPLE_TEXTS = [e[0] for e in SAMPLES.values()]
SAMPLE_CLASSES = [e[1] for e in SAMPLES.values()]


def process_inputs(image, text):

    query_output, search_output, answer_output = None, None, None
    for output, output_type in vsa.app_run(image, text):
        if output_type == 'query':
            query_output = ''
            for qid, query in enumerate(output):
                query_output += '[Area {}] '.format(qid) + query + '\n'
            search_output = search_output or ''
            answer_output = answer_output or ''
            yield query_output, search_output, answer_output
        elif output_type == 'search':
            search_output = ''
            for cid, context in enumerate(output):
                search_output += '[Context {}] '.format(cid) + context + '\n'
            query_output = query_output or ''
            answer_output = answer_output or ''
            yield query_output, search_output, answer_output
        elif output_type == 'answer':
            answer_output = 'answer output: ' + output
            query_output = query_output or ''
            search_output = search_output or ''
            yield query_output, search_output, answer_output


def select_sample_inputs(sample):
    if sample == 'none':
        return None, None, None
    image = sample
    text, classes = SAMPLES[sample]
    return image, text, classes

def confirm_sample_inputs(image, text, classes):
    return image, text, classes

if __name__ == '__main__':
    # Create a Blocks interface
    with gr.Blocks() as app:
        with gr.Tab("Run"):
            with gr.Row():
                with gr.Column():
                    # with gr.Row():
                    image_input = gr.Image(label="Input Image", height=300, width=300)
                    prompt_input = gr.Textbox(label="Input Text Prompt", lines=1, max_lines=1)
                    submit_button = gr.Button("Submit")
                    answer_output = gr.Textbox(label="Answer Output", lines=4, max_lines=4, interactive=False)
                with gr.Column():
                    query_output = gr.Textbox(label='Query Output', lines=14, max_lines=14, interactive=False)
                    search_output = gr.Textbox(label="Search Output", lines=14, max_lines=14, interactive=False)
        with gr.Tab("Samples"):
            sample_input = gr.Dropdown(label="Select One Sample", choices=SAMPLE_IMAGES)
            with gr.Row():
                sample_image = gr.Image(label="Sample Input Image", height=300, interactive=False, value=SAMPLE_IMAGES[0])
                with gr.Column():
                    sample_text = gr.Textbox(label="Sample Input Text", lines=4, max_lines=4, interactive=False, value=SAMPLE_TEXTS[0])
                    sample_classes = gr.Textbox(label="Sample Input Classes", lines=4, max_lines=4, interactive=False, value=SAMPLE_CLASSES[0])
            sample_button = gr.Button("Select This Sample")
            
        
        # Processing action
        submit_button.click(
            fn=process_inputs,
            inputs=[image_input, prompt_input],
            outputs=[query_output, search_output, answer_output],
            show_progress=True,
        )
        sample_input.change(
            fn=select_sample_inputs,
            inputs=[sample_input],
            outputs=[sample_image, sample_text, sample_classes]
        )
        sample_button.click(
            fn=confirm_sample_inputs,
            inputs=[sample_image, sample_text, sample_classes],
            outputs=[image_input, prompt_input],
        )

    print("init")

    vsa = VisionSearchAssistant()
    # Launch the app
    print('The app is launched.')
    
    app.launch(server_port=9999,share=True)
    print('The app is launched successfully.')