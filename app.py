import gradio as gr
from transformers import pipeline
import fitz

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text, model, max_length=1024):
    # Split the input text into smaller chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    # Summarize each chunk separately (You can use list comprehension)
    summaries = []
    for chunk in chunks:
        summary = model(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    # Concatenate the summaries into a single string
    summary = ' '.join(summaries)

    return summary

# Function to read PDF and summarize
def summarize_pdf(pdf_file, model):
    with fitz.open(pdf_file.name) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return summarize_text(text, model)

def summarize(input_type, input_text, uploaded_file):
    try:
        if input_type == "Text":
            summary = summarize_text(input_text, summarizer)
        else:
            summary = summarize_pdf(uploaded_file, summarizer)
        return summary
    except Exception as e:
        return "There was a problem summarizing the text. Please try again later."

# Define the footer
footer = """
<div style="text-align: center; margin-top: 20px;">
    <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
    <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a>
    <br>
    Made with 💖 by Pejman Ebrahimi
</div>
"""

# Define the inputs and outputs
inputs = [
    gr.Radio(["Text", "PDF"], label="Input Type"),
    gr.Textbox(lines=10, label="Enter Text to Summarize", visible=False),
    gr.File(label="Upload PDF file", visible=False)
]
outputs = [
    gr.Textbox(label="Summary"),
    gr.HTML(footer)
]

# Define the submit button
submit_btn = gr.Button("Submit")

# Define the Gradio interface
with gr.Blocks(theme='gradio/soft') as app:
    gr.Markdown("# Text and PDF Summarization App")
    gr.Markdown("Note: This model can handle a maximum of 1024 tokens. A token is a unit of text that the model can process at a time. When summarizing text, the input text is split into smaller chunks of up to 1024 tokens each, and each chunk is summarized separately. The summaries are then concatenated into a single summary.")
    with gr.Row():
        input_type = gr.Radio(["Text", "PDF"], label="Input Type")
    with gr.Row():
        input_text = gr.Textbox(lines=10, label="Enter Text to Summarize", visible=False)
        uploaded_file = gr.File(label="Upload PDF file", visible=False)
    with gr.Row():
        submit_btn = gr.Button("Submit")
    with gr.Row():
        summary = gr.Textbox(label="Summary")
    with gr.Row():
        footer = gr.HTML(footer)

    # Define the change event handler for the input type radio buttons
    def input_type_change(input_type):
        if input_type == "Text":
            return {input_text: gr.Textbox(visible=True), uploaded_file: gr.File(visible=False)}
        else:
            return {input_text: gr.Textbox(visible=False), uploaded_file: gr.File(visible=True)}
    input_type.change(fn=input_type_change, inputs=[input_type], outputs=[input_text, uploaded_file])

    # Define the click event handler for the submit button
    submit_btn.click(fn=summarize, inputs=[input_type, input_text, uploaded_file], outputs=[summary])

# Launch the Gradio interface
if __name__ == "__main__":
    app.launch()
