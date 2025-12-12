import gradio as gr
import requests
import re

# List of 35 supported languages
languages = [
    "Bulgarian",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hungarian",
    "Irish",
    "Italian",
    "Latvian",
    "Lithuanian",
    "Maltese",
    "Polish",
    "Portuguese",
    "Romanian",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swedish",
    "Arabic",
    "Catalan",
    "Chinese",
    "Galician",
    "Hindi",
    "Japanese",
    "Korean",
    "Norwegian",
    "Russian",
    "Turkish",
    "Ukrainian",
]


def extract_chunks(text):
    # Split into: (sentence/text) OR (one or more newlines)
    parts = re.split(r"(\n+)", text)

    # Join consecutive non-newline parts and track newline blocks separately
    sentences = []
    breaks = []

    for part in parts:
        if part.strip() == "":
            # It's a newline block
            breaks.append(part)
        else:
            # It's text (a sentence)
            sentences.append(part.strip())

    # Remove empty sentences
    sentences_no_empty = [s for s in sentences if s]

    return sentences_no_empty, breaks


def rebuild_text(translated_sentences, breaks):
    result = []
    s_idx = 0
    b_idx = 0

    # Alternate between sentence and break in the original order
    # starting with a sentence if the original text did
    while s_idx < len(translated_sentences) or b_idx < len(breaks):
        if s_idx < len(translated_sentences):
            result.append(translated_sentences[s_idx])
            s_idx += 1
        if b_idx < len(breaks):
            result.append(breaks[b_idx])
            b_idx += 1

    return "".join(result)


# Function to call Eole API
def translate_text(source_lang, target_lang, source_text):
    # Define the API URL for inference
    api_url = "http://127.0.0.1:5000/infer"

    # Replace this with the correct model ID from the server's configuration
    model_id = "Hunyuan-MT-7B-eole"  # Adjust this to match the server configuration

    # 1. Extract sentences and newline blocks
    paragraphs, breaks = extract_chunks(source_text)
    inputs = []
    for para in paragraphs:
        inputs.append(
            f"<|startoftext|>Translate the following text into {target_lang}"
            f", without additional explanation.｟newline｠｟newline｠"
            f"{para}<|extra_0|>"
        )
    # Construct the payload for the API
    payload = {
        "model": model_id,
        "inputs": inputs,
        # Optional: Add decoding settings (based on `DecodingConfig`)
        "temperature": 1,
        "max_length": 512,
    }

    try:
        # Send the POST request
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the response
        result = response.json()
        predictions = result.get("predictions", [])

        # Extract the first translation, if available
        if predictions and predictions[0]:
            translated_text = [
                predictions[i][0] for i in range(len(predictions))
            ]  # First prediction of the first input
        else:
            translated_text = ["Error: No translation returned by the server."]
    except Exception as e:
        translated_text = [f"Error: {str(e)}"]

    return rebuild_text(translated_text, breaks)


# Gradio Interface
def gradio_translation_interface(source_lang, target_lang, source_text):
    return translate_text(source_lang, target_lang, source_text)


with gr.Blocks(title="Hunyuan-Eole Multilingual Translator") as iface:
    # Add Title
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>Hunyuan-Eole Multilingual Translator</h1>")

    # Add side-by-side inputs
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            source_lang = gr.Dropdown(languages, label="Source Language", value="English")
            source_text = gr.Textbox(placeholder="Enter text here...", lines=10, label="Source Text")
        with gr.Column(scale=1):
            target_lang = gr.Dropdown(languages, label="Target Language", value="French")
            translated_text = gr.Textbox(
                placeholder="Translation will appear here...",
                lines=10,
                label="Translated Text",
                interactive=False,
            )

    # Add a narrower Translate button
    with gr.Row():
        with gr.Column(scale=1, min_width=200, elem_id="button-container"):
            translate_button = gr.Button("Translate")

    # Define interactions
    translate_button.click(
        gradio_translation_interface,
        inputs=[source_lang, target_lang, source_text],
        outputs=[translated_text],
    )

# Add custom CSS for styling
iface.css = """
#button-container {
    display: flex;
    justify-content: center;
}
button {
    width: 200px;
    background-color: #4285F4;
    color: white;
    font-size: 16px;
    border: none;
    border-radius: 4px;
    padding: 10px;
    cursor: pointer;
}
button:hover {
    background-color: #357AE8;
}
"""

iface.launch(share=True)
