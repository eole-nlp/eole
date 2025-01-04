import gradio as gr
import requests

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


# Function to call Eole API
def translate_text(source_lang, target_lang, source_text):
    # Define the API URL for inference
    api_url = "http://localhost:5000/infer"

    # Replace this with the correct model ID from the server's configuration
    model_id = "EuroLLM-9B-Instruct"  # Adjust this to match the server configuration

    # Construct the payload for the API
    payload = {
        "model": model_id,
        "inputs": (
            f"<|im_start|>system｟newline｠<|im_end|>｟newline｠<|im_start|>user｟newline｠"
            f"Translate the following text from {source_lang} into {target_lang}.｟newline｠"
            f"{source_lang}: {source_text}｟newline｠"
            f"{target_lang}: <|im_end|>｟newline｠<|im_start|>assistant｟newline｠"
        ),
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
            translated_text = predictions[0][0]  # First prediction of the first input
        else:
            translated_text = "Error: No translation returned by the server."
    except Exception as e:
        translated_text = f"Error: {str(e)}"

    return translated_text


# Gradio Interface
def gradio_translation_interface(source_lang, target_lang, source_text):
    return translate_text(source_lang, target_lang, source_text)


with gr.Blocks(title="EuroLLM-Eole Multilingual Translator") as iface:
    # Add Title
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>EuroLLM-Eole Multilingual Translator</h1>")

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
