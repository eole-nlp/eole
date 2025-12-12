import gradio as gr
import requests
import re
import concurrent.futures
from urllib.parse import urlparse, urlunparse

# Languages
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


# Helper functions
def extract_chunks(text):
    parts = re.split(r"(\n+)", text)
    sentences, breaks = [], []
    for part in parts:
        if part.strip() == "":
            breaks.append(part)
        else:
            sentences.append(part.strip())
    return [s for s in sentences if s], breaks


def rebuild_text(translated_sentences, breaks):
    result = []
    s_idx = b_idx = 0
    while s_idx < len(translated_sentences) or b_idx < len(breaks):
        if s_idx < len(translated_sentences):
            result.append(translated_sentences[s_idx])
            s_idx += 1
        if b_idx < len(breaks):
            result.append(breaks[b_idx])
            b_idx += 1
    return "".join(result)


def get_base_url(api_url):
    parsed = urlparse(api_url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def get_models(api_url):
    base_url = get_base_url(api_url)
    try:
        response = requests.get(f"{base_url}/models")
        data = response.json()
        models = [m.get("id", str(m)) for m in data.get("models", [])]
        return models if models else ["No models found"]
    except Exception as e:
        return [f"Error fetching models: {e}"]


def update_models(api_url):
    models = get_models(api_url)
    return {"choices": models, "value": models[0]}


def translate_text(api_url, model_id, prompt_template, source_lang, target_lang, source_text):
    paragraphs, breaks = extract_chunks(source_text)
    translated_sentences = []

    payloads = []
    for para in paragraphs:
        user_prompt = prompt_template.replace("{target_lang}", target_lang) + "\n\n" + para
        payloads.append(
            {
                "model": model_id,
                "messages": [{"role": "system", "content": ""}, {"role": "user", "content": user_prompt}],
            }
        )

    def send_request(payload):
        try:
            response = requests.post(api_url, json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paragraphs)) as executor:
        futures = [executor.submit(send_request, p) for p in payloads]
        results = [f.result() for f in futures]

    for data in results:
        if "choices" in data:
            translated_sentences.append(data["choices"][0]["message"]["content"])
        else:
            predictions = data.get("predictions", [])
            if predictions and predictions[0]:
                translated_sentences.append(predictions[0][0])
            else:
                translated_sentences.append("Error: No translation returned by the server.")

    return rebuild_text(translated_sentences, breaks)


custom_css = """
/* General layout */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f6fa;
}

/* Inputs and textboxes */
input, select, textarea {
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 8px;
    font-size: 14px;
    transition: all 0.2s;
}

input:focus, select:focus, textarea:focus {
    border-color: #4285F4;
    box-shadow: 0 0 5px rgba(66,133,244,0.5);
    outline: none;
}

/* Boxes */
.gr-box {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
}

/* Buttons */
button {
    width: 220px;
    background-color: #4285F4;
    color: white;
    font-size: 16px;
    border: none;
    border-radius: 6px;
    padding: 10px;
    cursor: pointer;
    margin-top: 10px;
    transition: all 0.2s;
}
button:hover {
    background-color: #357AE8;
    transform: translateY(-2px);
}

/* Scrollable settings column */
#settings-col {
    overflow-y: auto;
    max-height: 600px;
    padding: 10px;
}
"""
# Gradio Interface
with gr.Blocks(title="Eole Multilingual Translator", css=custom_css) as iface:
    gr.Markdown("<h1 style='text-align: center; font-family: Arial;'>Eole Multilingual Translator</h1>")

    with gr.Row(equal_height=True):
        # Left Column: Source language + text
        with gr.Column(scale=4):
            source_lang = gr.Dropdown(languages, label="Source Language", value="English")
            source_text = gr.Textbox(placeholder="Enter text here...", lines=15, label="Source Text")

        # Right Column: Target language + translated text
        with gr.Column(scale=4):
            target_lang = gr.Dropdown(languages, label="Target Language", value="French")
            translated_text = gr.Textbox(
                placeholder="Translation will appear here...", lines=15, label="Translated Text", interactive=False
            )

        # Settings Column (scrollable)
        with gr.Column(scale=2, elem_id="settings-col") as settings_col:
            api_url = gr.Dropdown(
                label="API URL",
                choices=["http://127.0.0.1:5000/infer", "http://127.0.0.1:5000/v1/chat/completions"],
                value="http://127.0.0.1:5000/infer",
                interactive=True,
            )
            model_id = gr.Dropdown(choices=get_models(api_url.value), label="Model", value=None)
            prompt_template = gr.Textbox(
                label="Prompt",
                value="Translate the following text into {target_lang}, without additional explanation.",
                lines=4,
            )

    # Update models when API URL changes
    api_url.change(update_models, inputs=[api_url], outputs=[model_id])

    # Translate button
    translate_button = gr.Button("Translate", elem_id="button-container")
    translate_button.click(
        translate_text,
        inputs=[api_url, model_id, prompt_template, source_lang, target_lang, source_text],
        outputs=[translated_text],
    )

iface.launch(share=True)
