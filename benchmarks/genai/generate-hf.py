# flake8: noqa
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import time

model_id = "google/gemma-3-1b-it"

model = (
    Gemma3ForCausalLM.from_pretrained(model_id, _attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    .cuda()
    .eval()
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

batch_messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate a 200 word text talking about George Orwell."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the meaning of life?"},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Who is Elon Musk?"},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is beyond the milky way?"},
            ],
        }
    ],
]


print("################ BATCH SIZE 4 ################################")
inputs = tokenizer.apply_chat_template(
    batch_messages,
    add_generation_prompt=True,
    tokenize=True,
    padding=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=2048, top_p=1.0, top_k=50, do_sample=False)
    input_ids = inputs.input_ids
prompt_len = input_ids.size(1)

generated_ids = generated_ids[:, prompt_len:]
nbtok = generated_ids.ne(0).sum().item()

output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(output_texts)
print(f"\nNumber of generated tokens {nbtok}, throughput {nbtok / (time.time() - start_time):.0f} tok/sec\n")

print("################ BATCH SIZE 1 ################################\n")
for i in range(4):
    inputs = tokenizer.apply_chat_template(
        batch_messages[i],
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048, top_p=1.0, top_k=50, do_sample=False)
        input_ids = inputs.input_ids
    prompt_len = input_ids.size(1)

    generated_ids = generated_ids[:, prompt_len:]
    nbtok = generated_ids.ne(0).sum().item()

    output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # print(output_texts)
    print(f"input {i}: number of generated tokens {nbtok}, throughput {nbtok / (time.time() - start_time):.0f} tok/sec")
