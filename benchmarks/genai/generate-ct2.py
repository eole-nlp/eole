# flake8: noqa
from transformers import AutoTokenizer
import ctranslate2
import time

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
gen = ctranslate2.Generator("$EOLE_MODEL_DIR/gemma-3-1b-it-ct2", device="cuda", device_index=0)

prompt = (
    "<start_of_turn>user\nGenerate a 200 word text talking about George Orwell.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is the meaning of life?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWho is Elon Musk?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is beyond the milky way?<end_of_turn>\n<start_of_turn>model\n",
)
tokens = [tok.convert_ids_to_tokens(tok.encode(p)) for p in prompt]

print("################ BATCH SIZE 1 ################################\n")
for i in range(4):
    start_time = time.time()
    results = gen.generate_batch(
        [tokens[i]], max_length=2048, max_batch_size=1, sampling_temperature=1.0, include_prompt_in_result=False
    )

    # print([tok.convert_tokens_to_string(res.sequences[0]) for res in results])
    nbtok = sum([len(res.sequences[0]) for res in results])
    print(f"Number of generated tokens {nbtok}, throughput {nbtok / (time.time() - start_time):.0f} tok/sec")
