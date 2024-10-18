# OpenAI GPT2

https://huggingface.co/openai-community/gpt2

## Convert

```bash
eole convert HF --model_dir openai-community/gpt2 --output $EOLE_MODEL_DIR/openai_gpt2 --token $HF_TOKEN
```

## Infer

```bash
echo -e "The European Union was created in" > lm_input.txt
eole predict -c inference.yaml
```

## HellaSwag benchmark

```bash
eole tools eval_hellaswag -c inference.yaml
```

Eole results, marginally different due to slight implementation differences (nn.Linear vs nn.Conv1D):

```
...
10040 acc: 0.2865 acc_norm: 2959/10040=0.2947
10041 acc: 0.2864 acc_norm: 2959/10041=0.2947
10042 acc: 0.2864 acc_norm: 2960/10042=0.2948
```

Comparable results from [llm.c script](https://github.com/karpathy/llm.c/blob/master/dev/data/hellaswag.py) using official huggingface implementation:

```
...
10040 acc: 0.2862 acc_norm: 2966/10040=0.2954
10041 acc: 0.2861 acc_norm: 2966/10041=0.2954
10042 acc: 0.2861 acc_norm: 2967/10042=0.2955
```