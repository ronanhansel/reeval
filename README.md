env setup
```
conda create -n certify python=3.10 -y
conda activate certify
pip install requirements.txt
```

```
export CUDA_VISIBLE_DEVICES=2
vllm serve meta-llama/Meta-Llama-3-8B-Instruct
```


export CUDA_VISIBLE_DEVICES=2
vllm serve meta-llama/Meta-Llama-3-8B-Instruct