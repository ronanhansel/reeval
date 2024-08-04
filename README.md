environment setup
```
conda create -n certify python=3.10 -y
conda activate certify
pip install requirements.txt
```

create a .env file
```
OPENAI_KEY = '...'
```

# vllm query for more testtakers
choose the GPU and start the vllm server on a chosen port
```
export CUDA_VISIBLE_DEVICES=3
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 4324
```

run the infer code
make sure --url use same port
start and end are the serial number of different system prompt, ranges from 1 to 1000
start 1, end 50 will run the first 50 system prompt
```
cd vllm_query
python vllm_openai_infer.py --url http://localhost:4324/v1 --start 1 --end 1
```
