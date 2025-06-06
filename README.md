# Reliable and Efficient Amortized Model-based Evaluation


This repository implements the paper Reliable and Efficient Amortized Model-based Evaluation.


To set up the Python environment:
```bash
conda create -n reeval python=3.10 -y
conda activate reeval
pip install -r requirements.txt
```


To download the `data/` folder:
```bash
python download.py --folder data
```


To set up the R environment:
```bash
conda env create -f cat.yml
conda activate cat
```


# Calibration and Generalization analysis
`cd calibration` and run `calibration.ipynb`.


# Adaptive Testing


While the result in the paper is from a synthetic adaptive testing experiment implemented in R, we also integrate the adaptive testing into HELM, implemented with Python. See [HELM tutorial](https://crfm-helm.readthedocs.io/en/latest/reeval/). We also include a toy Python script `cat/adap_test.py` to demonstrate the idea of adaptive testing. To reproduce the R-based synthetic experiment result in our paper, read the following:


Single dataset:
```bash
cd cat
python cat.py --scenario air_bench_2024
```


Use WandB to sweep all datasets:
```bash
cd cat
wandb sweep sweep.yaml
```


Open multiple terminals and run:
```bash
wandb agent xx/cat/xx
```


After all the jobs finish, run:
```bash
python cat_analysis.py
```


# (Optional) Download Result


To download the `result/` folder:
```bash
python download.py --folder result
```


# (Optional) Gather Data


If you want to reproduce the experiment from scratch (without using our HuggingFace cache), first download the HELM raw data of the leaderboard `Classic,
Lite, AIR-Bench, Thai Exam, and MMLU` according to [HELM tutorial](https://crfm-helm.readthedocs.io/en/latest/downloading_raw_results/). Save the downloaded folders to `gather_helm_data/helm_jsons/`.


Then run:
```bash
python json2csv.py # -> data/long.pkl
python embed.py # -> data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl and data/embed_mistralai_Mistral-7B-Instruct-v0.3.pkl
python csv2matrix.py # -> data/resmat.pkl
```


# Comments


We describe the Rasch model as P = sigmoid($\theta$ - z) in the paper, but implement it as P = sigmoid($\theta$ + z) in the codebase. This is because [a well-known R library `mirt`](https://cran.r-project.org/web/packages/mirt/index.html) uses P = sigmoid($\theta$ + z), and we want to test our result with their output. As a result, in our codebase, a large $\theta$ denotes high ability, and a large z denotes an easy question.

# Reference

Our R-based synthetic adaptive testing experiment is based on github.com/AnyaWMa/ROAR-CAT-Public. 