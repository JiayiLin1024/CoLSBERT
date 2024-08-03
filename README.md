# CoLSBERT



## Dependency 

```bash
pip install torch
pip install transformers
```

## Pre-train

- scaling training data

```bash
cd pre-train/training_data

sbatch run-1x.slurm

sbatch run-2x.slurm

sbatch run-4x.slurm

sbatch run-8x.slurm
```


- scaling model size

```bash
cd pre-train/model_size

sbatch run-124M.slurm

sbatch run-354M.slurm

sbatch run-757M.slurm

sbatch run-1.5B.slurm
```


- scaling computing resource

```bash
cd pre-train/computing_resource

sbatch run.slurm
```

- CoLSBERT

```bash
cd pre-train/CoLSBERT

sbatch run.slurm
```


## Downstream task

**Code Search**

- data download (CSN)

```bash
cd dataset
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
cd ../..
```
    
    
- fine-tune
    
```bash
## scaling training data
cd fine-tune/code_search/training_data
bash run.sh


## scaling model size
cd fine-tune/code_search/model_size
bash run_model_size_124M.sh
bash run_model_size_354M.sh
bash run_model_size_757M.sh
bash run_model_size_1.5M.sh



## scaling computing resource
cd fine-tune/code_search/computing_resource
bash run.sh


## CoLSBERT
cd fine-tune/code_search/CoLSBERT
bash CoLSBERT_python.sh
bash CoLSBERT_java.sh
bash CoLSBERT_go.sh
bash CoLSBERT_php.sh
bash CoLSBERT_javascript.sh
bash CoLSBERT_ruby.sh
```


**Clone Detection**

- data download (POJ-104)

```bash
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
tar -xvf programs.tar.gz
python preprocess.py
cd ..
```


- fine-tune
    
```bash
cd fine-tune/clone_detection


## CoLSBERT
bash run.sh
```



## Results

- code search 

| model | CSN-Ruby | CSN-Javascript | CSN-Go | CSN-Python | CSN-Java | CSN-Php | CSN-Avg | 
| - | - | - | - | - | - |- | - | 
| CodeGPT (124M) | 60.8 | 54.3 | 87.9 | 65.9 | 63.2 | 58.3 | 65.1 |
| CodeGen (350M) | 69.6 | 64.8 | 90.1 | 69.0 | 70.6 | 64.4 | 71.4 |
| phi-1 (1.3B) | 71.9 | 65.1 | 90.0 | 72.3 | 71.6 | 65.5 | 72.9 |
| CodeBERT (124M) | 68.0 | 62.6 | 89.2 | 68.5 | 68.6 | 64.2 | 70.2 |
| GraphCodeBEET (124M) | 70.6 | 65.6 | 90.0 | 70.6 | 70.0 | 65.7 | 72.1 | 
| SynCoBERT (124M) | 72.2 | 67.7 | 91.3 | 72.4 | 72.3 | 67.8 | 74.0 | 
| UniXcoder (124M) | 73.9 | 68.6 | 91.6 | 72.3 | 72.7 | 67.3 | 74.4 |
| CodeSage (1.3B) | 77.2 | 71.5 | 91.1 | 73.3 | 73.7 | 68.0 | 75.8 | 
| CoLSBERT (1.5B) | 76.8 | 72.2 | 92.2 | 75.9 | 75.1 | 70.0 | 77.0 |
| CoLSBERT-multilingual (1.5B) | 78.7 | 73.0 | 92.2 | 76.5 | 75.6 | 70.1 | 77.7 | 


- clone detection

| model | POJ-104 | 
| - | - |
| CodeGPT (124M) | 68.54 |
| CodeGen (350M) | 90.08 | 
| phi-1 (1.3B) | 89.20 |
| CodeBERT (124M) | 83.79 | 
| GraphCodeBEET (124M) | 85.50 | 
| SynCoBERT (124M) | 88.24 |
| UniXcoder (124M) | 89.56 | 
| CodeSage (1.3B) | 87.70 | 
| CoLSBERT (1.5B) | 92.91 | 