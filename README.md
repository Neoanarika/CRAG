# CRAG RAG item response model

This is a proof of concept. To make this alot more robust and interesting, alot more compute budget and resources would be required to bring this project into fruition. 

## 🚀 Setup

Set env variables 
```
export OPENAI_API_KEY=<YOU_API_KEY>
```

To set up the R environment:

conda env create -f cat.yml
conda activate cat

## 📊 Dataset 

Go to ```data``` directory then 

```
wget -O crag_task_1_and_2_dev_sample30_v4.jsonl.bz2 'https://www.dropbox.com/scl/fi/lv1b8sgkyoefc9w1pgzc7/crag_task_1_and_2_dev_top30.jsonl.bz2?rlkey=3w3w913mo0fltqpxx2rvxe6h0&st=sz4uk9l7&dl=1'
```

## ⚙️ Model

Download the models
```
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct   --local-dir models/meta-llama/Meta-Llama-3-8B-Instruct   --local-dir-use-symlinks False   --include "config.json"            "generation_config.json"            "tokenizer.json"            "tokenizer.model"            "tokenizer_config.json"            "*.safetensors"
```

```
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/sentence-transformers/all-MiniLM-L6-v2
```

## Running Post-Hoc CAT for RAG systems (proof of concept study)

1. Get calibration data
```
```

2. Train IRT model (Rasch model)
```
cd cat
python calibration.py
```

3. Perform CAT
```
python cat.py
```
## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).
