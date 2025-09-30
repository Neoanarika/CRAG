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

```
wget -O crag_task_1_and_2_dev_sample30_v4.jsonl.bz2 'https://www.dropbox.com/scl/fi/1q8c0zudgbf7k45ksw7x0/crag_task_1_and_2_dev_sample30_v4.jsonl.bz2?rlkey=pmi0b653xdqg0t4u3ua4ui276&e=1&st=i1d8ww2b&dl=1'
```

## ⚙️ Model

Download the models
```
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct   --local-dir models/meta-llama/Meta-Llama-3-8B-Instruct   --local-dir-use-symlinks False   --include "config.json"            "generation_config.json"            "tokenizer.json"            "tokenizer.model"            "tokenizer_config.json"            "*.safetensors"
```

```
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/sentence-transformers/all-MiniLM-L6-v2
```

## 📏 Evaluation Metrics
RAG systems are evaluated using a scoring method that measures response quality to questions in the evaluation set. Responses are rated as perfect, acceptable, missing, or incorrect:

- Perfect: The response correctly answers the user question and contains no hallucinated content.

- Acceptable: The response provides a useful answer to the user question, but may contain minor errors that do not harm the usefulness of the answer.

- Missing: The answer does not provide the requested information. Such as “I don’t know”, “I’m sorry I can’t find …” or similar sentences without providing a concrete answer to the question.

- Incorrect: The response provides wrong or irrelevant information to answer the user question


Auto-evaluation: 
- Automatic evaluation employs rule-based matching and LLM assessment to check answer correctness. It will assign three scores: correct (1 point), missing (0 points), and incorrect (-1 point).


Please refer to [local_evaluation.py](local_evaluation.py) for more details on how the evaluation was implemented.

## ✍️ How to run end-to-end evaluation?
1. **Install** specific dependencies
    ```bash
    pip install -r requirements.txt
    ```

2. Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models.

3. After writing your own model(s), update [models/user_config.py](models/user_config.py)

   For example, in models/user_config.py, specify InstructModel to call llama3-8b-instruct model
   ```bash
   from models.vanilla_llama_baseline import InstructModel 
   UserModel = InstructModel

   ```

4. Test your model locally using `python local_evaluation.py`. This script will run answer generation and auto-evaluation.


## 🏁 Baselines
We include three baselines for demonstration purposes, and you can read more about them in [docs/baselines.md](docs/baselines.md).

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).
