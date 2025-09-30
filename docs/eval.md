## 📏 Evaluation Metrics
RAG systems are evaluated using a scoring method that measures response quality to questions in the evaluation set. Responses are rated as perfect, acceptable, missing, or incorrect:

- Perfect: The response correctly answers the user question and contains no hallucinated content.

- Acceptable: The response provides a useful answer to the user question, but may contain minor errors that do not harm the usefulness of the answer.

- Missing: The answer does not provide the requested information. Such as “I don’t know”, “I’m sorry I can’t find …” or similar sentences without providing a concrete answer to the question.

- Incorrect: The response provides wrong or irrelevant information to answer the user question

Auto-evaluation: 
- Automatic evaluation employs rule-based matching and LLM assessment to check answer correctness. It will assign three scores: correct (1 point), missing (0 points), and incorrect (-1 point).

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


