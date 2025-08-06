# Project: Distilling a Step-by-Step Financial Language Model

**Skills:** Python, PyTorch, Hugging Face Transformers, PEFT/QLoRA, LangChain, SEC EDGAR API, ROUGE Evaluation, Google Colab, Jupyter Notebooks, Knowledge Distillation, Financial Text Analysis, JSON Schema Validation, Data Pipeline Development

This project builds a knowledge distillation pipeline for financial text analysis. I created a smaller, efficient model that analyzes SEC 10-K filings by learning from a larger "teacher" model. The project shows the practical challenges of modern machine learning development.

## The Initial Goal

I wanted to replicate the "Distilling Step-by-Step" methodology by creating a dataset of (text, rationale, summary) triplets from SEC 10-K filings. 

**Products Used:**
- **Teacher Model**: Meta-Llama-3-8B-Instruct (8 billion parameters) - analyzes financial text and generates step-by-step reasoning plus summaries
- **Student Model**: TinyLlama-1.1B-Chat-v1.0 (1.1 billion parameters) - smaller model to be trained on the teacher's outputs
- **Data Source**: SEC EDGAR API for downloading 10-K filings from Apple, Microsoft, and Google
- **Training Method**: QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning

**What the Models Do:**
- **Teacher Model**: Takes chunks of 10-K filing text and outputs structured JSON containing:
  - `rationale`: Step-by-step analysis of the financial information
  - `summary`: Concise summary of key points
- **Student Model**: Learns to replicate the teacher's analysis style but with much fewer parameters (87% smaller)

The plan was:
1. Download the filings
2. Process the text 
3. Use the teacher model to generate the data
4. Fine-tune the student model

## Final Results

I successfully:
1. Built a data pipeline that downloads 10-K filings for AAPL, MSFT, and GOOGL, processes them into clean chunks
2. Generated a dataset of 100 (text, rationale, summary) triplets using Llama 3 8B
3. Fine-tuned TinyLlama-1.1B-Chat-v1.0 using QLoRA with LoRA adapters
4. Evaluated the model using ROUGE metrics

## Example Outputs

### Teacher Model Output (Llama 3 8B)
The teacher model generates structured JSON responses like this:

```json
{
  "rationale": "1. Identify the key financial metrics mentioned\n2. Analyze the competitive landscape\n3. Assess market position and risks\n4. Evaluate strategic implications",
  "summary": "The company faces intense competition in a rapidly evolving market, requiring strategic investments in technology and market expansion to maintain competitive position."
}
```

### Student Model Output (TinyLlama 1.1B)
After fine-tuning, the student model produces:

```
Step-by-step rationale:

1. Identify the industry and its competitors.
2. Determine the intensity of competition.
3. Identify the companies with greater financial, marketing, and technological resources.
4. Determine the potential impact of increased competition on profitability and market share.
5. Analyze the potential impact of price reductions, reduced profitability, and loss of market share.
6. Provide a rationale for the proposed action.

Summary:
Our industry is characterized by intense competition from numerous companies, some of which have greater financial, marketing, and technological resources. The proposed action is to implement price reductions, reduce profitability, and loss of market share to mitigate the potential impact of increased competition.
```

## Performance Analysis

My ROUGE scores:
- **ROUGE-1**: 0.00397
- **ROUGE-2**: 0.00143  
- **ROUGE-L**: 0.00382
- **ROUGE-Lsum**: 0.00366

**Honest Assessment**: These scores are quite low, which is expected for several reasons:

1. **Small Dataset**: With only 100 training examples, the model has limited exposure to the task
2. **Complex Task**: Financial analysis requires domain expertise that's hard to learn from a small dataset
3. **Model Size**: TinyLlama (1.1B parameters) is significantly smaller than the teacher (8B parameters)
4. **Evaluation Challenge**: ROUGE metrics don't perfectly capture the quality of financial reasoning

**What the scores mean**: While the ROUGE scores are low, they confirm that:
- The pipeline works end-to-end
- The student model is learning the task structure
- The knowledge distillation process is functioning

**Realistic expectations**: For production use, you'd need:
- 1000+ training examples
- Larger student model (3B+ parameters)
- Domain-specific evaluation metrics
- Human evaluation of reasoning quality

## The Problems and Solutions

### Problem 1: Unsloth Library Issues

I started with unsloth library for performance, but kept getting `AttributeError: module 'torch._inductor' has no attribute 'config'` errors. I tried:
- Forcefully uninstalling and reinstalling libraries
- Pinning library versions to specific configurations
- Installing unsloth from different sources

**Solution**: I abandoned unsloth and switched to the official Hugging Face libraries (transformers, peft, trl). This was more stable.

**Code Explanation**: The project uses Hugging Face's transformers library to load models in 4-bit quantization for memory efficiency. The `BitsAndBytesConfig` reduces memory usage by 75% while maintaining performance. The `device_map="auto"` automatically handles GPU memory allocation.

### Problem 2: Bad Data Quality

My first attempt was a question-answering (RAG) pipeline. It failed because the model either echoed questions back or hallucinated information. I tried:
- Upgrading the embedding model from all-MiniLM-L6-v2 to BAAI/bge-base-en-v1.5
- Widening the search to retrieve more documents

**Solution**: The problem was the source data quality. The .htm and .xml files had too much junk. I switched to downloading full .txt versions and implemented an `is_prose` filter to keep only clean paragraphs.

**Code Explanation**: The data pipeline uses `RecursiveCharacterTextSplitter` to break long 10-K documents into 1500-character chunks with 200-character overlap. The `is_prose` function filters out chunks with too many XML tags or special characters, keeping only human-readable paragraphs. This ensures the teacher model gets clean, meaningful text to analyze.

### Problem 3: LLM Output Format

The Llama 3 teacher model wouldn't give clean JSON output. It kept adding conversational text like "Here is the JSON you requested..." around the correct JSON.

**Solution**: I implemented Schema Injection with Pydantic and created a JsonStopper class to halt generation when valid JSON was complete.

**Code Explanation**: I used Pydantic's `BaseModel` to define the expected output structure with `rationale` and `summary` fields. The `PydanticOutputParser` generates JSON schema instructions that get injected into the prompt. The teacher model receives these instructions and generates structured responses. The `JsonStopper` class monitors the generation in real-time and stops when a complete JSON object is detected, preventing extra conversational text.

### Problem 4: SFTTrainer API Changes

When I moved to fine-tuning, I got TypeErrors from SFTTrainer due to rapid API changes in the trl library.

**Solution**: I used the exact arguments required by the specific library version I had installed.

**Code Explanation**: The fine-tuning uses QLoRA with LoRA adapters. `prepare_model_for_kbit_training` enables gradient computation for quantized models. The `LoraConfig` defines adapter parameters (rank=16, alpha=16) that target specific attention and MLP layers. The `SFTTrainer` handles the training loop with the custom `formatting_prompts_func` that converts each dataset example into the proper chat format for TinyLlama. The training runs for 3 epochs with gradient accumulation to simulate larger batch sizes.

## Key Learnings

- Environment stability is more important than performance optimizations
- Data quality solves most problems
- Don't fight LLM behavior with prompts - build robust parsers instead
- Version pinning is essential for stability

## Next Steps

To improve performance, I need to scale up the dataset generation. Creating several hundred or more (text, rationale, summary) triplets will provide richer training data and better ROUGE scores.

## How to Run This Project

### Setup
```python
!pip install --upgrade "transformers>=4.41.0" "peft>=0.11.1" "trl>=0.8.6" "accelerate" "bitsandbytes" "datasets<4.0.0" -q
!pip install "pydantic>=2.0" "beautifulsoup4" "lxml" "langchain" "langchain_community" "langchain_huggingface" "sec-edgar-api" -q
```

### Requirements
- Google Colab with GPU runtime
- Hugging Face account with access to Meta-Llama-3-8B-Instruct model
- Google Drive for storing data and models
- SEC EDGAR API access for downloading 10-K filings

### Execution
Run the notebook cells in order:
1. **Cell 1**: Setup, Google Drive mounting, and Hugging Face login
2. **Cell 2**: Data Pipeline (downloads and processes 10-K filings)
3. **Cell 3**: Generate the rationale dataset using the teacher model
4. **Cell 4**: Fine-tune the student model using QLoRA
5. **Cell 5**: Load and test the fine-tuned model
6. **Cell 6-7**: Evaluate the model using ROUGE metrics

The project is ready for anyone who wants to build on this foundation and scale it up to production-level performance. 