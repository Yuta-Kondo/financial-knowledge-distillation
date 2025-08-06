# Setup Instructions

## Before Running the Notebook

1. **Replace Personal Information**: 
   - In Cell 0, change `USER_AGENT = "Your Name your.email@example.com"` to your actual name and email
   - This is required by the SEC EDGAR API

2. **Hugging Face Token**:
   - Set up your Hugging Face token in Google Colab's user data
   - Go to Runtime → Manage secrets → Add new secret
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token

3. **Model Access**:
   - Ensure you have access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face
   - Request access if needed: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

## Running the Project

1. Open the notebook in Google Colab with GPU runtime
2. Run cells in order from top to bottom
3. The project will:
   - Download 10-K filings for AAPL, MSFT, and GOOGL
   - Generate a dataset using the teacher model
   - Fine-tune the student model
   - Evaluate performance

## Customization

- **Target Companies**: Modify `TARGET_COMPANIES` in Cell 0 to analyze different companies
- **Dataset Size**: Change the range in Cell 2 (`chunks_to_process = hf_dataset.select(range(100))`) to process more or fewer examples
- **Training Parameters**: Adjust epochs, learning rate, etc. in Cell 3

## Output Files

- Downloaded 10-K filings: `data/10-K/`
- Generated dataset: `rationale_dataset_final.jsonl`
- Trained model: `models/` 