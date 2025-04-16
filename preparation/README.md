
# PREPARATION

Before running the unlearning code, please download the models, data, and additionally tokenize the web text data.

Please run each script from the original folder mt-unlearn.

### Code to download the MUSE benchmark

```python
from datasets import load_dataset

# Download and save MUSE-News training data
load_dataset("muse-bench/MUSE-News", "train").save_to_disk("muse-news-train.hf")

# Download and save sustainability splits
load_dataset("muse-bench/MUSE-News", "sust").save_to_disk("muse-news-sust.hf")

# Download and save MUSE-Books data
load_dataset("muse-bench/MUSE-Books", "train").save_to_disk("muse-books-train.hf")
```

### Code to download the MUSE target models and the Llama-2 tokenizer

Run this script to download MUSE models

```bash
huggingface-cli download muse-bench/MUSE-news_target --local-dir muse-news-target
huggingface-cli download muse-bench/MUSE-books_target --local-dir muse-books-target
```

Run this python script to download the Llama-2 tokenizer

```python
from transformers import AutoTokenizer

from huggingface_hub import login
login("YOUR_HUGGINGFACE_TOKEN_WITH_LLAMA2_ACCESS_APPROVED")

# Download and save MUSE-News training data
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.save_pretrained("llama2-7b")
```

### Run the following script to download and tokenizer OpenWebText-2 dataset

```bash
huggingface-cli download Skylion007/openwebtext --repo-type dataset --local-dir web-text.hf
python3 preparation/tokenize_web_text.py --tokenizer="llama2-7b" --dataset_path="web-text.hf" --save_to="web-text-tok-llama2.hf"
```

Note that the latter script uses tokenizer saved to path 'llama2-7b' in the previous python script.
