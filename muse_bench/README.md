This is copy of MUSE benchmark's code adapted for integrated evaluations during unlearning. See references to the original work below:

[Paper](https://www.arxiv.org/abs/2407.06460) | [Website](https://muse-bench.github.io/) | [Code](https://github.com/swj0419/muse_bench) | [MUSE-News Benchmark](https://huggingface.co/datasets/muse-bench/MUSE-News) | [MUSE-News Benchmark](https://huggingface.co/datasets/muse-bench/MUSE-News) |  [MUSE-Books Benchmark](https://huggingface.co/datasets/muse-bench/MUSE-Books) 

The following changes were made:

- use vLLM for faster generations

- added MMLU evaluation (subsampled 1535 questions)

- Q/A pairs added directly to repository in ```./data``` folder