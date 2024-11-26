# LLM Context Enhancement Experiment

This project implements an experimental framework for evaluating how providing relevant high quality context affects the quality of language model responses. It uses a vector database to retrieve similar question-answer pairs and compares model outputs with and without this additional context.

## Overview

The system:
1. Loads reference QA pairs from specified datasets
2. Stores them in a vector database for similarity search
3. For each experimental question:
   - Retrieves similar QA pairs as context
   - Generates responses both with and without context
   - Evaluates response quality using a reward model
   - Stores results for analysis

## Features

- Parallel processing for efficient vector database population
- Support for multiple LLM architectures
- Configurable embedding and reward models
- SQLite results storage with comprehensive metrics
- GPU acceleration support
- Batched processing for memory efficiency

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- vLLM
- LangChain
- ChromaDB
- SQLAlchemy
- Datasets (HuggingFace)
- tqdm

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize:

- Dataset sources
- Model selections
- Database paths
- Experiment parameters

Default configuration:
```python
reference_datasets = [
    ("mlabonne/orca-agentinstruct-1M-v1-cleaned", "default"),
]
experiment_dataset = "HuggingFaceTB/smoltalk"
embedding_model = "BAAI/bge-small-en-v1.5"
llm_model = "Qwen/Qwen2.5-7B-Instruct"
reward_model = "internlm/internlm2-7b-reward"
```

## Usage

### Quick Start

Run the complete experiment:
```bash
bash run_experiment.sh
```

This will:
1. Populate the vector database using parallel processing
2. Execute the main experiment
3. Store results in SQLite database

### Manual Execution

1. Populate vector database:
```bash
python parallel_insertion.py --use_gpu
```

2. Run experiment:
```bash
python main.py
```

### Additional Options

Vector database population:
```bash
# CPU-only mode
python parallel_insertion.py --num_workers 4

# Specify GPU count
python parallel_insertion.py --use_gpu --num_workers 2
```

## Project Structure

- `config.py`: Configuration parameters
- `data_loader.py`: Dataset loading utilities
- `database.py`: Vector and SQL database management
- `experiment.py`: Core experimental logic
- `model_manager.py`: Model loading and inference
- `parallel_insertion.py`: Parallel vector database population
- `main.py`: Experiment entry point
- `run_experiment.sh`: Convenience script

## Key Components

### DataLoader
Handles loading and preprocessing of reference and experimental datasets.

### DatabaseManager
Manages two database systems:
- ChromaDB for vector similarity search
- SQLite for experimental results storage

### ModelManager
Handles:
- Model loading/unloading
- Response generation
- Response quality evaluation

### OptimizedExperiment
Orchestrates the experimental process:
1. Vector database setup
2. Batch processing of questions
3. Context-based response generation
4. Quality evaluation
5. Results storage

## Results Storage

Results are stored in SQLite with the following schema:
- question: Original question
- context_score: Similarity score of retrieved context
- context_qa: Retrieved similar QA pair
- with_context_answer: Model response with context
- without_context_answer: Model response without context
- with_context_score: Quality score with context
- without_context_score: Quality score without context
- with_context_better: Boolean indicating if context improved response
