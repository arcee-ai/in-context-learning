# parallel_insertion.py
import os
import argparse
from typing import List, Dict
import numpy as np
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
import torch
from tqdm import tqdm
import multiprocessing
from config import Config

config = Config()

lock = threading.Lock()
progress_bars = {}


class SharedChromaDB:
    def __init__(self, vector_db_path: str, embedding_model_name: str):
        # Initialize embedding model
        self.embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={
                "device": "cpu"
            },  # Start with CPU, we'll override device per thread
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize ChromaDB client once
        self.connection = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings_model,
        )


def get_device_count():
    try:
        return torch.cuda.device_count()
    except:
        return 0


def partition_data(data: List[Dict], num_partitions: int):
    """Partition data into roughly equal chunks."""
    return np.array_split(data, num_partitions)


def process_partition(
    worker_id: int,
    data_partition: List[Dict],
    shared_db: SharedChromaDB,
    embedding_model_name: str,
    use_gpu: bool = True,
):
    """Process a partition of data on a specific worker (GPU or CPU)."""
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        device = "cuda"
    else:
        device = "cpu"

    # Initialize embedding model for this worker
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Use the shared ChromaDB connection but with our thread's embedding model
    connection = Chroma(
        persist_directory=shared_db.connection._persist_directory,
        embedding_function=embeddings_model,
    )

    # Create progress bar for this worker
    worker_type = "GPU" if use_gpu else "CPU"
    with lock:
        progress_bars[worker_id] = tqdm(
            total=len(data_partition),
            desc=f"{worker_type} {worker_id}",
            position=worker_id,
            leave=True,
        )

    # Process documents in smaller batches to avoid memory issues
    batch_size = 100
    try:
        for i in range(0, len(data_partition), batch_size):
            batch = data_partition[i : i + batch_size]
            docs = []
            for qa_pair in batch:
                metadata = {"question_id": qa_pair["id"], "answer": qa_pair["answer"]}
                docs.append(
                    Document(page_content=qa_pair["question"], metadata=metadata)
                )

            connection.add_documents(documents=docs)

            with lock:  # Only lock the progress bar update
                progress_bars[worker_id].update(len(batch))

    except Exception as e:
        with lock:
            progress_bars[worker_id].write(
                f"Error in {worker_type} {worker_id}: {str(e)}"
            )
        raise
    finally:
        with lock:
            progress_bars[worker_id].close()


def verify_insertion(shared_db: SharedChromaDB, expected_count: int) -> bool:
    """Verify that all documents were inserted correctly."""
    print("\nVerifying insertion...")
    try:
        # Get actual count from the database
        actual_count = len(shared_db.connection._collection.get()["ids"])
        print(f"Expected documents: {expected_count}")
        print(f"Actual documents: {actual_count}")

        return actual_count == expected_count

    except Exception as e:
        print(f"Error during verification: {str(e)}")
        return False


def load_reference_datasets(
    dataset_configs: List[tuple], max_items: int = None
) -> List[Dict]:
    """Load all reference datasets and combine them."""
    qa_pairs = []
    i = 0
    print("Loading datasets...")
    for dataset_name, config in dataset_configs:
        dataset = load_dataset(dataset_name, config, split="train")
        for item in dataset:
            if max_items and i >= max_items:
                break
            messages = item["messages"]
            question = next(m["content"] for m in messages if m["role"] == "user")
            answer = next(m["content"] for m in messages if m["role"] == "assistant")
            qa_pairs.append({"id": str(i), "question": question, "answer": answer})
            i += 1
            if max_items and i >= max_items:
                break
    return qa_pairs


def run_test(
    vector_db_path: str,
    embedding_model: str,
    num_workers: int = 4,
    test_size: int = 1000,
):
    """Run a test insertion with verification."""
    print(f"Running test with {num_workers} workers and {test_size} documents...")

    # Create test dataset config
    dataset_configs = [["mlabonne/orca-agentinstruct-1M-v1-cleaned", "default"]]

    # Load limited dataset
    qa_pairs = load_reference_datasets(dataset_configs, max_items=test_size)
    print(f"Loaded {len(qa_pairs)} test QA pairs")

    # Initialize shared ChromaDB instance
    shared_db = SharedChromaDB(vector_db_path, embedding_model)

    # Partition data
    partitions = partition_data(qa_pairs, num_workers)

    # Process partitions in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id in range(num_workers):
            future = executor.submit(
                process_partition,
                worker_id,
                partitions[worker_id],
                shared_db,
                embedding_model,
                False,  # Use CPU for testing
            )
            futures.append(future)

        # Wait for all tasks to complete and handle any exceptions
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
                raise

    # Verify results
    success = verify_insertion(shared_db, len(qa_pairs))
    if success:
        print("\nTest passed! All documents were inserted correctly.")
    else:
        print("\nTest failed! Document count mismatch.")

    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU instead of CPU")
    parser.add_argument(
        "--test", action="store_true", help="Run test mode with verification"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of documents to use in test mode",
    )
    args = parser.parse_args()

    if args.test:
        run_test(
            config.vector_db_path,
            config.embedding_model,
            num_workers=args.num_workers or 4,
            test_size=args.test_size,
        )
        return

    # Determine number of workers and type
    if args.use_gpu:
        available_workers = args.num_workers or get_device_count()
        if available_workers == 0:
            print("No GPUs available, falling back to CPU")
            args.use_gpu = False
            available_workers = args.num_workers or max(
                1, multiprocessing.cpu_count() - 1
            )
    else:
        available_workers = args.num_workers or max(1, multiprocessing.cpu_count() - 1)

    worker_type = "GPU" if args.use_gpu else "CPU"
    print(f"Using {available_workers} {worker_type}s for parallel processing")

    # Load all data
    qa_pairs = load_reference_datasets(config.reference_datasets)
    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Initialize shared ChromaDB instance
    shared_db = SharedChromaDB(config.vector_db_path, config.embedding_model)

    # Partition data
    partitions = partition_data(qa_pairs, available_workers)

    # Process partitions in parallel
    with ThreadPoolExecutor(max_workers=available_workers) as executor:
        futures = []
        for worker_id in range(available_workers):
            future = executor.submit(
                process_partition,
                worker_id,
                partitions[worker_id],
                shared_db,
                config.embedding_model,
                args.use_gpu,
            )
            futures.append(future)

        # Wait for all tasks to complete and handle any exceptions
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
                raise

    # Verify final results
    if verify_insertion(shared_db, len(qa_pairs)):
        print("\nAll documents were inserted successfully!")
    else:
        print("\nWarning: Document count mismatch in final verification!")


if __name__ == "__main__":
    main()
