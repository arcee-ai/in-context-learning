import chromadb
from chromadb.utils import embedding_functions
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from typing import List, Dict, Tuple
from config import Config
from sqlalchemy import create_engine, Column, Integer, String, JSON, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

# Set multiprocessing start method to 'spawn' for CUDA support
mp.set_start_method("spawn", force=True)

Base = declarative_base()


class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True)
    question = Column(String)
    context_qa = Column(JSON)
    with_context_answer = Column(String)
    without_context_answer = Column(String)
    with_context_score = Column(Float)
    without_context_score = Column(Float)


def process_chunk(args):
    """Standalone function for processing chunks to avoid pickling issues"""
    gpu_id, chunk, model_name = args

    # Initialize device
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, device_map=device, torch_dtype=torch.float16
    )
    model.eval()

    # Process data
    all_embeddings = []
    all_ids = []
    all_metadata = []
    all_documents = []

    # Process in batches
    batch_size = 32
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i : i + batch_size]

        # Prepare batch data
        questions = [item["question"] for item in batch]

        # Get embeddings
        inputs = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

        # Collect results
        all_embeddings.extend(embeddings)
        all_ids.extend(item["id"] for item in batch)
        all_metadata.extend(
            {"answer": item["answer"], "question_id": item["id"]} for item in batch
        )
        all_documents.extend(questions)

    return all_embeddings, all_ids, all_metadata, all_documents


class ParallelDatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        print(f"Found {self.num_gpus} GPUs")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=config.vector_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="qa_pairs", metadata={"hnsw:space": "cosine"}
        )

        # Results database
        self.results_engine = create_engine(config.results_db_path)

    def store_qa_pairs(self, qa_pairs: List[dict]):
        """Store QA pairs using parallel GPU processing"""
        # Split data among available GPUs
        chunk_size = max(1, len(qa_pairs) // self.num_gpus)
        chunks = [
            qa_pairs[i : i + chunk_size] for i in range(0, len(qa_pairs), chunk_size)
        ]

        # Prepare arguments for the process_chunk function
        args = [
            (i, chunk, self.config.embedding_model) for i, chunk in enumerate(chunks)
        ]

        # Use Pool to process chunks in parallel
        with mp.Pool(processes=self.num_gpus) as pool:
            results = pool.map(process_chunk, args)

        # Combine results
        all_embeddings = []
        all_ids = []
        all_metadata = []
        all_documents = []

        for emb, ids, meta, docs in results:
            all_embeddings.extend(emb)
            all_ids.extend(ids)
            all_metadata.extend(meta)
            all_documents.extend(docs)

        # Batch insert into ChromaDB
        self.collection.add(
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadata,
            documents=all_documents,
        )

    def find_similar_question(self, question: str) -> Tuple[str, str]:
        """Find similar questions using the embedding model"""
        # Process single question using the same function
        result = process_chunk(
            (
                0,
                [{"id": "query", "question": question, "answer": ""}],
                self.config.embedding_model,
            )
        )

        embeddings = result[0][0]  # Get first embedding

        # Query the collection
        results = self.collection.query(
            query_embeddings=[embeddings.tolist()], n_results=1
        )

        if results and len(results["documents"]) > 0:
            return results["documents"][0][0], results["metadatas"][0][0]["answer"]
        return None, None

    def store_result(self, result: Dict):
        """Store experiment results"""
        Session = sessionmaker(bind=self.results_engine)
        with Session() as session:
            result_record = Result(**result)
            session.add(result_record)
            session.commit()
