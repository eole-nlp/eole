# flake8: noqa

import os
from rich import print
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

from eole.utils.logging import init_logger
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY

# Set up logging
logger = init_logger()

# 1. Load and Split the Document
logger.info("Loading and splitting the document...")
loader = PyMuPDFLoader("./OJ_L_202401689_EN_TXT.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
documents = text_splitter.split_documents(documents=docs)
print(f"[INFO] Total chunks: {len(documents)}")

# 2. Set Up ChromaDB Client and Collection
logger.info("Setting up ChromaDB client...")
chroma_client = chromadb.PersistentClient(path="chromadb_data")
collection = chroma_client.get_or_create_collection(name="test-eu")

# 3. Insert Documents into Collection
logger.info("Checking for existing data in the collection...")

# Retrieve all existing IDs from the collection
existing_ids = set(collection.get(ids=None)["ids"])  # Fetches all IDs in the collection
logger.info(f"Found {len(existing_ids)} existing documents in the collection.")
batch_size = 100
for i in tqdm(
    range(0, len(documents), batch_size),
    desc="Upserting batches in the vector database",
):
    batch_ids = [str(k) for k in range(i, min(len(documents), i + batch_size))]
    new_ids = [
        id_ for id_ in batch_ids if id_ not in existing_ids
    ]  # Filter out existing IDs

    if new_ids:  # Only upsert if there are new IDs
        new_docs = [documents[int(id_)] for id_ in new_ids]
        collection.upsert(
            ids=new_ids,
            documents=[doc.page_content for doc in new_docs],
            metadatas=[doc.metadata for doc in new_docs],
        )
        logger.info(f"Upserted {len(new_ids)} new documents.")
    else:
        logger.info(f"Skipping batch {i // batch_size + 1}, all IDs already exist.")

# print(collection.peek(10))

# 4. Query the Collection
QUERY = "What is the general position around using biometrics and facial recognition in public places?"
# QUERY = "Are there any derogations for specific actors?"
# QUERY = "What are the main obligations of importers?"
# QUERY = "What are the main risks and penalties incurred?"
# QUERY = "What is the maximum fine for potential offenders?"
# QUERY = "What are the main prohibited practices coverd by the act?"
# QUERY = "What are the main accepted practices covered by the act?"

print(f"[INFO] Querying collection with: {QUERY}")
results = collection.query(query_texts=[QUERY], n_results=5)

best_id = int(results["ids"][0][0])
print(f"[INFO] Best result ID: {best_id}")
context_docs = collection.get(ids=[str(best_id - 1), str(best_id), str(best_id + 1)])

# 5. Prepare the Prompt for Inference
PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. The answer should be understandable outside of its context.
The context comes from this document: "Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU) No 168/2013, (EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797 and (EU) 2020/1828 (Artificial Intelligence Act)Text with EEA relevance."
Question: {question}
Context:  {context}
Answer:"""

context = "\n".join(context_docs["documents"])
prompt = PROMPT.format(question=QUERY, context=context)
logger.info("Generated Prompt:")
print(prompt)


# 6. Perform Inference
logger.info("Running inference...")
config = PredictConfig(
    model_path=os.path.expandvars("${EOLE_MODEL_DIR}/llama3.1-8b"),
    src="dummy",
    max_length=500,
    gpu_ranks=[0],
    # Uncomment to activate bnb quantization
    # quant_type="bnb_NF4",
    # quant_layers=[
    #     "gate_up_proj",
    #     "down_proj",
    #     "up_proj",
    #     "linear_values",
    #     "linear_query",
    #     "linear_keys",
    #     "final_linear",
    #     "w_in",
    #     "w_out",
    # ],
    top_p=0.3,
    temperature=0.35,
    beam_size=5,
    seed=42,
    batch_size=1,
    batch_type="sents",
)

engine = InferenceEnginePY(config)

_, _, predictions = engine.infer_list([prompt])

# 7. Display the Prediction
answer = predictions[0][0].replace("｟newline｠", "\n")
logger.info("Final Answer:")
print(answer)
