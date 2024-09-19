# Notice: Repository Deprecation
This repository is deprecated and no longer actively maintained. It contains outdated code examples or practices that do not align with current MongoDB best practices. While the repository remains accessible for reference purposes, we strongly discourage its use in production environments.
Users should be aware that this repository will not receive any further updates, bug fixes, or security patches. This code may expose you to security vulnerabilities, compatibility issues with current MongoDB versions, and potential performance problems. Any implementation based on this repository is at the user's own risk.
For up-to-date resources, please refer to the [MongoDB Developer Center](https://mongodb.com/developer).


# TimeSeries ⏰ Anomaly Detection
A guides to creating sentence embeddings across documents + master embeddings collection

Ingest New Time Series Data
When new data arrives, you can directly process it to generate embeddings and then store both the individual embeddings and the master embedding.
This can be done in real-time or in batches, depending on your use case.

2. Generate Embeddings in Real-Time
As each new data point is ingested, generate its embedding immediately:
```
from langchain.embeddings import OpenAIEmbeddings

def process_new_record(record):
    # Generate embedding for the incoming record
    embedding = OpenAIEmbeddings().embed(record['text_field'])
    
    # Optionally, you could store the individual record and its embedding
    collection.insert_one({
        "timestamp": record['timestamp'],
        "text_field": record['text_field'],
        "embedding": embedding
    })
    
    return embedding
```
3. Batch Process and Create Master Embedding
If you’re working with data in batches (e.g., collecting 20 records before processing), you can aggregate the embeddings after a batch is completed:
python
```
batch_embeddings = []

def ingest_batch(batch):
    for record in batch:
        embedding = process_new_record(record)
        batch_embeddings.append(embedding)
    
    # Create a master embedding from the batch
    master_embedding = np.mean(batch_embeddings, axis=0)
    
    # Store the master embedding
    master_document = {
        "time_range": {
            "start": batch[0]['timestamp'],
            "end": batch[-1]['timestamp']
        },
        "master_embedding": master_embedding.tolist()
    }
    master_collection.insert_one(master_document)

    # Clear the batch embeddings for the next batch
    batch_embeddings.clear()
```
4. Store and Index the Master Embedding
After each batch is processed, you store the master embedding and ensure it's indexed for efficient search:
```
master_collection.create_index([("master_embedding", "2dvector")])
```
6. Integrate LangChain for Real-Time Processing
LangChain can be used to orchestrate these processes, especially if you’re integrating this workflow into a larger application or pipeline.
If your ingestion process is streaming, you can leverage LangChain to manage state, handle retries, and coordinate the embedding generation with MongoDB storage.
Example Workflow for Real-Time Ingestion:
```
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

batch_embeddings = []

def process_new_record(record):
    embedding = OpenAIEmbeddings().embed(record['text_field'])
    collection.insert_one({
        "timestamp": record['timestamp'],
        "text_field": record['text_field'],
        "embedding": embedding
    })
    return embedding

def ingest_batch(batch):
    for record in batch:
        embedding = process_new_record(record)
        batch_embeddings.append(embedding)
    
    master_embedding = np.mean(batch_embeddings, axis=0)
    master_document = {
        "time_range": {
            "start": batch[0]['timestamp'],
            "end": batch[-1]['timestamp']
        },
        "master_embedding": master_embedding.tolist()
    }
    master_collection.insert_one(master_document)
    batch_embeddings.clear()
```
# Simulate ingestion of new data
```
new_data_batch = [{"timestamp": ts, "text_field": text} for ts, text in incoming_data]
ingest_batch(new_data_batch)
```
Considerations:
Streaming vs. Batching: Decide if you're processing data in real-time (streaming) or in batches. 

For streaming, you might store each individual record's embedding immediately, whereas, for batching, you aggregate embeddings after a certain number of records.

Error Handling: Implement error handling for cases where embedding generation or database operations might fail.
