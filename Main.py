import sys
import time
import subprocess
import h5py
import numpy as np
import faiss

def load_vectors(file_name):
    file = h5py.File(file_name, "r")
    ids = file["id"][:]             
    vectors = file["embedding"][:]   
    file.close()
    ids = ids.astype(str)
    vectors = vectors.astype(np.float32)
    print("loaded", len(ids), "from", file_name)
    return ids, vectors

if __name__ == "__main__":
    passage_ids, passage_vectors = load_vectors("msmarco_passages_embeddings_subset.h5")
    query_ids, query_vectors = load_vectors("QUERY_FILE_NAME.h5")
    print(passage_vectors.shape) 
    print(query_vectors.shape)     
    print(passage_ids[:3])         