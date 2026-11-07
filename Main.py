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


def read_queries(file_name):                            
    queries = {}
    for line in open(file_name, encoding="utf-8"):
        parts = line.strip().split("\t")
        if len(parts) == 2:
            queries[parts[0]] = parts[1]
    return queries

    
def build_hnsw(passage_vectors, M, efc):
    index = faiss.IndexHNSWFlat(384, M, faiss.METRIC_INNER_PRODUCT)  
    index.hnsw.efConstruction = efc
    start = time.time()
    index.add(passage_vectors)
    print("build time:", time.time() - start, "seconds")
    return index


def search_hnsw(index, queries_file, out_file):
    index.hnsw.efSearch = 1000  
    queries = read_queries(queries_file)
    out = open(out_file, "w")
    total = 0

    for qid in queries:
        if qid not in query_row:
            continue
        vector = query_vectors[query_row[qid]].reshape(1, -1)

        start = time.time()
        scores, rows = index.search(vector, 1000)
        total += time.time() - start

        for rank in range(1000):
            row = rows[0][rank]
            if row == -1:
                break
            out.write(f"{qid} Q0 {passage_ids[row]} {rank + 1} {scores[0][rank]} hnsw\n")

    out.close()
    print(out_file, "ms per query:", 1000 * total / len(queries))


if __name__ == "__main__":                              
    if sys.argv[1] == "hnsw":
        passage_ids, passage_vectors = load_vectors("msmarco_passages_embeddings_subset.h5")
        query_ids, query_vectors = load_vectors("msmarco_queries_dev_eval_embeddings.h5")
        query_row = {qid: i for i, qid in enumerate(query_ids)} 

        for M, efc in [(4, 50), (8, 200)]:
            index = build_hnsw(passage_vectors, M, efc)
            for s in ["dl19", "dl20", "dev"]:
                search_hnsw(index, s + ".queries.tsv", f"hnsw_M{M}_{s}.txt")