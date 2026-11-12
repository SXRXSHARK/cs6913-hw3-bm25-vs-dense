import sys
import time
import subprocess
import h5py
import numpy as np
import faiss

def load_vectors(file_name):
    file = h5py.File(file_name, "r") # open the h5 file in read mode              
    ids = file["id"][:].astype(str) # convert the ids to string 
    vectors = file["embedding"][:].astype(np.float32) # convert the embeddings to float32 (converting every vector is stored at that size)
    file.close()
    return ids, vectors                             
 
def read_queries(file_name):                            
    queries = {} # empty dictionary to store the queries
    for line in open(file_name, encoding="utf-8"): # read each line in the file and handle any special characters 
        parts = line.strip().split("\t") # splits the tab-separated values into a list of parts
        if len(parts) == 2:# checks if there are two parts which is an ID and a question 
            queries[parts[0]] = parts[1] # part 0 is the ID and part 1 is the question, so we store them in the dictionary
    return queries

def read_qrels(file_name):
    qrels = {} 
    for line in open(file_name):
        parts = line.split()            
        if len(parts) < 3 or not parts[0].isdigit():
            continue                   
        qid = parts[0]               
        pid = parts[-2]                
        rel = int(parts[-1])            
        if qid not in qrels:
            qrels[qid] = []             
        qrels[qid].append((pid, rel))  
    return qrels

# this is to prepare. The prepare step includes the read_queries and read_qrels functions. (This is the first step of the assigmnent)
def prepare(qrels_file, queries_file, name):
    qrels = read_qrels(qrels_file) # answer key
    queries = read_queries(queries_file) # questions

    questions_out = open(name + ".queries.tsv", "w", encoding="utf-8")
    answers_out = open(name + ".qrels", "w")

    for qid in qrels: # each question in the answer key
        questions_out.write(f"{qid}\t{queries[qid]}\n") # write the question
        for pid, rel in qrels[qid]:
            answers_out.write(f"{qid} 0 {pid} {rel}\n") # write its answers

    questions_out.close() 
    answers_out.close()

# this is to build the search index for the passage vectors using HNSW algorithm. (This is the second step of the assignment)
def build_hnsw(passage_vectors, M, efc): # we build a graph of the passage vectors using HNSW algorithm
    index = faiss.IndexHNSWFlat(384, M, faiss.METRIC_INNER_PRODUCT) # vectors have 384 dimensions, M is the number of connections per node, and we use inner product as the distance metric
    index.hnsw.efConstruction = efc # sets the number of connections to consider during the construction of the graph.
    start = time.time() # how long it takes to build the index
    index.add(passage_vectors) # build the graph 
    print("build time:", time.time() - start, "seconds")
    return index

# this is the search function for the HNSW index. It takes the index, a file containing queries, and an output file to write the results. (This is the third step of the assignment)
def search_hnsw(index, queries_file, out_file):
    index.hnsw.efSearch = 1000 # as asked in the assignment, we set the number to 1000. 
    queries = read_queries(queries_file) # open the queries file and read the queries into a dictionary
    out = open(out_file, "w") # open the output file in write mode to store the results
    total = 0 # time counter to measure how long it takes to search for all queries.

    for qid in queries:
        vector = query_vectors[query_row[qid]].reshape(1, -1) # this question's 384 numbers

        start = time.time()
        scores, rows = index.search(vector, 1000) # find the 1000 closest passages
        total += time.time() - start

        for rank in range(1000):
            pid = passage_ids[rows[0][rank]] # row number to real passage ID
            out.write(f"{qid} Q0 {pid} {rank + 1} {scores[0][rank]} hnsw\n")

    out.close()
    print(out_file, "ms per query:", 1000 * total / len(queries))

# re ranking which is the to combine system 1 and 2 into system 3. (This is the fourth step of the assignment)
# evaluate the results using trec_eval. (This is the fifth step of the assignment)


if __name__ == "__main__":                              
    if sys.argv[1] == "hnsw":
        passage_ids, passage_vectors = load_vectors("msmarco_passages_embeddings_subset.h5")
        query_ids, query_vectors = load_vectors("msmarco_queries_dev_eval_embeddings.h5")
        query_row = {qid: i for i, qid in enumerate(query_ids)} 

        for M, efc in [(4, 50), (8, 200)]:
            index = build_hnsw(passage_vectors, M, efc)
            for s in ["dl19", "dl20", "dev"]:
                search_hnsw(index, s + ".queries.tsv", f"hnsw_M{M}_{s}.txt")



#.h5 are vector files in this assignment which will be used for system 2 and system 3. The .tsv files are text files which will be used for system 1 and system 3 for reranking.
# HNSW stands for Hierarchical Navigable Small World. It’s a fast way to find the vectors closest to a query vector, without comparing the query against all 1 million passages.
# Imports: the tools Python needs.
# load_vectors: read the vectors from the .h5 files.
# read_queries, read_qrels, prepare: get the questions and answer keys ready.
# build_hnsw, search_hnsw: System 2.
# rerank: System 3.
# evaluate: grade everything with trec_eval.
# The bottom section: which part runs when you type python Main.py <step>.
# we must run the code in the order of system 1, system 2, and system 3.
# like how to build the index for system one, we use parser and merger and to search index we use query processor. 
# We use build and search functions for system for HNSW algorithm. 
# build helps to connect passages with similar meaning. 
# trec_eval looks at your system’s ranking, the order of the results, rather than the exact score numbers.
# System 1 finds passages by words.
# System 2 finds passages by meaning, searching all 1M with the HNSW graph.
# System 3 finds passages by words, then orders them by meaning.
