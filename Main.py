import sys
import time
import subprocess
import h5py
import numpy as np
import faiss
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# the vector files are msmarco_passages_embeddings_subset.h5 and msmarco_queries_dev_eval_embeddings.h5. The .tsv files are dl19.queries.tsv, dl20.queries.tsv, and dev.queries.tsv. The .qrels files are dl19.qrels, dl20.qrels, and dev.qrels.
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
    index.add(passage_vectors) # build the graph  (This is used to build the index)
    print("build time:", time.time() - start, "seconds")
    return index

# this is the search function for the HNSW index. It takes the index, a file containing queries, and an output file to write the results. (This is the third step of the assignment)
def search_hnsw(index, queries_file, out_file, efs, k=100):
    index.hnsw.efSearch = efs # how many candidates the graph explores per query, much be >= the number of results returned
    queries = read_queries(queries_file) # open the queries file and read the queries into a dictionary
    out = open(out_file, "w") # open the output file in write mode to store the results
    total = 0 # time counter to measure how long it takes to search for all queries.

    for qid in queries:
        vector = query_vectors[query_row[qid]].reshape(1, -1) # this question's 384 numbers

        start = time.time()
        scores, rows = index.search(vector,k) 
        total += time.time() - start

        n = rows.shape[1]
        for rank in range(n):
            if rows[0][rank] == -1: # faiss pads with -1 when it can't fill the array --> stop writing
                break

            pid = passage_ids[rows[0][rank]] # row number to real passage ID
            out.write(f"{qid} Q0 {pid} {rank + 1} {scores[0][rank]} hnsw\n")

    out.close()
    print(out_file, "ms per query:", 1000 * total / len(queries))

# reranking which is the to combine system 1 and 2 into system 3. (This is the fourth step of the assignment)
def reranking(bm25_file, k, out_file): # this is to read the passages BM25 has found by words and rerank them by meaning using the HNSW index.
    top_k = {} # empty dictionary to store the top k passages for each query
    for line in open(bm25_file): 
        qid, _, pid, rank, _, _ = line.split() # split the line into its components
        if int(rank) <= k:
            if qid not in top_k:
                top_k[qid] = []
            top_k[qid].append(pid) # add the passage ID to the list of top k passages for this question 

    out = open(out_file, "w") # open the output file in write mode to store the reranked results
    total = 0

    for qid in top_k: # for each question in the top k passages
        pids = top_k[qid] # get the passage IDs for this question
        start = time.time() # start the timer to measure how long it takes to rerank the passages
        vectors = passage_vectors[[passage_row[p] for p in pids]] # get the vectors for the top k passages
        with np.errstate(all="ignore"):
            scores = vectors @ query_vectors[query_row[qid]] # compute the inner product between the passage vectors and the query vector to get the scores
        results = sorted(zip(scores, pids), reverse=True) # sort the results by score in descending order
        total += time.time() - start # add the time taken to rerank this question's passages to the total time

        rank = 1 # initialize the rank counter to 1
        for score, pid in results: # write the reranked results to the output file
            out.write(f"{qid} Q0 {pid} {rank} {score} rerank{k}\n") # write the question ID, passage ID, rank, score, and the name of the reranking method to the output file
            rank += 1

    out.close()
    print(out_file, "ms per query:", 1000 * total / len(top_k))

# evaluate the results using trec_eval
def evaluate(qrels_file, run_file, name, trec_eval_path):
    if name == "dev":
        metrics = [
            ("MAP", ["-m", "map"]),
            ("MRR@10", ["-m", "recip_rank", "-M", "10"]),
            ("Recall@100", ["-m", "recall.100"]),
        ]
    else:
        metrics = [
            ("MRR@10", ["-m", "recip_rank", "-M", "10"]),
            ("Recall@100", ["-m", "recall.100"]),
            ("NDCG@10", ["-m", "ndcg_cut.10"]),
            ("NDCG@100", ["-m", "ndcg_cut.100"]),
        ]

    for label, args in metrics:
        cmd = [trec_eval_path] + args + [qrels_file, run_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        value = result.stdout.strip().split()[-1] if result.stdout.strip() else "N/A"
        print(f"{label}      {run_file}      {value}")

# plots asked for by the TAs
METRICS = ["MRR@10", "NDCG@10", "MAP"]

def per_query(qrels_file, run_file, metric):
    # trec_eval -q gives one score per question; returns {query ID: score}
    args = {"MRR@10": ["-M", "10", "-m", "recip_rank"],
            "NDCG@10": ["-m", "ndcg_cut.10"],
            "MAP": ["-m", "map"]}[metric]
    output = subprocess.run(["./trec_eval/trec_eval", "-q"] + args + [qrels_file, run_file],
                            capture_output=True, text=True).stdout
    scores = {}
    for line in output.splitlines():
        name, qid, value = line.split()
        if qid != "all":                       # skip trec_eval's own average line
            scores[qid] = float(value)
    return scores

def rerank_keep_rest(bm25_file, k, out_file):
    # re-rank BM25's top k by meaning, then keep BM25's order for the rest (so MAP stays fair)
    ranked = {}
    for line in open(bm25_file):
        qid, _, pid, rank, _, _ = line.split()
        if qid not in ranked:
            ranked[qid] = []
        ranked[qid].append(pid)

    out = open(out_file, "w")
    for qid in ranked:
        top, rest = ranked[qid][:k], ranked[qid][k:]
        vectors = passage_vectors[[passage_row[p] for p in top]]
        with np.errstate(all="ignore"):
            scores = vectors @ query_vectors[query_row[qid]]
        new_top = [pid for score, pid in sorted(zip(scores, top), reverse=True)]
        for rank, pid in enumerate(new_top + rest, start=1):
            out.write(f"{qid} Q0 {pid} {rank} {1001 - rank} rerank\n")
    out.close()

def plot_rerank_k():
    # Plot 1: each metric as we re-rank BM25's top k (k = 0 means BM25 only)
    ks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for s in ["dl19", "dl20", "dev"]:
        for k in ks[1:]:
            rerank_keep_rest(f"bm25_{s}.txt", k, f"rerank_k{k}_{s}.txt")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, METRICS):
        for s in ["dl19", "dl20", "dev"]:
            values = []
            for k in ks:
                run_file = f"bm25_{s}.txt" if k == 0 else f"rerank_k{k}_{s}.txt"
                scores = per_query(s + ".qrels", run_file, metric)
                values.append(sum(scores.values()) / len(scores))
            ax.plot(ks, values, marker="o", label=s)
        ax.set_title(f"{metric} vs k")
        ax.set_xlabel("k = BM25 results re-ranked (0 = BM25 only)")
        ax.set_ylabel(metric)
        ax.legend()
    plt.tight_layout()
    plt.savefig("rerank_k.png", dpi=150)

def plot_query_length(s="dev"):
    # Plot 3: average score of the questions of each length (dev has the most questions)
    queries = read_queries(s + ".queries.tsv")
    length = {qid: len(text.split()) for qid, text in queries.items()}   # length = number of words

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, METRICS):
        for system in ["bm25", "hnsw_M8", "rerank1000"]:
            scores = per_query(s + ".qrels", f"{system}_{s}.txt", metric)
            groups = {}                                       # length -> scores of those questions
            for qid in scores:
                L = length[qid]
                if L not in groups:
                    groups[L] = []
                groups[L].append(scores[qid])
            lengths = sorted(L for L in groups if len(groups[L]) >= 3)   # skip lengths with < 3 questions
            ax.plot(lengths, [sum(groups[L]) / len(groups[L]) for L in lengths], marker="o", label=system)
        ax.set_title(f"{s}: {metric} by query length")
        ax.set_xlabel("query length (words)")
        ax.set_ylabel(metric)
        ax.legend()
    plt.tight_layout()
    plt.savefig("query_length.png", dpi=150)

if __name__ == "__main__":      
    if sys.argv[1] == "prepare":
        prepare("qrels.eval.one.tsv", "queries.eval.tsv", "dl19")
        prepare("qrels.eval.two.tsv", "queries.eval.tsv", "dl20")
        prepare("qrels.dev.tsv", "queries.dev.tsv", "dev")  

    if sys.argv[1] == "hnsw":
        passage_ids, passage_vectors = load_vectors("msmarco_passages_embeddings_subset.h5")
        query_ids, query_vectors = load_vectors("msmarco_queries_dev_eval_embeddings.h5")
        passage_row = {pid: i for i, pid in enumerate(passage_ids)}
        query_row = {qid: i for i, qid in enumerate(query_ids)}

        for M, efc, efs in [(4, 100, 100), (8, 200, 200)]:
                    index = build_hnsw(passage_vectors, M, efc)
                    for s in ["dl19", "dl20", "dev"]:
                        search_hnsw(index, s + ".queries.tsv", f"hnsw_M{M}_{s}.txt", efs, 100)

    if sys.argv[1] == "rerank":
        passage_ids, passage_vectors = load_vectors("msmarco_passages_embeddings_subset.h5")
        query_ids, query_vectors = load_vectors("msmarco_queries_dev_eval_embeddings.h5")
        passage_row = {pid: i for i, pid in enumerate(passage_ids)}
        query_row = {qid: i for i, qid in enumerate(query_ids)}

        for k in [100, 1000]:
            for s in ["dl19", "dl20", "dev"]:
                reranking(f"bm25_{s}.txt", k, f"rerank{k}_{s}.txt")

    if sys.argv[1] == "evaluate":
        trec_eval_path = "./trec_eval/trec_eval"

        for s in ["dl19", "dl20", "dev"]:
            qrels_file = s + ".qrels"
            # evaluate BM25
            evaluate(qrels_file, f"bm25_{s}.txt", s, trec_eval_path)
            # evaluate HNSW for both parameter settings
            for M, efc, efs in [(4, 100, 100), (8, 200, 200)]:
                evaluate(qrels_file, f"hnsw_M{M}_{s}.txt", s, trec_eval_path)
            # evaluate reranking for both candidate list sizess
            for k in [100, 1000]:
                evaluate(qrels_file, f"rerank{k}_{s}.txt", s, trec_eval_path)
        

#.h5 are vector files in this assignment which will be used for system 2 and system 3. The .tsv files are text files which will be used for system 1 and system 3 for reranking.
# HNSW stands for Hierarchical Navigable Small World. It’s a fast way to find the vectors closest to a query vector, without comparing the query against all 1 million passages.
# Imports: the tools Python needs.
# load_vectors: read the vectors from the .h5 files.
# read_queries, read_qrels, prepare: get the questions and answer keys ready.
# build_hnsw, search_hnsw: System 2.
# reranking: System 3.
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