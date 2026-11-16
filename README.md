# bm25-vs-dense-retrieval

- set up python virtual environment:
  python3 -m venv .venv
  source .venv/bin/activate

- set up necessary output files : 
  g++ parser.cpp -o parser && ./parser
  g++ merger.cpp -o merger && ./merger
  .venv/bin/python Main.py prepare
  g++ query-processor.cpp -o query-processor && ./query-processor
  .venv/bin/python Main.py hnsw
  .venv/bin/python Main.py rerank
  .venv/bin/python Main.py evaluate

- install and compile trec_eval:
  git clone https://github.com/usnistgov/trec_eval.git
  cd trec_eval
  make
  copy trec_eval and delete the trec_eval directory