# simple-search-system

- set up necessary output files : 
  g++ parser.cpp -o parser && ./parser
  g++ merger.cpp -o merger && ./merger

- run search query :
  g++ query-processor.cpp -o query-processor && ./query-processor
  enter disjunctive or conjunctive to select query mode
  enter query

- install and compile trec_eval:
  git clone https://github.com/usnistgov/trec_eval.git
  cd trec_eval
  make
  copy trec_eval and delete the trec_eval directory