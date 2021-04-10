# python_elasticesearch_evaluation
python, Elasticsearch, evaluation, top-10, top-5, P@K, P@5, R@K, R@5, database - dataset: movies from wikipedia

evaluate a system, information needs must be germane (relevant) to the documents in the test document collection, and appropriate for predicted usage of the system. 

Given information needs and documents, collect relevance assessments. This is a time-consuming and expensive process involving human beings. 

For tiny collections, exhaustive judgments of relevance for each query and document pair can be obtained. 
For large modern collections, it is usual for relevance to be assessed only for a subset of the documents for each query. 
The most standard approach is pooling, 
where relevance is assessed over a subset of the collection that is formed 
from the top k documents returned by  many different IR systems (usually the ones to be evaluated).

The Document Collection (dataset): Wikipedia Movie. 
data download: https://www.kaggle.com/jrobischon/wikipedia-movieplots?select=wiki_movie_plots_deduped.csv .

In the program:
1) ·	Building a Test Collection:
  devise a small test collection that contains a number of queries, together with their expected results.
  Identify three information needs covered by the collection and then compose a sample queries for each.
2) ·	IR systems, 2 here to compare with each other
  IR 1 & IR 2 have different configuration settings, namely, with different parameters.
  IR 2 with stemming, while IR 1 without.
3) ·	Pooling, pooling the results retrieved by these 2 IR systems
  construct a pool by putting together the top 10 retrieval results from your 2 IR systems.
4) ·	Assessing relevance
  binary relevance judgements
  A document is either relevant or non-relevant (not relevant) for an information need. 
5) ·	Evaluation 
  identify a suitable metric. Use P@5 and R@5 as the metric of choice for this program

// this code is the continuation of ' python_elasticesearch_search_engine ', link: https://github.com/mellowLoveGH/python_elasticesearch_search_engine .
