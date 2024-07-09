import metrics
import dataset
import colbert_helper

from bisect import bisect_left
from itertools import accumulate

'''
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
#from colbert.data import Queries, Collection
'''

#==============================================================================================

if __name__ == "__main__":

    #dataset.collection_preprocessing()

    docs_dataset, queries_dataset = dataset.load_datasets("proj/docs", "proj/cfquery_detailed")
    searcher = colbert_helper.get_searcher(docs_dataset)
    

    avg_dcg = [0 for i in range(0,queries_dataset.num_rows)]
    avg_idcg = [0 for i in range(0,queries_dataset.num_rows)]

    for q in queries_dataset:
        id = q["qid"]
        query = q["query"]
        relevant = q["answers"]["docs"]

        answer = colbert_helper.search(searcher, docs_dataset, query, 20)

        #Metrics
        #====================================================================================

        dcg_vector, idcg_vector = metrics.dcg(answer, relevant, q["answers"]["scores"])
        
        avg_dcg = [a + b for a,b, in zip(avg_dcg, dcg_vector)]
        avg_idcg = [a + b for a,b, in zip(avg_idcg, idcg_vector)]

    ndcg = [a/b for a,b in zip(avg_dcg, avg_idcg)]

    print(ndcg)
        