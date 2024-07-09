'''
Colbert indexing and searching
'''

import sys;
sys.path.insert(0, '/home/zoukos/ceid/Information_Retrieval/ColBERT/')

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
#from colbert.data import Queries, Collection

#==============================================================================================

checkpoint = "colbertv2.0"
doc_maxlen = 512
nbits = 2
index_name = "index_1"
nranks = 1 #Number of GPUs
kmeans_niters = 4 #Number of iterations of k-means clustering

#==============================================================================================

def create_index(docs_dataset):
    '''
    Creates an index from the input dataset.
    Accepts a dataset created with load_dataset containing all the documents.
    '''
    global index_name
    global nbits
    global doc_maxlen
    global checkpoint

    with Run().context(RunConfig(nranks=1)):

        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)                                          

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=docs_dataset["text"], overwrite=True)

        print("PATH: ", indexer.get_index())

#==============================================================================================

def get_searcher(docs_dataset):
    global index_name

    searcher = None

    with Run().context(RunConfig()):
        searcher = Searcher(index=index_name, collection=docs_dataset["text"])

    return searcher

#==============================================================================================

def search(searcher, docs_dataset, query, num_results: int):
    
    search_results = searcher.search(query, k=num_results)[0]

    #Maps the ids returned by colbert to our own ids
    #Colbert ids are in search_results
    #Our ids are stored in the dataset under "doc"
    return list(map(lambda passage_id: docs_dataset[passage_id]["doc"], search_results))