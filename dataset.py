'''
A module for preprocessing and loading datasets
'''

import sys;
sys.path.insert(0, '/home/zoukos/ceid/Information_Retrieval/ColBERT/')

import json
import re
import os
import pandas as pd
from datasets import load_dataset

#==============================================================================================

def relevant():
    '''
    Returns a list indexed by the query id - 1.
    Each element is a list of relevant documents, in no particular order
    '''
    res = []

    with open("proj/Relevant_20", "r") as f:
        for relevant_list in f:
            relevant_list = relevant_list.split()
            res.append(relevant_list)

    return res

#==============================================================================================

def break_query(query):
    '''
    File cfquery_detailed contains 20 sections, one for each query. This function breaks each sections into its parts:
    - QN is the query id, now saved as "qid"
    - QU is the query, now saved as "query"
    - RD is the relevant documents and their scores, now saved as "answers: {docs: [], scores: []}"
    '''
    #What
    pattern = re.compile("^QN\s+(?P<qid>[0-9]{5})\s+QU\s+(?P<query>.*?)NR\s+(?P<relevant_count>[0-9]{5})\s+RD\s+(?P<relevant_docs>.*)")
    match = re.match(pattern, query)

    query_dict = {"qid": None, "query": None, "answers": {"docs": [], "scores": []}}

    replace_spaces = re.compile("\s{2,}")

    query_dict["qid"] = int(match.group("qid"))
    query_dict["query"] = replace_spaces.sub(" ", match.group("query"))[:-1]

    relevant_docs = replace_spaces.sub(" ", match.group("relevant_docs").strip()).split(" ")

    for i in range(0, int(match.group("relevant_count"))):
        query_dict["answers"]["docs"].append(int(relevant_docs[2*i]))
        query_dict["answers"]["scores"].append(relevant_docs[2*i + 1])

    return query_dict

#==============================================================================================

def collection_preprocessing(docs_path, queries_path):
    '''
    Preprocesses our collection to make it usable with ColBERT's data structures.
    docs_path (e.g. proj/docs/) should be a directory containing all the documents.
    queries_path (e.g. proj/cfquery_detailed) should lead to the DETAILED query list.
    Creates two directories json_docs and json_queries.

    In json_docs, each document is represented as a json file containing:
    - doc: The document id, from 1 to 1239
    - text: The actual contents of the document

    In json_query, a single file called queries.json contains all queries as json objects, keeping track of:
    
    - qid: The id of the query, from 1 to 20
    - query: The query itself
    - answers: An object containing two arrays "docs" and "scores", the first one containing all relevant documents
    and the second one containing their respective scores as a 4-digit string
    '''

    os.makedirs("json_docs", exist_ok=True)
    os.makedirs("json_queries", exist_ok=True)

    #Documents
    #===============================================================================
    for i in range(1, 1240):
        try:
            f = open(docs_path + f"{i:05}")

            text = ""

            for word in f:
                text += word[:-1] + " "
                #print(word)

            f.close()

            json_obj = {"doc": i, "text": text[:-1]}

            with open("json_docs/" + f"{i:05}" + ".json", "w") as json_file:
                json.dump(json_obj, json_file, indent="\t")

        except FileNotFoundError as ferr:
            continue

    #Queries
    #===============================================================================
    queries = None

    with open(queries_path, "r") as f:
        queries_file = f.read().replace("\n", " ")

    #print(queries)
        
    #Retrieves all the text between two occurencies of "QN"
    extract_query = re.compile("^QN.*?(?=(QN|$))")

    queries = []

    #Get each query section
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    while(True):

        if queries_file == "":
            break
        
        #Get first match and remove it from the document
        #In the next iteration, you'll retrieve the next match
        queries.append(re.match(extract_query, queries_file).group())
        queries_file = extract_query.sub("", queries_file)
    
    #Break query section into its parts
    queries = list(map(break_query, queries))

    with open("json_queries/queries.json", "w") as f:
        json.dump(queries, f, indent="\t")

#==============================================================================================

def load_datasets(docs_path, queries_path):
    '''
    Loads the two preprocessed datasets from the existing json_queries/ json_docs directories.
    If these directories don't exist, they will be created from the documents in docs_path and the DETAILED query list pointed to by queries_path

    Returns:
    - docs_dataset
    - queries_dataset
    '''

    #Structure of docs_dataset (for document i >= 0):
    #   docs_dataset[i]["doc"]: The document ID, from 1 to 1239
    #   docs_dataset[i]["text"]: The actual text content

    #Structure of queries_dataset (for query i >= 0):
    #   queries_dataset[i]["qid"]: The query ID, from 1 to 20
    #   queries_dataset[i]["query"]: The actual query
    #   queries_dataset[i]["answers"]: A dictionary with the following:
    #       ["docs"]: A list relevant documents to this query
    #       ["scores"]: A list of scores, each represented as a 4-digit string, for the respective relevant documents

    if not (os.path.exists("json_queries/") and os.path.exists("json_docs/")):
        print("Preprocessing collection...\n")
        collection_preprocessing(docs_path, queries_path)

    queries_dataset = load_dataset("json_queries")["train"]
    docs_dataset = load_dataset("json_docs")["train"]

    return docs_dataset, queries_dataset

#==============================================================================================

def excel(filename, dict_data, query_ids = None):

    if query_ids is None:
        query_ids = range(1,21)

    df = pd.DataFrame(dict_data, index=query_ids)
    df.to_excel(filename, index_label="Query ID")