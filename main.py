import dataset
import vsm
import colbert_helper
import metrics

import os
import shutil

if __name__ == "__main__":

    #Set to true if ColBERT index doesn't already exist
    index_first = False

    #Initialization
    #==============================================================================================

    #Name of the index that ColBERT will use for searching (will be recreated if index_first = True)
    colbert_helper.index_name = "index_1"

    #The number of results you want each search to return
    num_results = 20

    #Determines which implementation of TF-IDF weights is used
    vsm_weighting_method = 0

    #Change depending on where the original documents and queries are stored
    path_to_docs = "original_dataset/docs/"
    path_to_cfquery_detailed = "original_dataset/cfquery_detailed"

    docs_dataset, queries_dataset = dataset.load_datasets(path_to_docs, path_to_cfquery_detailed)

    num_queries = queries_dataset.num_rows
    num_docs = docs_dataset.num_rows

    if index_first:
        colbert_helper.create_index(docs_dataset)


    #Results path
    os.makedirs("results", exist_ok=True)

    #Vector Space Model
    #==============================================================================================
    vsm_results = []

    index, doc_norms, max_doc_freq = vsm.write_index(path_to_docs, vsm_weighting_method)

    for q in queries_dataset:

        search_results = vsm.search(index, doc_norms, max_doc_freq, q["query"], num_results, vsm_weighting_method)
        vsm_results.append(search_results)

    #Colbert
    #==============================================================================================
    colbert_results = []

    searcher = colbert_helper.get_searcher(docs_dataset)

    for q in queries_dataset:

        search_results = colbert_helper.search(searcher, docs_dataset, q["query"], num_results)
        colbert_results.append(search_results)

    #Metrics
    #==============================================================================================
        
    #Results
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("Vector Space Model Results\n=======================================================")
    for i, results in enumerate(vsm_results):
        print(f"Query {i+1}: {results}")

    print("\nColBERT Results\n=======================================================")
    for i, results in enumerate(colbert_results):
        print(f"Query {i+1}: {results}")

    print("")

    #Precision-Recall
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    precision = {"Vector Space": [], "ColBERT": []}
    recall = {"Vector Space": [], "ColBERT": []}

    print("Precision-Recall for Vector Space Model\n=======================================================\n")

    for i, results in enumerate(vsm_results):
        p = f"{round(metrics.precision(results, queries_dataset[i]['answers']['docs']), 3):.03f}"
        r = f"{round(metrics.recall(results, queries_dataset[i]['answers']['docs']), 3):.03f}"

        precision["Vector Space"].append(p)
        recall["Vector Space"].append(r)

        print(f"Query {i+1}\n-------------------")

        print(f"Precision: {p}")
        print(f"Recall: {r}\n")

    print("Precision-Recall for ColBERT\n=======================================================\n")

    for i, results in enumerate(colbert_results):
        p = f"{round(metrics.precision(results, queries_dataset[i]['answers']['docs']), 3):.03f}"
        r = f"{round(metrics.recall(results, queries_dataset[i]['answers']['docs']), 3):.03f}"

        precision["ColBERT"].append(p)
        recall["ColBERT"].append(r)

        print(f"Query {i+1}\n-------------------")

        print(f"Precision: {p}")
        print(f"Recall: {r}\n")


    dataset.excel("results/precision.xlsx", precision)
    dataset.excel("results/recall.xlsx", recall)

    #F-Score
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fscore = {"Vector Space": [], "ColBERT": []}

    print("\nF-Score for Vector Space Model\n=======================================================")

    for i, results in enumerate(vsm_results):
        f = f"{round(metrics.fscore(results, queries_dataset[i]['answers']['docs']), 3):.03f}"
        fscore["Vector Space"].append(f)
        print(f"F-Score for Query {i+1}: {f}")

    print("\nF-Score for ColBERT\n=======================================================")

    for i, results in enumerate(colbert_results):
        f = f"{round(metrics.fscore(results, queries_dataset[i]['answers']['docs']), 3):.03f}"
        fscore["ColBERT"].append(f)
        print(f"F-Score for Query {i+1}: {f}")

    dataset.excel("results/fscore.xlsx", fscore)

    print("")

    #Mean Reciprocal Rank
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"Mean Reciprocal Rank for Vector Space Model: {round(metrics.mean_reciprocal_rank(vsm_results, queries_dataset), 3):03}")
    print(f"Mean Reciprocal Rank for ColBERT: {round(metrics.mean_reciprocal_rank(colbert_results, queries_dataset), 3):03}")

    print("")

    #Mean Average Precision
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"Mean Average Precision for Vector Space Model: {round(metrics.mean_average_precision(vsm_results, queries_dataset), 3):03}")
    print(f"Mean Average Precision for ColBERT: {round(metrics.mean_average_precision(colbert_results, queries_dataset), 3):03}")

    print("")

    #Average NDCG
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"Average NDCG for Vector Space Model: {round(metrics.average_ndcg(vsm_results, queries_dataset)[-1], 3):03}")
    print(f"Average NDCG for ColBERT: {round(metrics.average_ndcg(colbert_results, queries_dataset)[-1], 3):03}")

    #NDCG
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ndcg = {"Vector Space": [], "ColBERT": []}

    print("\nNDCG for Vector Space Model\n=======================================================")

    for i, results in enumerate(vsm_results):
        res = f"{round(metrics.ndcg(results, queries_dataset[i]['answers']['docs'], queries_dataset[i]['answers']['scores'])[-1], 3):.03f}"

        ndcg["Vector Space"].append(res)

        print(f"NDCG for Query {i+1}: {res}")

    print("\nNDCG for ColBERT\n=======================================================")

    for i, results in enumerate(colbert_results):
        res = f"{round(metrics.ndcg(results, queries_dataset[i]['answers']['docs'], queries_dataset[i]['answers']['scores'])[-1], 3):.03f}"

        ndcg["ColBERT"].append(res)

        print(f"NDCG for Query {i+1}: {res}")


    dataset.excel("results/ndcg.xlsx", ndcg)

     #Precision-Recall Diagram
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #See results/plots/ directory for all the saved plots 
    show_on_screen = False
    metrics.precision_recall_diagram(vsm_results, colbert_results, queries_dataset, None, show_on_screen)
