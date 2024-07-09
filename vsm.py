'''
Everything related to the Vector Space Model
'''

import math

#==============================================================================================

def write_index(docs_path, weighting_method):
    '''
    Creates an inverted index from the documents in docs_path.
    It writes the resulting index to disk as "inverted_index.txt".
    It also returns the index as a dictionary of terms.

    For each term, a tuple of (document frequency, occurencies).
    "occurencies" is an array of occurencies.
    Each occurence is represented as a tuple of (document_id, frequency, [word positions]).

    - docs_path: A path (e.g. proj/docs/) to a directory will all the documents

    Returns:
    - The index
    - A dictionary of terms containing their norms
    - A dictionary of terms containing their max frequency between all documents
    '''

    index = dict()

    max_doc_freq = dict() #id: int => max_freq: int

    N = 0 #Number of documents

    for i in range(1, 1240):
        linecount = 1
        temp = dict() #For each word, give the number of occurencies in the current document
        pos = dict()

        #First get the occurencies and the positions for each term
        try:
            f = open(docs_path + f"{i:05}")
            
            for word in f:
                word = word.strip()

                if word in temp:
                    temp[word] += 1
                    pos[word].append(linecount)
                else:
                    temp[word] = 1
                    pos[word] = [linecount]

                linecount += 1

            f.close()

            #Updating the inverted file using data from the new document
            for term, freq in temp.items():
                if term in index:
                    index[term].append((i, freq, pos[term]))
                else:
                    index[term] = [(i, freq, pos[term])]

                if i in max_doc_freq:
                    max_doc_freq[i] = max(max_doc_freq[i], freq)
                else:
                    max_doc_freq[i] = freq

            N += 1

        except FileNotFoundError as ferr:
            continue

    #Calculate the norms of each document in the collection, examining one term at a time
    #Write final index to file
    #=====================================================================================================

    doc_norms = dict() #id: int, norm: float
    
    with open("results/inverted_index.txt", "w") as out:
        for term, postings in index.items():
            #Each posting is p = (doc_id, num_occurencies, [list_of_occurencies])
            
             #Store this term's document frequency in the index
            out.write(f"{term}: ({len(postings)}, {postings})\n")
            index[term] = (len(postings), postings)

            #Calculate this term's contribution to each document's norm
            #====================================================================
            for p in postings:

                val = calculate_weight(p[1], max_doc_freq[p[0]], N, len(postings), weighting_method)

                if p[0] in doc_norms: #If this text's norm has been initialized
                    doc_norms[p[0]] += val**2
                else:
                    doc_norms[p[0]] = val**2

        for doc in doc_norms:
            doc_norms[doc] = math.sqrt(doc_norms[doc])
            #print(f"Document #{doc}: {doc_norms[doc]}")

    return index, doc_norms, max_doc_freq

#==============================================================================================

def freq(index: dict, term: str, doc_id: int):
    '''
        Return a term's frequency inside of a document.
        Uses binary search to locate the requested doc_id, assuming that the postings list is sorted by document id.
    '''
    postings = index[term][1]

    low = 0
    high = len(postings) - 1

    while low <= high:
        mid = (low + high) // 2
        val = postings[mid][0]

        if val == doc_id:
            #print(postings[mid])
            return postings[mid][1]
        elif val < doc_id:
            low = mid + 1
        else:
            high = mid - 1

    return 0

#==============================================================================================
def calculate_weight(f, max_doc_freq, num_docs, doc_freq, weighting_method):
    '''
    Calculate TF-IDF for a document with one of two implementations
    Which implementation gets used is determined by the weighting_method parameter.
    '''

    if weighting_method == 0: #tfc
        return f*math.log(num_docs/doc_freq, 10)
    else: #txc
        return f
    
#==============================================================================================
def calculate_query_weight(f, max_query_freq, num_docs, doc_freq, weighting_method):
    '''
    Calculate TF-IDF for a query with one of two implementations.
    Which implementation gets used is determined by the weighting_method parameter.
    '''

    return (0.5 + 0.5*f/max_query_freq)*math.log(num_docs/doc_freq, 10) 


#==============================================================================================

def search(index: dict, doc_norms: dict, max_doc_freq: dict, query: str, num_results: int, weighting_method: int) -> list:
    '''
    Search using the Vector Space Model.

    Parameters:
        - index: The inverted file returned by write_index
        - doc_norms: The norms of all documents, returned by write_index
        - max_doc_freq: The frequency of the most frequent term for each document, returned by write_index
        - query: ...query
        - num_results: The number of document that the model should return
        - weighting_method: Which implementation of TF-IDF weights should be used (0 or 1)

    Returns:
        - A list with the retrieved documents' IDs, sorted in descending order of similarity score
    '''
    
    result_list = []

    #Remove special characters from the query
    remove_chars = str.maketrans('', '', '?,()')
    query = query.translate(remove_chars).strip().upper().split()

    term_set = set(filter(lambda term: term in index, query))
    #print(term_set)

    #Get max term frequency in query
    max_query_freq = 0

    for term in term_set:
        temp = query.count(term)

        if temp > max_query_freq:
            max_query_freq = temp

    #Get the similarity between the query and every document
    for doc in range(1, 1240):
    
        if doc not in doc_norms:
            continue

        similarity = 0

        #Only the terms in the query contribute to the similarity
        for term in term_set:

            #Query weight
            query_weight = calculate_query_weight(query.count(term), max_query_freq, len(doc_norms.keys()), index[term][0], weighting_method)

            #Document weight
            #=======================================================
            doc_weight = 0

            if term in index:

                doc_weight = calculate_weight(freq(index, term, doc), max_doc_freq[doc], len(doc_norms.keys()), index[term][0], weighting_method)

                #print(f"{term}: {doc_weight}")

            similarity += doc_weight*query_weight

        #Normalization
        similarity /= doc_norms[doc]

        #print(doc, similarity)

        result_list.append((similarity, doc))

    #Sort all documents by their similarity to the query
    result_list = sorted(result_list, key = lambda x: x[0], reverse=True)

    #Only return top k results, and only the doc ids (NOT their similarity scores)
    return list(map(lambda x: x[1], result_list[:num_results]))