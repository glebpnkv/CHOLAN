"""Candidate Generation"""

from elasticsearch import Elasticsearch
from natsort import natsorted

import requests
import re


def initialization():
    es = Elasticsearch(['http://localhost:9200/'])
    es.count()
    return es


def entity_search(query, size):
    es = initialization()
    index_name = "wikidataentityindex"
    match = []

    body = {
        "query": {
            "match": {
                "label": query,
            }
        },
        "size": size
    }

    elastic_results = es.search(index=index_name, body=body)
    print("%d Hits :" % elastic_results['hits']['total'])

    for result in elastic_results['hits']['hits']:
        match.append([result["_source"]["label"], result["_source"]["uri"].strip("<http://www.wikidata.org/entity/>")])

    return match


def fuzzy_search(query, size):
    es = initialization()
    index_name = "wikidataentityindex"
    results = []
    body = {
        "query": {
            "fuzzy": {"label": query}
        },
        "size": size
    }

    elastic_results = es.search(index=index_name, body=body)

    for result in elastic_results['hits']['hits']:
        if result["_source"]["label"].lower() == query.replace(" ", "_").lower():
            results.append([result["_source"]["label"], result["_source"]["uri"]])
        else:
            results.append([result["_source"]["label"], result["_source"]["uri"]])
    return results


def falcon_search(query):
    my_json = {"text": query, "spans": []}
    try:
        wikidata_uri_list = requests.post("https://labs.tib.eu/falcon/falcon2/api?mode=short&k=20", json=my_json).json()
        wikidata_uri = [uri[0].strip("<http://www.wikidata.org/entity/>") for uri in wikidata_uri_list["entities"]]
        return wikidata_uri
    except:
        print("No response")
        return ''


def sort(sub_list):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used

    sub_list_sorted = natsorted(sub_list, key=lambda x: x[1].strip("Q"))
    return sub_list_sorted


def srt(val):
    """split and sort"""
    old = val.split(", ")
    new = ["{}{:0>2.0f}".format(i[0], int(i[1:])) for i in old]
    new.sort()
    out = ", ".join([i for i in new])
    return out


def convert(text: str):

    out = int(text) if text.isdigit() else text

    return out


def alphanum_key(key):

    out = [convert(c) for c in re.split('([0-9]+)', key)]

    return out


def sorted_nicely(input_list):
    """
    Sorts the given iterable in the way that is expected.

    Required arguments:
    input_list -- The iterable to be sorted.
    """

    return sorted(input_list, key=alphanum_key)


def append_list(final_list, last_list):
    for i in last_list:
        final_list.append(i)
    return final_list


if __name__ == '__main__':

    # "Kuleshov"
    query_input = "India"
    size_input = 40

    # Normal Match
    match_input = entity_search(query_input, size_input)
    matchFirstList = match_input[:5]
    matchLastList = match_input[10:]
    matchLastListSorted = sort(matchLastList)

    matchFinalList = matchFirstList
    matchFinalList = append_list(matchFinalList, matchLastListSorted[:5])

    print("\n### Normal Match - Possible Candidate Items for ", query_input, "###")
    for i, x in enumerate(matchFinalList):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    # Fuzzy Search
    match_input = fuzzy_search(query_input, size_input)
    print("\n### Fuzzy Search - Possible Candidate Items for ", query_input, "###")
    for i, x in enumerate(match_input):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    result_output = falcon_search(query_input)
    print("\n### Falcon Search - Possible Candidate Items for ", query_input, "###")
    print(result_output)
