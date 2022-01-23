""" Candidate Generation """

from elasticsearch import Elasticsearch

import requests


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
        match.append([result["_source"]["label"], result["_source"]["uri"]])

    return match


def fuzzy_search(query):
    es = initialization()
    index_name = "wikidataentityindex"
    results = []
    body = {
        "query": {
            "fuzzy": {"label": query}
        }, "size": 10
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
        wikidata_uri_list = requests.post("https://labs.tib.eu/falcon/falcon2/api?mode=short&k=10", json=my_json).json()
        wikidata_uri = [uri[0].strip("<http://www.wikidata.org/entity/>") for uri in wikidata_uri_list["entities"]]
        return wikidata_uri
    except:
        print("No response")
        return ''


if __name__ == '__main__':

    # "Kuleshov"
    query_input = "GPLed"

    # Normal Match
    match_input = entity_search(query_input, 10)
    print("\n### Normal Match - Possible Candidate Items for ", query_input, "###")
    for i, x in enumerate(match_input):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    # Fuzzy Search
    match_input = fuzzy_search(query_input)
    print("\n### Fuzzy Search - Possible Candidate Items for ", query_input, "###")
    for i, x in enumerate(match_input):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    result_output = falcon_search(query_input)
    print("\n### Falcon Search - Possible Candidate Items for ", query_input, "###")
    print(result_output)
