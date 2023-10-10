from html2text import read_warc
from NER import ner
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from w2v_rank import EntityRank
from WSD import WSD
from multiprocessing.pool import ThreadPool
i = 0

def entity_ranking(ent, results, text):
    candidate = []
    re_url = []
    meaning = WSD(ent, text).LeskDisambiguisation()
    #if WSD algorithm can not find the meaning of the mention，
    # use the sentence it appears to replace
    if meaning == '':
        meaning = text
    for result in results["results"]["bindings"]:
        candidate.append(result['o']['value'])
        re_url.append(result['article']['value'])
    #print(candidate)
    if len(candidate) == 0:
        return
    if len(candidate) == 1:
        return (result["article"]["value"])
    else:
        e_r = EntityRank(meaning, candidate)
        id = e_r.cal()
    #print(re_url[id])

    return re_url[id]

def wiki_query(ent,text):
    # Function to modify the entity name if it is a multi-word entity
    def entity_check(ent):
        if len(ent.split(" ")) > 2:
            return ent.lower()
        else:
            return ent

    # Set the endpoint URL for the Wikidata SPARQL query service
    endpoint_url = "https://query.wikidata.org/sparql"

    # Modify the entity name if necessary
    ent = entity_check(ent)

    # SPARQL query to find the Wikipedia article related to the given entity
    #the return includes url, label, description ...
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?article ?wikipediaLabel ?s ?o WHERE {{
    ?s rdfs:label "{}"@en;
        schema:description ?o.
    FILTER (lang(?o) = "en").
    ?article schema:about ?s .     
    ?article schema:isPartOf <https://en.wikipedia.org/>; 
        schema:name ?wikipediaLabel  }}""".format(str(ent))

    # Send the query and get the results
    def get_results(endpoint_url, query):
        user_agent = "WDPS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        # TODO adjust user agent; see https://w.wiki/CX6
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

    # Return the URL of the Wikipedia article if it was found, otherwise return None
    results = get_results(endpoint_url, query)
    return entity_ranking(ent, results, text)

    #return

def entity_linking(entity,text):
    # Function to link the given entities to Wikipedia articles
    l = wiki_query(entity[0], str(text))
    if l:
        return l
    return

'''
    #if run this file independently， please use the following code
    #to replace the current code in function entity_linking
        l = []
        for e,_ in entity:
            if wiki_query(e,text):
                l.append((e,wiki_query(e, text)))
        return l
'''


# Function to modify the label if it is a multi-word label
def label_process(label):
    if len(label.split(" "))>1:
        return label.title()
    return label


#multi-process
def run(tup):
    entity = ner(tup[1])
    #print(len(entity))
    global i
    i += 1
    print(i)
    result = entity_linking(entity, tup[1])
    for label, wikipeida_uri in result:
        print('Entity:' + '\t' + tup[0] + '\t' + label_process(label) + '\t' + f"<{wikipeida_uri}>")

if __name__ == '__main__':
    pool_size = 10
    tup_list = [(key,text) for key, text in read_warc("data/sample.warc.gz")]
    tup_list = tup_list[:10]
    #key, text = read_warc("data/sample.warc.gz")
    pool = ThreadPool(pool_size)
    pool.map(run,tup_list)
    pool.close()
    pool.join()