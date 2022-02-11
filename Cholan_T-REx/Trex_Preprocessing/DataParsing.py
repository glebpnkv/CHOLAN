import pandas as pd
import multiprocessing as mp
import time
import os
from IPython.display import display
from DataSequence import limit_sent_length

pd.set_option('display.max_colwidth', -1)

# Initialization
print("--Data Parsing--")
# TODO[グ fix hard-coded paths
trex_path = '/data/prabhakar/manoj/arjun/dataset/Trex_raw/'

# TODO[グ fix hard-coded paths
entityData_Sep_dir = "/data/prabhakar/manoj/arjun/dataset/Trex_tsv_1/"
doc_id = 0

# TODO[グ fix hard-coded paths
df_input = pd.read_csv(
    '/data/prabhakar/manoj/arjun/dataset/entityData_Sep/' + 'WikidataLabel_clean.csv',
    encoding='utf-8',
    header=None,
    names=['qValue', 'entity'],
    sep='\t'
)
qDict = dict(zip(df_input.qValue, df_input.entity))


def parse_doc(docid, sentences, entity_json):
    # global df_file
    global doc_id
    entity_list_dict = {entity['boundaries'][0]: entity['surfaceform'] for entity in entity_json
                        if entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker'
                        }
    entity_list_dict_sep = {entity['boundaries'][0]: entity['surfaceform'] + ' EntityMentionSEP'
                            for entity in entity_json
                            if entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker'
                            }
    entity_list_dict_uri = {entity['boundaries'][0]: entity['uri']
                            for entity in entity_json
                            if entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker'
                            }
    entity_list = [entity_list_dict[key].strip() for key in sorted(entity_list_dict.keys())]
    entities = ' '.join(entity for entity in entity_list)

    entity_list_sep = [entity_list_dict_sep[key].strip() for key in sorted(entity_list_dict_sep.keys())]
    entities_sep = ' '.join(entity for entity in entity_list_sep)

    entity_list_uri = [entity_list_dict_uri[key].replace('http://www.wikidata.org/entity/', '').strip() for key in
                       sorted(entity_list_dict_uri.keys())]
    entities_uri = ' '.join(entity for entity in entity_list_uri)

    df = pd.DataFrame()
    entity_uri_entity = None

    if len(entity_list) == len(entity_list_uri):
        entity_uri_entity = [(entity.strip(), entityUri.strip())
                             for entity, entityUri in zip(entity_list, entity_list_uri)]
        d = {'docid': docid.replace('http://www.wikidata.org/entity/', '').strip(), 'sequence1': sentences,
             'sequence2': entities, 'sequence2Sep': entities_sep, 'uri': entities_uri}
        df = pd.DataFrame(data=d, index=[0])
        df = df.fillna('')
    else:
        #         pass
        doc_id = doc_id + 1
    return df, entity_uri_entity


def parser_file(filename):
    df = pd.read_json(filename, encoding='utf-8')
    df = df[[u'docid', u'text', u'entities', u'sentences_boundaries']]
    df_sfile = pd.DataFrame()
    #     df.info()
    #     df.head()

    file_entity_uri_entity = []

    for i in range(len(df.index)):
        df_d, entity_uri_entity = parse_doc(df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2])
        file_entity_uri_entity = file_entity_uri_entity + entity_uri_entity
        df_sfile = df_sfile.append(df_d, ignore_index=True)
    return df_sfile, file_entity_uri_entity


def convert_uri_qvalue_to_entity(uris):
    str_list = uris.strip().split()
    # print(uris[1])
    str_list = [str(qDict[qValue]) + ' WikiLabelSEP' if qValue in qDict else 'NIL WikiLabelSEP' for qValue in str_list]
    # print(strList[1])
    uris = ' '.join(str_list)
    # print(uris[1])
    return uris


def main():
    df_file = pd.DataFrame()
    url_entity_dict = []

    files_list = sorted(os.listdir(trex_path))
    pool = mp.Pool(28)

    files_20 = [trex_path + file for file in files_list]
    print('\n\nNumber of JSON files = ', len(files_20))
    print(files_20)

    for result in pool.map(parser_file, files_20):
        start_time = time.time()
        print(len(result))
        df_file = df_file.append(result[0], ignore_index=True)
        elapsed_time = time.time() - start_time

        # url_entity_dict = merge_two_dicts(url_entity_dict, result[1])
        url_entity_dict = url_entity_dict + result[1]
        print("TimeTakenByFileParsing", elapsed_time)
        del result

    pool.close()
    pool.join()

    df_file.info()
    df_file.head(2)

    df_file = df_file.dropna()
    df_file = df_file.drop_duplicates(keep='first')
    df_file = df_file.reset_index(drop=True)
    df_file.to_csv(entityData_Sep_dir + 'merged_465_entity_uri_sep' + '.tsv', sep='\t', encoding='utf-8', index=False)
    print(df_file.info())

    df_entity_uri = pd.DataFrame.from_records(url_entity_dict, columns=['Surface-Form', 'QUri'])
    df_entity_uri = df_entity_uri.dropna()
    df_entity_uri = df_entity_uri.drop_duplicates(keep='first')
    df_entity_uri = df_entity_uri.reset_index(drop=True)
    df_entity_uri.to_csv(entityData_Sep_dir + 'surfaceForm_wikiUri_sep' + '.tsv', sep='\t', encoding='utf-8',
                         index=False)
    print(df_entity_uri.info())

    del url_entity_dict
    del df_file
    del df_entity_uri

    # Replacing WikiUri with WikiDataEntity
    chunks = pd.read_csv(
        entityData_Sep_dir + 'merged_465_entity_uri_sep.tsv',
        encoding='utf-8',
        chunksize=100000,
        sep='\t')

    df = pd.DataFrame()
    ctr = 0
    for chunk in chunks:
        chunk = chunk.dropna()
        print("Chunk = ", len(chunk))
        chunk['uriSequence2'] = chunk['uri'].apply(convert_uri_qvalue_to_entity)
        df = df.append(chunk, ignore_index=True)
        print("WikiLabels = ", len(df['uriSequence2']))
        df.to_csv(
            entityData_Sep_dir + 'merged_465_entity_uri_entity_sep_' + str(ctr) + '_.tsv',
            encoding='utf-8',
            index=False,
            sep='\t')

    print(df.info())

    df.to_csv(entityData_Sep_dir + 'merged_465_entity_uri_entity_sep.tsv', encoding='utf-8', index=False, sep='\t')

    print(df.info())
    display(df.head(2))

    del df
    del chunks

    limit_sent_length()


if __name__ == '__main__':
    main()
