import pandas as pd
import numpy as np


def calculate_sent_length(sent):
    sent_tokens = str(sent.encode('utf-8')).split()
    return len(sent_tokens)


# filter sentence with length and prepareVocabList
def limit_sent_length():
    data_dir = "/data/prabhakar/manoj/arjun/dataset/Trex_tsv_1/"
    data = pd.read_csv(
        data_dir+'merged_465_entity_uri_entity_sep.tsv',
        sep='\t',
        encoding='utf-8',
        usecols=['docid', 'sequence1', 'sequence2Sep', 'uri', 'uriSequence2'])

    data = data.dropna(subset=['sequence1', 'sequence2Sep', 'uriSequence2'])
    print(data.info())
    data.head(2)

    data['SentenceLen'] = data['sequence1'].apply(lambda sent: calculate_sent_length(sent))
    data['EntityLen'] = data['sequence2Sep'].apply(lambda sent: calculate_sent_length(sent))
    data = data.drop_duplicates(subset=['sequence1'], keep='first')
    data = data.reset_index(drop=True)
    data.info()
    data.head(2)

    data_analysis = data[['SentenceLen', 'EntityLen']]
    print(data_analysis.describe())

    # to change
    data_ranged = data[(data['SentenceLen'] <= 25)]
    print(data_ranged.describe())

    brange = np.arange(1, 100, 10)
    print(brange[0:10])

    data_ranged[['SentenceLen', 'EntityLen']].plot.hist(bins=brange, histtype='bar', alpha=0.5)

    data_ranged = data_ranged.reset_index(drop=True)
    print(data_ranged.head(5))

    trex_data_final = pd.DataFrame(data_ranged)
    trex_data_final.rename(
        columns={"sequence1": "Sentence", "sequence2Sep": "EntitySep", "uri": "Qid", "uriSequence2": "WikiLabel"}
    )
    trex_data_final.to_csv(data_dir+'Trex_final_seq25.tsv', sep='\t', encoding='utf-8', index=False)
    # to change
    print(data_ranged.info)
    print(data_ranged[['SentenceLen', 'EntityLen']].describe())


'''if __name__ == '__main__':
    limitSentLength()'''