from candidate_generation import entity_search

import json
import pandas as pd


def read_ent_desc():

    # TODO[グ fix hard-coded paths
    ent_desc_file = "/data/prabhakar/CG/CONLL-AIDA/ent2desc.json"
    print("---Reading Entity Description File ---", ent_desc_file)
    with open(ent_desc_file, "r") as read_file:
        ent_desc = json.load(read_file)
        entities = list(ent_desc.keys())
    return ent_desc, entities


def get_ent_desc(wiki_label, cand_ent_desc):
    try:
        cand_ent = wiki_label.strip().replace(" ", "_")
        entity_description = " ".join(cand_ent_desc[cand_ent])
    except:
        return "NA"
    return entity_description


def load_dca_candidates(dataset):

    # TODO[グ fix hard-coded paths
    local_cand_file = "/data/prabhakar/CG/DCA/" + dataset + ".tsv"
    print("--- Loading Candidate File ---", local_cand_file)
    df_candidates = pd.read_csv(local_cand_file, sep='\t', encoding='utf-8', usecols=['mention', 'gold', 'candidates',
                                                                                      'context', 'mtype'])
    return df_candidates


def search_dca_candidates(mention, df_candidates):
    candidates = []
    mention = mention.rstrip(' ')
    df_match = df_candidates['candidates'][(df_candidates.mention == mention)].head(1)
    try:
        if df_match.empty is not True:
            candidates = [row.split(' | ') for row in df_match]
        if len(candidates[0]) >= 5:
            return candidates[0][0:5]
        return candidates[0]
    except:
        return "NA"


def process_dataset(df, size, local_candidates_flag, dataset):
    # Load the dataset into a pandas dataframe.
    df = df.dropna()
    df = df.rename(
        columns={"Sentence": "Sentence",
                 "predictedEnt": "EntitySep",
                 "Uri": "Qid",
                 "WikiTitle": "WikiLabel",
                 "Entity": "TargetEnt"}
    )

    df['label'] = int(1)

    df_target = pd.DataFrame()
    df_final = pd.DataFrame()

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    print('df', df.columns)

    target_entity_mentions = [row['TargetEnt'].split('EntityMentionSEP') for index, row in df.iterrows()]
    target_qids = [row['Qid'].split() for index, row in df.iterrows()]
    target_wiki_labels = [row['WikiLabel'].split('WikiLabelSEP') for index, row in df.iterrows()]

    df_candidates = load_dca_candidates(dataset)

    ent = 0

    for i in range(0, df.shape[0]):
        for j in range(0, len(target_wiki_labels[i])-1):
            ent += 1
            print('Sentence - ', i+1, ', Entity - ', j+1, ', Total_Entity - ', ent)
            pos_flag = False

            if target_wiki_labels[i][j] != '':
                cg_query = target_entity_mentions[i][j]

                if local_candidates_flag:
                    cg_result = search_dca_candidates(cg_query, df_candidates)
                    cg_result_set = set(cg_result)
                else:
                    cg_result = entity_search(cg_query, size)
                    cg_result_set = set(tuple(x) for x in cg_result)

                df_intermediate = pd.DataFrame(columns=['sequence1', 'sequence2', 'label'])
                df_target_intermediate = pd.DataFrame()

                seq1 = target_entity_mentions[i][j] + ' | ' + df.iloc[i]['Sentence']

                if j < len(target_entity_mentions[i])-1:
                    df_tgt = df_target_intermediate.append(
                        {'EntityMention': target_entity_mentions[i][j],
                         'Sentence': df.iloc[i]['Sentence'],
                         'Target_Qid': target_qids[i][j],
                         'Target_Wikilabel': target_wiki_labels[i][j],
                         'label': int(1)
                         }, ignore_index=True)
                    df_target = df_target.append(df_tgt)

                for r, result in enumerate(cg_result_set):
                    if local_candidates_flag:
                        result_wikilabel = result
                    else:
                        result_wikilabel = result[0]

                    pred_seq2 = result_wikilabel.replace('_', ' ')

                    if result_wikilabel != target_wiki_labels[i][j]:
                        if not pos_flag:
                            pos_flag = True
                        df_cg = df_intermediate.append(
                            {'sequence1': seq1, 'sequence2': pred_seq2, 'label': int(0)},
                            ignore_index=True
                        )

                        df_final = df_final.append(df_cg)

    return df_final, df_target


if __name__ == '__main__':
    dataset_input = "msnbc"
    predict_data_type = "data_full/Zeroshot/"

    # TODO[グ fix hard-coded paths
    data_dir = "/data/prabhakar/CG/WNED/" + dataset_input + "/prediction_data/" + predict_data_type

    df_input = pd.read_csv(
        data_dir + "ner_data.tsv",
        sep='\t',
        encoding='utf-8',
        usecols=['Sentence', 'Entity', 'Uri', 'WikiTitle', 'predictedEnt']
    )

    df_final_output, df_target_output = process_dataset(df_input, 5, False, dataset_input)
    df_final_output.to_csv(data_dir + "ned_data.tsv", index=False, sep="\t")
    df_target_output.to_csv(data_dir + "ned_target_data.tsv", index=False, sep="\t")
