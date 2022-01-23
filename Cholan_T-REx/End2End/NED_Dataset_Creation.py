from candidate_generation import entity_search

import pandas as pd


def process_dataset(df, size):

    # Load the dataset into a pandas dataframe.
    # df = pd.read_csv("/data/prabhakar/CG/Trex_data/Trex_train_50000.tsv",
    #                  sep='\t',
    #                  encoding='utf-8',
    #                  usecols=['sequence1', 'sequence2Sep', 'uri', 'uriSequence2'])

    df = df.dropna()
    df = df.rename(columns={"sequence1": "Sentence",
                            "predictedEnt": "EntitySep",
                            "uri": "Qid",
                            "uriSequence2": "WikiLabel",
                            "sequence2Sep": "TargetEnt"
                            })
    df['label'] = int(1)

    df_target = pd.DataFrame()
    df_final = pd.DataFrame()

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    print('df', df.columns)

    # wiki_label_search = ''
    target_entity_mentions = [row['TargetEnt'].split('EntityMentionSEP') for index, row in df.iterrows()]
    target_qids = [row['Qid'].split() for index, row in df.iterrows()]
    target_wiki_labels = [row['WikiLabel'].split('WikiLabelSEP') for index, row in df.iterrows()]
    predicted_entity_mentions = [row['EntitySep'].split('EntityMentionSEP') for index, row in df.iterrows()]

    # for index, row in df.iterrows():
    #     print("Sentence - ", index)
    #     predicted_entity_mentions = row['EntitySep'].split('EntityMentionSEP')

    ent = 0

    for i in range(0, df.shape[0]):
        for j in range(0, len(predicted_entity_mentions[i])-1):
            ent += 1
            print('Sentence - ', i+1, ', Entity - ', j+1, ', Total_Entity - ', ent)
            # POS_FLAG = False

            cg_query = predicted_entity_mentions[i][j]
            cg_result = entity_search(cg_query, size)
            cg_result_set = set(tuple(x) for x in cg_result)

            df_target_intermediate = pd.DataFrame()
            seq1 = predicted_entity_mentions[i][j] + ' | ' + df.iloc[i]['Sentence']
            # target_seq2 = target_wiki_labels[i][j] + ' | ' + target_qids[i][j]
            # target_sequence = seq1 + ' | ' + target_seq2
            if j < len(target_entity_mentions[i])-1:
                df_tgt = df_target_intermediate.append(
                    {'EntityMention': target_entity_mentions[i][j],
                     'Sentence': df.iloc[i]['Sentence'],
                     'Target_Qid': target_qids[i][j],
                     'Target_Wikilabel': target_wiki_labels[i][j],
                     'label': int(1)},
                    ignore_index=True
                )
                df_target = df_target.append(df_tgt)

            for r, result in enumerate(cg_result_set):
                df_intermediate = pd.DataFrame(columns=['sequence', 'label'])
                result_wikilabel = result[0]
                result_qid = result[1].strip("<http://www.wikidata.org/entity/>")

                # if result_qid
                pred_seq2 = result_wikilabel + ' | ' + result_qid
                pred_sequence = seq1 + ' | ' + pred_seq2
                df_cg = df_intermediate.append({'sequence': pred_sequence, 'label': int(0)}, ignore_index=True)
                df_final = df_final.append(df_cg)

                # if (result_qid != target_qids[i][j] and POS_FLAG == False):
                #     sequence = seq1 + ' | ' + pos_seq2
                #     df_pos = df_intermediate.append({'sequence' : sequence, 'label' : int(1)}, ignore_index=True)
                #     df_final = df_final.append(df_pos)
                #     POS_FLAG=True
                # elif(result_qid != target_qids[i][j]):
                #     sequence = seq1 + ' | ' + neg_seq2
                #     df_neg = df_intermediate.append({'sequence': sequence, 'label': int(0)}, ignore_index=True)
                #     df_final = df_final.append(df_neg)

    return df_final, df_target


if __name__ == '__main__':
    data_dir = "/data/prabhakar/CG/prediction_data/data_10000/"
    df_input = pd.read_csv(data_dir + "ner_data.tsv",
                           sep='\t',
                           encoding='utf-8',
                           usecols=['sequence1', 'sequence2Sep', 'uri', 'uriSequence2', 'predictedEnt']
                           )

    df_final_output, df_target_output = process_dataset(df_input, 30)
    df_final_output.to_csv(data_dir + "ned_data.tsv", index=False, sep="\t")
    df_target_output.to_csv(data_dir + "ned_target_data.tsv", index=False, sep="\t")
