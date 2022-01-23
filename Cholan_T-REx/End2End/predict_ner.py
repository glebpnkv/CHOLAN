import pandas as pd


def predict_ner(model, df):
    final_entity_list = []
    predicted_entity_list = []

    df = pd.DataFrame(df)
    sentence = df.sequence1.values

    for sent in sentence:
        output = model.predict(sent)
        predicted_entity_list.append(output)

    iflag = True

    for item, prediction in enumerate(predicted_entity_list):
        print("Item ", item)
        entity_list = []
        for i in range(0, len(prediction)):
            if prediction[i][1] == 'B-ENT':
                if prediction[-1][1] == 'B-ENT' or prediction[i + 1][1] != 'I-ENT':
                    b_word = prediction[i][0]
                    combined_word = b_word + ' EntityMentionSEP'
                    entity_list.append(combined_word)
                elif prediction[i + 1][1] == 'I-ENT':
                    if prediction[-1][1] == 'I-ENT':
                        b_word = prediction[i][0]
                        i_word = prediction[i + 1][0]
                        combined_word = b_word + ' ' + i_word + ' EntityMentionSEP'
                        entity_list.append(combined_word)
                    elif prediction[i + 2][1] == 'I-ENT':
                        b_word = prediction[i][0]
                        i_word = prediction[i + 1][0]
                        combined_word = b_word + ' ' + i_word
                        entity_list.append(combined_word)
                    elif prediction[i + 2][1] != 'I-ENT':
                        b_word = prediction[i][0]
                        i_word = prediction[i+1][0]
                        combined_word = b_word + ' ' + i_word + ' EntityMentionSEP'
                        entity_list.append(combined_word)

            elif prediction[i - 1][1] != 'B-ENT' and prediction[i][1] == 'I-ENT' and iflag is True:
                i_word = prediction[i][0]
                if prediction[i + 1][1] == 'I-ENT':
                    combined_word = i_word + ' '
                    entity_list.append(combined_word)
                else:
                    combined_word = i_word + ' EntityMentionSEP'
                    iflag = False
                    entity_list.append(combined_word)
            else:
                pass

        final_entity_list.append(' '.join(entity_list))

    df['predictedEnt'] = final_entity_list

    return df
