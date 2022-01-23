from bert_ner import Ner
from predict_ned import *
from predict_ner import predict_ner
from post_processing import *
from NED_Dataset_Creation import *
from evaluation import el_evaluation

predict_data_size = "10000"
predict_data_dir = "/data/prabhakar/CG/prediction_data/data_" + predict_data_size + "/"
ned_model_dir = "/data/prabhakar/CG/NED_pretrained/model_data_50000/"
ner_model_dir = "/data/prabhakar/manoj/code/NER/BERT-NER/pretrained_TREX_models/out_base_2/"


def ner():
    # Load a trained NED_old model and vocabulary that you have fine-tuned
    ner_model = Ner(ner_model_dir)
    df = pd.read_csv(predict_data_dir + "To_predict.tsv", sep='\t', encoding='utf-8',
                     usecols=['sequence1', 'sequence2Sep', 'uri', 'uriSequence2'])
    df_ner = predict_ner(ner_model, df)
    df_ner.to_csv(predict_data_dir + "ner_data.tsv", index=False, sep="\t")

    return df_ner


def ned(df_ned):

    dataset_cols = ['EntityMention', 'Sentence', 'Predicted_Wikilabel', 'Predicted_Qid', 'predictedLabels']

    # Load a trained NED_old model and vocabulary that you have fine-tuned
    ned_model = BertForSequenceClassification.from_pretrained(ned_model_dir)
    tokenizer = BertTokenizer.from_pretrained(ned_model_dir)

    prediction_dataloader, encoded_sequence = create_prediction_data(tokenizer, df_ned)
    predicted_labels, true_labels = test(ned_model, prediction_dataloader)

    df_predicted = pd.DataFrame(df_ned)
    df_predicted['predictedLabels'] = predicted_labels
    df_predicted_split, df_predicted_split_1, df_predicted_split_0 = process_sequence(df_predicted)

    df_predicted_split.to_csv(
        predict_data_dir + "predicted_data.tsv",
        index=False,
        sep="\t",
        columns=dataset_cols
    )
    df_predicted_split_0.to_csv(
        predict_data_dir + "predicted_data_0.tsv",
        index=False,
        sep="\t",
        columns=dataset_cols
    )
    df_predicted_split_1.to_csv(
        predict_data_dir + "predicted_data_1.tsv",
        index=False,
        sep="\t",
        columns=dataset_cols
    )

    return df_predicted_split_1


if __name__ == '__main__':

    # Evaluation
    df_target_input = pd.read_csv(predict_data_dir + "ned_target_data.tsv", sep="\t", encoding='utf-8')
    df_predicted_input = pd.read_csv(predict_data_dir + "predicted_data_1.tsv", sep="\t", encoding='utf-8')
    target_qids_list, predicted_qids_list = combine_sentence(df_target_input, df_predicted_input, predict_data_dir)

    precision, recall, fscore = el_evaluation(target_qids_list, predicted_qids_list)

    print("### Evaluation Results ###")
    print("Precision = ", precision, "\nRecall = ", recall, "\nF-Score = ", fscore)
