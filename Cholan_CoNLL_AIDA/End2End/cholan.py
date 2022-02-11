from transformers import BertTokenizer, BertForSequenceClassification

from bert_ner import Ner
from predict_ned import create_prediction_data
from predict_ner import predict_NER
from post_processing import combine_sentence, process_sequence
from test_model import test
from evaluation import strong_matching

import pandas as pd

dataset = "msnbc"
data_type = "Zeroshot/"
predict_data_type = "data_full/" + data_type

# TODO[グ fix hard-coded paths
predict_data_dir = "/data/prabhakar/CG/WNED/" + dataset + "/prediction_data/" + predict_data_type

# TODO[グ fix hard-coded paths
ned_model_dir = "/data/prabhakar/CG/NED_pretrained/model_data_50000/"
ner_model_dir = "/data/prabhakar/manoj/code/NER/BERT-NER-CoNLL/pretrained_ner/"


def ner():
    # Load a trained NED_old model and vocabulary that you have fine-tuned
    ner_model = Ner(ner_model_dir)
    df = pd.read_csv(predict_data_dir + "To_predict.tsv", sep='\t', encoding='utf-8',
                     usecols=['Sentence', 'Entity', 'Uri', 'WikiTitle'])
    df = df.dropna()
    df_ner = predict_NER(ner_model, df)
    df_ner.to_csv(predict_data_dir + "ner_data.tsv", index=False, sep="\t")

    return df_ner


def ned(df_ned):
    # Load a trained NED_old model and vocabulary that you have fine-tuned
    ned_model = BertForSequenceClassification.from_pretrained(ned_model_dir)
    tokenizer = BertTokenizer.from_pretrained(ned_model_dir)
    prediction_dataloader, encoded_sequence = create_prediction_data(tokenizer, df_ned)

    predicted_labels, true_labels = test(ned_model, prediction_dataloader)
    df_predicted = pd.DataFrame(df_ned)
    df_predicted['predictedLabels'] = predicted_labels
    df_predicted_split, df_predicted_split_1, df_predicted_split_0 = process_sequence(df_predicted)

    dataset_cols = ['EntityMention', 'Sentence', 'Predicted_Wikilabel', 'predictedLabels']

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

    # NED Prediction
    df_ned_input = pd.read_csv(
        predict_data_dir + "ned_data.tsv",
        sep='\t',
        encoding='utf-8',
        usecols=['sequence1', 'sequence2', 'label']
    )

    # Evaluation
    df_target = pd.read_csv(predict_data_dir + "ned_target_data.tsv", sep="\t", encoding='utf-8')
    df_predicted_output = pd.read_csv(predict_data_dir + "predicted_data_1.tsv", sep="\t", encoding='utf-8')
    target_wikilabel_list, predicted_wikilabel_list = combine_sentence(df_target, df_predicted_output, predict_data_dir)

    print("### Evaluation Results ###")

    precision, recall, fscore = strong_matching(target_wikilabel_list, predicted_wikilabel_list)
    print("--- Micro Scores - WikiLabel ---")
    print("Precision = %.1f" % precision, "\tRecall = %.1f" % recall, "\tF-Score = %.1f" % fscore)
