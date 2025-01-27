from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

from test import test

import pandas as pd
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4


def create_prediction_data(tokenizer, df):
    sequences = df.sequence.values
    labels = df.label.values.astype(str).astype(int)
    input_ids = []
    attention_masks = []

    # For every sentence...
    for seq in sequences:
        encoded_seq = tokenizer.encode(seq, pad_to_max_length=True, add_special_tokens=True)
        input_ids.append(encoded_seq)

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    return prediction_dataloader, encoded_seq


if __name__ == '__main__':

    # TODO[グ fix hard-coded paths
    predict_dir = "/data/prabhakar/CG/prediction_data/"
    predict_data_dir = predict_dir + "data_10000/"

    # TODO[グ fix hard-coded paths
    output_dir = "/data/prabhakar/CG/NED_pretrained/without_localcontext/"

    df_input = pd.read_csv(predict_data_dir + "ned_data.tsv", encoding='utf-8', usecols=['sequence', 'label'], sep='\t')
    print('Number of test sentences: {:,}\n'.format(df_input.shape[0]))

    # Load a trained model and vocabulary that you have fine-tuned
    ned_model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    prediction_dataloader, encoded_sequence = create_prediction_data(tokenizer, df_input)

    predicted_labels, true_labels = test(ned_model, prediction_dataloader)

    original_sequence = tokenizer.decode(encoded_sequence)
    sequences = df_input.sequence.values
    labels = df_input.label.values

    df_predicted = pd.DataFrame(df_input)
    df_predicted['predictedLabels'] = predicted_labels
    df_predicted_0 = df_predicted[df_predicted['predictedLabels'] == 0]
    df_predicted_1 = df_predicted[df_predicted['predictedLabels'] == 1]

    df_predicted_0.to_csv(predict_data_dir + "predicted_data_0.tsv", index=False, sep="\t")
    df_predicted_1.to_csv(predict_data_dir + "predicted_data_1.tsv", index=False, sep="\t")
