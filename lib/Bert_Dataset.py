import torch
from torch.utils.data import Dataset


class Bert_Dataset(Dataset):

    def __init__(self, data, tokenizer, data_collator):
        self.data = data
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sentence = self.data[i]

        inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt", max_length=15, padding='max_length', truncation=True)

        batch_encoding = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        masked_inputs = self.data_collator([batch_encoding[0]])
        input_ids = masked_inputs['input_ids']
        labels = masked_inputs['labels']

        return input_ids[0].numpy(), attention_mask[0].numpy(), labels[0].numpy()
