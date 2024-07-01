import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling


class PretrainBertForMLM(nn.Module):
    def __init__(self, bert_config, bert_model_path=None, device="cpu"):
        super(PretrainBertForMLM, self).__init__()

        self.device = device
        self.bert_config = bert_config
        if bert_model_path != None:
            self.bert_model = BertForMaskedLM.from_pretrained(bert_model_path, config=self.bert_config, ignore_mismatched_sizes=True)
        else:
            self.bert_model = BertForMaskedLM(config=self.bert_config, ignore_mismatched_sizes=True)


    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs
    

    def save_training_records(self, text, save_path):
        f = open(save_path + f"bert_pretrain_record_{datetime.now().date()}.txt", mode='a')
        f.write(text)
        f.close()


    def pre_training(self, bert_data_loader, args):
        model_save_base_path = args['model_save_base_path']
        train_records_save_base_path = args['train_records_save_base_path']

        optimizer = torch.optim.AdamW(self.parameters(), lr=args['lr'])
        num_epochs = args['epochs']

        self.save_training_records("Start pre-training...\n", save_path=train_records_save_base_path)
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(bert_data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(bert_data_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

            self.save_training_records(f'Time {datetime.now()}, Epoch {epoch + 1}/{num_epochs}, Loss {avg_loss:.4f}\n', save_path=train_records_save_base_path)

            if (epoch + 1) % 5 == 0:
                self.bert_model.save_pretrained(model_save_base_path + f"bert_mlm_{str(epoch+1)}")

        

