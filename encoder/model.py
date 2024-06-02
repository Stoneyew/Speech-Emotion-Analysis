import torch
import torch.nn as nn
from transformers import BertModel

class BertEncoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
        pooled_output = last_hidden_state[:, 0, :]  # Use the [CLS] token's representation
        return pooled_output.float()  # Ensure outputs are float
