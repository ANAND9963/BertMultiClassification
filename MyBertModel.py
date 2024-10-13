import torch
from torch import nn
from transformers import BertModel, AutoModel, RobertaModel
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

class Classifier(nn.Module):  # Classifier to be attached to the last part of the model
    def __init__(self, bert_embedding, num_classes):
        super(Classifier, self).__init__()
        hidden_layer = 100
        self.fc1 = nn.Linear(bert_embedding, hidden_layer)
        self.dropout1 = nn.Dropout(0.2)
        self.act1 = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.ident = nn.Identity()

    def forward(self, encoded_input):
        out1 = self.dropout1(self.fc1(encoded_input))
        out2 = self.act1(out1)
        logits = self.fc2(out2)
        probabilities = self.softmax(logits)
        return probabilities

class MyBertModel(nn.Module):  # Pretrained BERT + our own classifier
    def __init__(self, dropout_probability=0.2, use_dropout=True):
        super(MyBertModel, self).__init__()
        # self.bert_model = BertModel.from_pretrained('bert-base-cased')
        # self.bert_model = AutoModel.from_pretrained("distilbert/distilbert-base-cased")
        self.bert_model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        print(self.bert_model)
        hidden_size = self.bert_model.config.hidden_size
        num_classes = 6
        self.classifier = Classifier(hidden_size, num_classes)

        # Freeze parameters
        modules = [self.bert_model.embeddings, self.bert_model.encoder.layer[:5]]
        for module in modules:  # Freeze first 5 encoder layers
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_id, mask):
        pooled_output = self.bert_model(input_ids=input_id, attention_mask=mask)
        out = self.classifier(pooled_output[0][:, 0, :])
        return out
