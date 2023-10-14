import torch
import torch.nn as nn
import os
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

PRETRAINED_MODEL_ARCHIVE_MAP = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese'
)


# Based on the input model name, determine whether it's a pre-trained model (downloadable from the internet)
# or a folder name (read corresponding student models from the target folder).
def get_model(model_name_or_path, num, aug_num):
    # When the input model name is a folder name.
    if model_name_or_path not in PRETRAINED_MODEL_ARCHIVE_MAP:
        model_path = 'model_' + str(num)
        resolved_model_file = os.path.join(model_name_or_path, model_path)
    else:
        resolved_model_file = model_name_or_path

    # Check if the current model's data augmentation method is dropout; if not, deactivate BERT's built-in dropout.
    if (num == 0 and (aug_num - 1) % 4 == 2) or (num == 1 and (aug_num - 1) // 4 == 2):
        model = BertModel.from_pretrained(resolved_model_file)
    else:
        model = BertModel.from_pretrained(resolved_model_file,
                                          attention_probs_dropout_prob=0.0,
                                          hidden_dropout_prob=0.0)
    return model


# The model class used for TinyBert during General Distill.
class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, student_num=2, fit_size=768):
        super(TinyBertForPreTraining, self).__init__(config)
        self.student_num = student_num
        self.student = nn.ModuleList([BertModel(config) for _ in range(self.student_num)])
        self.fit_dense = nn.ModuleList([nn.Linear(config.hidden_size, fit_size) for _ in range(self.student_num)])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_hidden_states=False,
                output_attentions=False):
        # output=("last_hidden_state", "pooler_output", "all_hidden_states", "attentions")
        all_encoder_layers = []
        all_encoder_atts = []
        for i in range(self.student_num):
            output = self.student[i](input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     output_hidden_states=output_hidden_states,
                                     output_attentions=output_attentions)
            tmp = []
            # Expand the dimensions of all hidden states in the student model
            # using a fully connected layer to match the dimensions of the teacher model.
            for sequence_layer in output.hidden_states:
                tmp.append(self.fit_dense[i](sequence_layer))
            all_encoder_layers.append(tmp)
            all_encoder_atts.append(output.attentions)
        return all_encoder_atts, all_encoder_layers


# Head for the extractive summarization task.
class ExtractiveSummaryHead(nn.Module):
    def __init__(self, hidden_size):
        super(ExtractiveSummaryHead, self).__init__()
        self.encoder = nn.Linear(hidden_size, 1)

    def forward(self, bert_output, cls_mask):
        # Select hidden states corresponding to all [CLS] tokens in the sentence.
        flatten_enc_sent_embs = bert_output.last_hidden_state.masked_select(cls_mask)
        enc_sent_embs = flatten_enc_sent_embs.view(-1, bert_output.last_hidden_state.size(-1))

        logits = self.encoder(enc_sent_embs).view(-1)
        return logits


# Head for tasks in the GLUE dataset.
class GlueTaskHead(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, num_classes):
        super(GlueTaskHead, self).__init__()
        self.num_classes = num_classes
        self.encoder = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, bert_output):
        pooled_output = bert_output.pooled_output
        pooled_output = self.dropout(pooled_output)
        logits = self.encoder(pooled_output)
        return logits


'''
class GlueTaskHead(nn.Module):
    def __init__(self, hidden_size,hidden_dropout_prob, num_classes):
        super(GlueTaskHead, self).__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
                                    nn.Linear(hidden_size, 128),
                                     nn.Tanh(),
                                     nn.Linear(128, num_classes))

    def forward(self, bert_output):
        pooled_output = torch.mean(bert_output.last_hidden_state, 1)
        logits = self.encoder(pooled_output)
        return logits
'''


# Model class used in the DisCo mutual learning framework.
class ConsistSum(nn.Module):
    def __init__(self, model_name_or_path, task_name, student_num=2, num_classes=1, aug_num=11):
        super(ConsistSum, self).__init__()
        self.student_num = student_num
        self.extractive_summary = task_name == "sum"
        self.student = nn.ModuleList([get_model(model_name_or_path, num, aug_num)
                                      for num in range(self.student_num)])
        if self.extractive_summary:
            self.head = nn.ModuleList(
                [ExtractiveSummaryHead(self.student[num].config.hidden_size) for num in range(self.student_num)])
        else:
            self.head = nn.ModuleList([GlueTaskHead(self.student[num].config.hidden_size,
                                                    self.student[num].config.hidden_dropout_prob, num_classes) for num
                                       in range(self.student_num)])
        
        self.prediction_weight = nn.ParameterList(nn.Parameter(torch.tensor(1.0 / student_num, requires_grad=False))
                                                  for _ in range(self.student_num))

    def set_prediction_weight(self, last_loss):
        for i in range(self.student_num):
            self.prediction_weight[i] = nn.Parameter(last_loss[i].item(), requires_grad=False)

    def predict(self, batch, specific_student, ensemble_prediction=True):
        input_ids, attn_mask, seg, pos_ids = batch.input_ids, batch.attn_mask, batch.seg, batch.pos_ids
        cls_mask = batch.cls_mask.unsqueeze(-1) if self.extractive_summary else None

        # Using Bert to extract sentence features.
        ensemble_logit = 0.0
        if ensemble_prediction:
            for i in range(self.student_num):
                output = self.student[i](input_ids=input_ids,
                                         attention_mask=attn_mask,
                                         token_type_ids=seg,
                                         position_ids=pos_ids)

                if self.extractive_summary:
                    logits = self.head[i](output, cls_mask)
                else:
                    logits = self.head[i](output)
                ensemble_logit += self.prediction_weight[i].item() * logits
        else:
            output = self.student[specific_student](input_ids=input_ids,
                                                    attention_mask=attn_mask,
                                                    token_type_ids=seg,
                                                    position_ids=pos_ids)

            if self.extractive_summary:
                ensemble_logit = self.head[specific_student](output, cls_mask)
            else:
                ensemble_logit = self.head[specific_student](output)

        return ensemble_logit

    def forward(self, batch, specific_student=None, inputs_embeds=None, do_adversary=False):
        input_ids, attn_mask, seg, pos_ids = batch.input_ids, batch.attn_mask, batch.seg, batch.pos_ids
        cls_mask = batch.cls_mask.unsqueeze(-1) if self.extractive_summary else None

        # Using Bert to extract sentence features.
        all_logits = []
        all_embeds = []
        for i in range(self.student_num):
            if specific_student is None or i == specific_student:
                output = self.student[i](input_ids=input_ids if inputs_embeds is None else None,
                                         inputs_embeds=inputs_embeds,
                                         attention_mask=attn_mask,
                                         token_type_ids=seg,
                                         position_ids=pos_ids,
                                         output_hidden_states=do_adversary)

                if self.extractive_summary:
                    logits = self.head[i](output, cls_mask)
                else:
                    logits = self.head[i](output)
                all_logits.append(logits)
                # This is written in this way to facilitate adversarial augmentation for unlabeled data. It allows
                # obtaining the original logits as labels while also obtaining the original input embeddings.
                # However, for labeled data augmentation, where the original logits are not needed as labels,
                # it results in the model running the logits again unnecessarily.
                if do_adversary:
                    all_embeds.append(output.hidden_states[0])

        return all_logits, all_embeds
