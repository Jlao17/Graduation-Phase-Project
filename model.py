from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import (RobertaConfig, RobertaModel)
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
# config_path = Path('../Dstill/tiny_bert_config.json')
# student_config = RobertaConfig.from_json_file(config_path)
# student_config.num_labels = 2
# codeBert = RobertaModel(student_config)
# codeBert.load_state_dict(torch.load('../Dstill/saved_models/gs5205.pkl'))
#

class Multi_Model(nn.Module):
    def __init__(self, text_encoder, code_encoder, text_hidden_size, code_hidden_size):
        super(Multi_Model, self).__init__()

        self.code_encoder = code_encoder  # Pre-trained CodeBERT encoder
        self.text_encoder = text_encoder
        self.similarity = nn.CosineSimilarity(dim=1)

        for param in self.text_encoder.parameters():
            param.requires_grad = True

        for param in self.code_encoder.parameters():
            param.requires_grad = True

        # Shared components
        self.norm = nn.LayerNorm(1536)
        self.act_layer_gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.6)

        # Task-specific layers for Issue-Commit
        self.issue_commit_dense1 = nn.Linear(text_hidden_size + code_hidden_size, int((text_hidden_size + code_hidden_size) / 2))
        self.issue_commit_dense2 = nn.Linear(int((text_hidden_size + code_hidden_size) / 2), int((text_hidden_size + code_hidden_size) / 4))
        self.issue_commit_output_layer = nn.Linear(int((text_hidden_size + code_hidden_size) / 4), 2)
        self.issue_commit_bn1 = nn.BatchNorm1d(768)
        self.issue_commit_bn2 = nn.BatchNorm1d(384)
        # Task-specific layers for Release-Issue
        self.release_issue_dense1 = nn.Linear(text_hidden_size + code_hidden_size, int((text_hidden_size + code_hidden_size) / 2))
        self.release_issue_dense2 = nn.Linear(int((text_hidden_size + code_hidden_size) / 2), int((text_hidden_size + code_hidden_size) / 4))
        self.release_issue_output_layer = nn.Linear(int((text_hidden_size + code_hidden_size) / 4), 2)

    def forward(
            self,
            issue_commit_inputs=None,
            issue_code_inputs=None,
            release_issue_inputs=None,
            release_description_inputs=None,
            issue_commit_target=None,
            release_issue_target=None,
            issue_commit_embeds=None,
            code_embeds=None,
            release_embeds=None,
            description_embeds=None,
            mode='train'
    ):
        # Handle issue-commit inputs

        if issue_commit_embeds is not None and code_embeds is not None:
            text_embedding = \
            self.text_encoder(inputs_embeds=issue_commit_embeds, attention_mask=(issue_commit_embeds.sum(-1) != 0))[1]
            code_embedding = self.code_encoder(inputs_embeds=code_embeds, attention_mask=(code_embeds.sum(-1) != 0))[1]
        else:
            text_embedding = self.text_encoder(input_ids=issue_commit_inputs, attention_mask=issue_code_inputs.ne(1))[1]
            code_embedding = self.code_encoder(input_ids=issue_code_inputs, attention_mask=issue_code_inputs.ne(1))[1]

        concatenated_issue_commit = torch.cat([text_embedding, code_embedding], dim=-1)

        # Handle release-issue inputs
        if release_embeds is not None and description_embeds is not None:
            release_embedding = \
            self.text_encoder(inputs_embeds=release_embeds, attention_mask=(release_embeds.sum(-1) != 0))[1]
            description_embedding = \
            self.text_encoder(inputs_embeds=description_embeds, attention_mask=(description_embeds.sum(-1) != 0))[1]
        else:
            release_embedding = \
            self.text_encoder(input_ids=release_issue_inputs, attention_mask=release_description_inputs.ne(1))[1]
            description_embedding = \
            self.text_encoder(input_ids=release_description_inputs, attention_mask=release_description_inputs.ne(1))[1]

        concatenated_release_issue = torch.cat([release_embedding, description_embedding], dim=-1)

        # Feedforward layers (unchanged)
        x1 = self.issue_commit_dense1(concatenated_issue_commit)
        x1 = self.dropout1(x1)
        x1 = self.issue_commit_dense2(x1)
        x1 = self.dropout(x1)
        logits_issue_commit = self.issue_commit_output_layer(x1)
        prob_issue_commit = torch.softmax(logits_issue_commit, dim=-1)

        x2 = self.release_issue_dense1(concatenated_release_issue)
        x2 = self.dropout1(x2)
        x2 = self.release_issue_dense2(x2)
        x2 = self.dropout(x2)
        logits_release_issue = self.release_issue_output_layer(x2)
        prob_release_issue = torch.softmax(logits_release_issue, dim=-1)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        issue_commit_loss = loss_fct(logits_issue_commit, issue_commit_target)
        release_issue_loss = loss_fct(logits_release_issue, release_issue_target)
        total_loss = issue_commit_loss + release_issue_loss

        if mode == 'eval':
            return total_loss, logits_issue_commit, logits_release_issue, prob_issue_commit, prob_release_issue
        elif mode == 'train':
            return total_loss
        else:
            raise ValueError(f"Invalid mode: {mode}")
