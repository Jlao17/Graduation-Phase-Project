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
        self.release_issue_dense1 = nn.Linear(text_hidden_size + text_hidden_size, int((text_hidden_size + text_hidden_size) / 2))
        self.release_issue_dense2 = nn.Linear(int((text_hidden_size + text_hidden_size) / 2), int((text_hidden_size + text_hidden_size) / 4))
        self.release_issue_output_layer = nn.Linear(int((text_hidden_size + text_hidden_size) / 4), 2)


    def forward(self, issue_commit_inputs, issue_code_inputs, release_issue_inputs, release_description_inputs, issue_commit_target,
                release_issue_target, mode='train'):
        """
        Forward method for link prediction between issues, commits, and release notes.
        Args:
            commit_inputs: Commit inputs.
            issue_inputs: Issue inputs.
            release_inputs: Release note inputs.
            issue_commit_target: Labels for commit-issue links.
            release_issue_target: Labels for release-issue links.
            mode: 'train' or 'eval'.
        """
        #             loss = model(issue_commit_inputs, issue_code_inputs, release_issue_inputs, issue_commit_target, release_issue_target)
        # Shared encoding for all inputs
        class_weights = torch.tensor([0.3, 0.7]).to(issue_commit_target.device)

        text_embedding = self.text_encoder(issue_commit_inputs, attention_mask=issue_code_inputs.ne(1))[1]  # [batch_size, hiddensize]
        code_embedding = self.code_encoder(issue_code_inputs, attention_mask=issue_code_inputs.ne(1))[1]
        concatenated_issue_commit = torch.cat([text_embedding, code_embedding], dim=-1)

        release_embedding = self.text_encoder(release_issue_inputs, attention_mask=release_description_inputs.ne(1))[1]
        description_embedding = self.text_encoder(release_description_inputs, attention_mask=release_description_inputs.ne(1))[1]
        concatenated_release_issue = torch.cat([release_embedding, description_embedding], dim=-1)

        # Issue-Commit pathway
        x1 = self.issue_commit_dense1(concatenated_issue_commit)
        x1 = self.dropout1(x1)
        x1 = self.issue_commit_dense2(x1)
        x1 = self.dropout(x1)
        logits_issue_commit = self.issue_commit_output_layer(x1)
        prob_issue_commit = torch.softmax(logits_issue_commit, dim=-1)

        # Release-Issue pathway
        x2 = self.release_issue_dense1(concatenated_release_issue)
        x2 = self.dropout(x2)
        x2 = self.release_issue_dense2(x2)
        x2 = self.dropout(x2)
        logits_release_issue = self.release_issue_output_layer(x2)
        prob_release_issue = torch.softmax(logits_release_issue, dim=-1)

        # Compute losses if targets are provided

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        issue_commit_loss = loss_fct(logits_issue_commit, issue_commit_target)
        release_issue_loss = loss_fct(logits_release_issue, release_issue_target)
        total_loss = issue_commit_loss + release_issue_loss

        # Return logits and loss (if in training or evaluation mode)
        if mode == 'eval':
            return total_loss, prob_issue_commit, prob_release_issue
        elif mode == 'train':
            return total_loss
        else:
            raise ValueError(f"Invalid mode: {mode}")

class Multi_Model_Heavy(nn.Module):
    def __init__(self, text_encoder, code_encoder, text_hidden_size, code_hidden_size):
        super(Multi_Model_Heavy, self).__init__()

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
        self.dropout = nn.Dropout(0.3)
        self.dropout1 = nn.Dropout(0.4)

        # Task-specific layers for Issue-Commit
        self.issue_commit_dense1 = nn.Linear(text_hidden_size + code_hidden_size, int((text_hidden_size + code_hidden_size) / 2))
        self.issue_commit_dense2 = nn.Linear(int((text_hidden_size + code_hidden_size) / 2), int((text_hidden_size + code_hidden_size) / 4))
        self.issue_commit_output_layer = nn.Linear(int((text_hidden_size + code_hidden_size) / 4), 2)
        self.issue_commit_bn1 = nn.BatchNorm1d(768)
        self.issue_commit_bn2 = nn.BatchNorm1d(384)
        # Task-specific layers for Release-Issue
        self.release_issue_dense1 = nn.Linear(text_hidden_size + text_hidden_size, int((text_hidden_size + text_hidden_size) / 2))
        self.release_issue_dense2 = nn.Linear(int((text_hidden_size + text_hidden_size) / 2), int((text_hidden_size + text_hidden_size) / 4))
        self.release_issue_output_layer = nn.Linear(int((text_hidden_size + text_hidden_size) / 4), 2)

        self.shared_dense1 = nn.Linear(text_hidden_size + text_hidden_size, int((text_hidden_size + text_hidden_size) / 2))
        self.shared_dense2 = nn.Linear(int((text_hidden_size + text_hidden_size) / 2), int((text_hidden_size + text_hidden_size) / 4))

    def forward(self, issue_commit_inputs, issue_code_inputs, release_issue_inputs, release_description_inputs, issue_commit_target,
                release_issue_target, mode='train'):
        """
        Forward method for link prediction between issues, commits, and release notes.
        Args:
            commit_inputs: Commit inputs.
            issue_inputs: Issue inputs.
            release_inputs: Release note inputs.
            issue_commit_target: Labels for commit-issue links.
            release_issue_target: Labels for release-issue links.
            mode: 'train' or 'eval'.
        """
        #             loss = model(issue_commit_inputs, issue_code_inputs, release_issue_inputs, issue_commit_target, release_issue_target)
        # Shared encoding for all inputs
        text_embedding = self.text_encoder(issue_commit_inputs, attention_mask=issue_code_inputs.ne(1))[1]  # [batch_size, hiddensize]
        code_embedding = self.code_encoder(issue_code_inputs, attention_mask=issue_code_inputs.ne(1))[1]
        concatenated_issue_commit = torch.cat([text_embedding, code_embedding], dim=-1)

        release_embedding = self.text_encoder(release_issue_inputs, attention_mask=release_description_inputs.ne(1))[1]
        description_embedding = self.text_encoder(release_description_inputs, attention_mask=release_description_inputs.ne(1))[1]
        concatenated_release_issue = torch.cat([release_embedding, description_embedding], dim=-1)

        x_issue_commit_shared = self.shared_dense1(concatenated_issue_commit)
        x_issue_commit_shared = self.dropout(x_issue_commit_shared)

        x_release_issue_shared = self.shared_dense1(concatenated_release_issue)  # Same shared layer
        x_release_issue_shared = self.dropout(x_release_issue_shared)

        x_issue_commit_shared = self.shared_dense2(x_issue_commit_shared)
        x_issue_commit_shared = self.dropout1(x_issue_commit_shared)

        x_release_issue_shared = self.shared_dense2(x_release_issue_shared)  # Same shared layer
        x_release_issue_shared = self.dropout1(x_release_issue_shared)


        logits_issue_commit = self.issue_commit_output_layer(x_issue_commit_shared)
        prob_issue_commit = torch.softmax(logits_issue_commit, dim=-1)


        logits_release_issue = self.release_issue_output_layer(x_release_issue_shared)
        prob_release_issue = torch.softmax(logits_release_issue, dim=-1)

        # Compute losses if targets are provided

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        issue_commit_loss = loss_fct(logits_issue_commit, issue_commit_target)
        release_issue_loss = loss_fct(logits_release_issue, release_issue_target)
        total_loss = issue_commit_loss + release_issue_loss

        # Return logits and loss (if in training or evaluation mode)
        if mode == 'eval':
            return total_loss, prob_issue_commit, prob_release_issue
        elif mode == 'train':
            return total_loss
        else:
            raise ValueError(f"Invalid mode: {mode}")