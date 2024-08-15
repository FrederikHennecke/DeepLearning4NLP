from bert import BertModel
import torch
import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(self, config, num_models=3):
        super(CombinedModel, self).__init__()
        self.model1 = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        self.model2 = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        self.model3 = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )

        self.gating_netweork = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        self.gatiing_linear = nn.Linear(config.hidden_size, num_models)

    def softmax_to_onehot(self, logits):
        max_indices = torch.argmax(logits, dim=-1)
        onehot = torch.zeros_like(logits)
        onehot.scatter_(-1, max_indices.unsqueeze(-1), 1)
        return onehot

    def forward(self, input_ids, attention_mask):
        _, gating_pooler_output, _, _ = self.gating_netweork(input_ids, attention_mask)
        gating_logits = self.gatiing_linear(gating_pooler_output)
        gating_weights = torch.softmax(gating_logits, dim=-1)
        gating_probs = self.softmax_to_onehot(gating_weights)

        _, pooler_output1, all_sequences1, _ = self.model1(input_ids, attention_mask)
        _, pooler_output2, all_sequences2, _ = self.model2(input_ids, attention_mask)
        _, pooler_output3, all_sequences3, _ = self.model3(input_ids, attention_mask)

        pooler_output = (
            pooler_output1 * gating_probs[:, :1]
            + pooler_output2 * gating_probs[:, 1:2]
            + pooler_output3 * gating_probs[:, 2:3]
        )
        last_hidden_state = (
            all_sequences1["last_hidden_state"] * gating_probs[:, :1, None]
            + all_sequences2["last_hidden_state"] * gating_probs[:, 1:2, None]
            + all_sequences3["last_hidden_state"] * gating_probs[:, 2:3, None]
        )
        return {"pooler_output": pooler_output, "last_hidden_state": last_hidden_state}
