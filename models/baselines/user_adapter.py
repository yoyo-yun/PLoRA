import torch, math
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooled_output: Optional[Tuple[torch.FloatTensor]] = None


class UserAdapter(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.user_embedding = torch.nn.Embedding(self.config.usr_size, self.config.usr_dim,
                                                 _weight=torch.zeros(self.config.usr_size, self.config.usr_dim),
                                                 _freeze=False)
        self.base_model = model
        self.get_word_embedding()  # self.word_embeddings

    def forward(self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        p=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,):
        batch_size = input_ids.shape[0]
        if attention_mask is not None and p is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, 1).to(self.config.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if p is not None:
            inputs_embeds_A, inputs_embeds_B = inputs_embeds[:, :1], inputs_embeds[:, 1:]
            user_prompts = self.user_embedding(p).unsqueeze(1)  # (bs, 1, dim)
            prompts = user_prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((inputs_embeds_A, prompts, inputs_embeds_B), dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


    def get_word_embedding(self):
        for named_param, value in list(self.base_model.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = self.base_model.get_submodule(named_param.replace(".weight", ""))
                break