import torch
import torch.nn as nn
from models.baselines.rnn import WordLevelRNN, WordLevelRNN_LA, SentLevelRNN, SentLevelRNN_LA

class RNNOutput:
    logits = None
    loss = None
    def __init__(self, logits=None, loss=None):
        self.logits = logits
        self.loss = loss


class Classifier(nn.Module):
    def __init__(self,input_dim, config):
        super(Classifier, self).__init__()
        self.pre_classifier = nn.Linear(input_dim, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_labels)

    def forward(self, hidden):
        pre_hidden = torch.tanh(self.pre_classifier(hidden))
        logits = self.classifier(pre_hidden)
        return logits


class NSC(torch.nn.Module):
    def __init__(self, config):
        super(NSC, self).__init__()
        self.config = config
        self.text_embed = nn.Embedding(config.vectors.size(0), config.vectors.size(1),
                                       padding_idx=config.pad_idx,
                                       _weight=self.config.vectors)
        self.text_embed.weight.requires_grad = False
        if config.nop:
            self.word_attention_rnn = WordLevelRNN_LA(config)
            self.sentence_attention_rnn = SentLevelRNN_LA(config)
            self.register_buffer("p_embedding", None)
        else:
            self.word_attention_rnn = WordLevelRNN(config)
            self.sentence_attention_rnn = SentLevelRNN(config)

            self.p_embedding = torch.nn.Embedding(self.config.usr_size, self.config.usr_dim,
                                                  _weight=torch.zeros(self.config.usr_size, self.config.usr_dim),
                                                  )
            self.text_embed.weight.requires_grad = True

        self.classifier = Classifier(config.sent_hidden_dim, config)

    def forward(self, input_ids, attention_mask, labels=None, p=None):
        if self.p_embedding is not None and p is not None:
            p = self.p_embedding(p)

        input_ids = input_ids.permute(1, 0, 2)  # input_ids: (sent, batch, word)
        input_embeds = self.text_embed(input_ids) # input_embeds: (sent, batch, word_dim)
        num_sentences = input_embeds.size(0)
        words_text = []

        mask_word = attention_mask.permute(1, 0, 2)  # text: (sent, batch, word)
        mask_sent = attention_mask.long().sum(2) > 0  # (batch, sent)

        # for i in self.text_embed.weight[:2]:
        #     print(i)
        # exit()

        for i in range(num_sentences):
            if self.config.nop:
                text_word = self.word_attention_rnn(input_embeds[i], mask=mask_word[i])
            else:
                text_word = self.word_attention_rnn(input_embeds[i], p, mask=mask_word[i])
            words_text.append(text_word)
        words_text = torch.stack(words_text, 1) # (batch, sents, dim)

        if self.config.nop:
            sents = self.sentence_attention_rnn(words_text, mask=mask_sent)
        else:
            sents = self.sentence_attention_rnn(words_text, p, mask=mask_sent)
        logits = self.classifier(sents)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return RNNOutput(
            logits=logits,
            loss=loss
        )


