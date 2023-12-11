import torch
import torch.nn as nn
from models.baselines.baseline_att import SelfAttention, UAttention


class WordLevelRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        words_dim = config.words_dim
        word_hidden_dim = config.word_hidden_dim
        usr_dim = config.usr_dim

        self.gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)
        self.att = UAttention(word_hidden_dim, usr_dim, config)

    def forward(self, input_embeded, usr, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.att(hidden_states, usr, mask=mask)
        return output


class SentLevelRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        sentence_hidden_dim = config.sent_hidden_dim
        word_hidden_dim = config.word_hidden_dim
        usr_dim = config.usr_dim

        self.gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
        self.att = UAttention(sentence_hidden_dim, usr_dim, config)

    def forward(self, input_embeded, usr, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.att(hidden_states, usr, mask=mask)
        return output


class WordLevelRNN_LA(nn.Module):

    def __init__(self, config):
        super().__init__()
        words_dim = config.words_dim
        word_hidden_dim = config.word_hidden_dim

        self.gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)
        self.att = SelfAttention(word_hidden_dim, config)

    def forward(self, input_embeded, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.att(hidden_states, mask=mask)
        return output


class SentLevelRNN_LA(nn.Module):
    def __init__(self, config):
        super().__init__()
        sentence_hidden_dim = config.sent_hidden_dim
        word_hidden_dim = config.word_hidden_dim

        self.gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
        self.att = SelfAttention(sentence_hidden_dim, config)


    def forward(self, input_embeded, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.att(hidden_states, mask=mask)
        return output