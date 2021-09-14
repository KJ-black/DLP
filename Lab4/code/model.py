import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size, lat_dim):
        """Define layers for a vanilla rnn encoder"""
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, output_size)
        
        self.hidden2mean = nn.Linear(output_size, lat_dim)
        self.hidden2logv = nn.Linear(output_size, lat_dim)
        
        self.cell2mean = nn.Linear(output_size, lat_dim)
        self.cell2logv = nn.Linear(output_size, lat_dim)

    def forward(self, input_seqs, input_lengths, label, hidden=None, cell=None):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths)
        label = label.view(1, label.shape[0], label.shape[1])
        hidden = torch.cat([hidden, label], dim=2)
        cell = torch.cat([cell, label], dim=2)
        packed_outputs, (hidden, cell) = self.lstm(packed, (hidden, cell))
        outputs, output_lengths = pad_packed_sequence(packed_outputs)
        
        hidden_means = self.hidden2mean(hidden)
        hidden_logv = self.hidden2logv(hidden)
        
        cell_means = self.cell2mean(cell)
        cell_logv = self.cell2logv(cell)
        
        return outputs, hidden_means, hidden_logv, cell_means, cell_logv
    
    def initHidden(self, batch_size):
        return torch.zeros([1, batch_size, 256-4]) # (D * num_layers, batch_size, H_out) 
    
class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, lat_dim, max_length, teacher_forcing_ratio, sos_id, use_cuda):
        """Define layers for a vanilla rnn decoder"""
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.latent2hidden = nn.Linear(lat_dim, hidden_size-4)
        self.latent2cell = nn.Linear(lat_dim, hidden_size-4)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)  # work with NLLLoss = CrossEntropyLoss

        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_id = sos_id
        self.use_cuda = use_cuda

    def forward_step(self, inputs, hidden, cell):
        # inputs: (time_steps=1, batch_size)
        batch_size = inputs.size(1)
        embedded = self.embedding(inputs)
        embedded.view(1, batch_size, self.hidden_size)  # S = T(1) x B x N
        rnn_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # S = T(1) x B x H
        rnn_output = rnn_output.squeeze(0)  # squeeze the time dimension
        output = self.log_softmax(self.out(rnn_output))  # S = B x O
        return output, hidden, cell

    def forward(self, context_vector, decoder_cell, targets):

        # Prepare variable for decoder on time_step_0
        target_vars, target_lengths = targets
        batch_size = context_vector.size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))

        # Pass the context vector
        decoder_hidden = context_vector

        max_target_length = max(target_lengths)
        decoder_outputs = Variable(torch.zeros(
            max_target_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size)

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Unfold the decoder RNN on the time dimension
        for t in range(max_target_length):
            decoder_outputs_on_t, decoder_hidden, decoder_cell = self.forward_step(decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs[t] = decoder_outputs_on_t
            if use_teacher_forcing:
                decoder_input = target_vars[t].unsqueeze(0)
            else:
                decoder_input = self._decode_to_index(decoder_outputs_on_t)

        return decoder_outputs, decoder_hidden, decoder_cell

    def evaluate(self, context_vector, decoder_cell):
        batch_size = context_vector.size(1) # get the batch size
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            self.max_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size)

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        # Unfold the decoder RNN on the time dimension
        for t in range(self.max_length):
            decoder_outputs_on_t, decoder_hidden, decoder_cell = self.forward_step(decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs[t] = decoder_outputs_on_t
            decoder_input = self._decode_to_index(decoder_outputs_on_t)  # select the former output as input

        return self._decode_to_indices(decoder_outputs)

    def _decode_to_index(self, decoder_output):
        """
        evaluate on the logits, get the index of top1
        :param decoder_output: S = B x V or T x V
        """
        value, index = torch.topk(decoder_output, 1)
        index = index.transpose(0, 1)  # S = 1 x B, 1 is the index of top1 class
        if self.use_cuda:
            index = index.cuda()
        return index

    def _decode_to_indices(self, decoder_outputs):
        """
        Evaluate on the decoder outputs(logits), find the top 1 indices.
        Please confirm that the model is on evaluation mode if dropout/batch_norm layers have been added
        :param decoder_outputs: the output sequence from decoder, shape = T x B x V 
        """
        decoded_indices = []
        batch_size = decoder_outputs.size(1)
        decoder_outputs = decoder_outputs.transpose(0, 1)  # S = B x T x V

        for b in range(batch_size):
            top_ids = self._decode_to_index(decoder_outputs[b])
            decoded_indices.append(top_ids.data[0].cpu().numpy())
        return decoded_indices
    
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, targets, label):
        self.encoder.train()
        self.decoder.train()
        
        # variables
        input_vars, input_lengths = inputs
        batch_size = input_vars.shape[1]
        encoder_hidden = self.encoder.initHidden(batch_size).to(device)
        encoder_cell = self.encoder.initHidden(batch_size).to(device)
        
        # encoder
        encoder_outputs, hidden_means, hidden_logv, cell_means, cell_logv = \
            self.encoder.forward(input_vars, input_lengths, label, hidden=encoder_hidden, cell=encoder_cell)  

        # reparaterization trick
        encoder_hidden = self.reparaterization_trick(hidden_means, hidden_logv)
        encoder_hidden = self.decoder.latent2hidden(encoder_hidden)
        encoder_cell = self.reparaterization_trick(cell_means, cell_logv)
        encoder_cell = self.decoder.latent2cell(encoder_cell)
        encoder_hidden = torch.cat([encoder_hidden, label.view(1, label.shape[0], label.shape[1])], dim=2)
        encoder_cell = torch.cat([encoder_cell, label.view(1, label.shape[0], label.shape[1])], dim=2)
        
        # decoder
        decoder_outputs, decoder_hidden, decoder_cell = self.decoder.forward(context_vector=encoder_hidden, 
                                                               decoder_cell=encoder_cell, targets=targets)
        
        return decoder_outputs, decoder_hidden, hidden_means, hidden_logv, cell_means, cell_logv

    def evaluate(self, inputs, src_label, trg_label):
        self.encoder.eval()
        self.decoder.eval()
        
        # variables
        input_vars, input_lengths = inputs
        batch_size = input_vars.shape[1]
        encoder_hidden = self.encoder.initHidden(batch_size).to(device)
        encoder_cell = self.encoder.initHidden(batch_size).to(device)
        
        # encoder
        encoder_outputs, hidden_means, hidden_logv, cell_means, cell_logv = \
            self.encoder.forward(input_vars, input_lengths, src_label, hidden=encoder_hidden, cell=encoder_cell)  

        # reparaterization trick
        encoder_hidden = self.reparaterization_trick(hidden_means, hidden_logv)
        encoder_hidden = self.decoder.latent2hidden(encoder_hidden)
        encoder_cell = self.reparaterization_trick(cell_means, cell_logv)
        encoder_cell = self.decoder.latent2cell(encoder_cell)
        encoder_hidden = torch.cat([encoder_hidden, trg_label.view(1, trg_label.shape[0], trg_label.shape[1])], dim=2)
        encoder_cell = torch.cat([encoder_cell, trg_label.view(1, trg_label.shape[0], trg_label.shape[1])], dim=2)
        
        # decoder
        decoded_sentence = self.decoder.evaluate(context_vector=encoder_hidden, decoder_cell=encoder_cell)
        
        return decoded_sentence
    
    def reparaterization_trick(self, mean, logv):
        std = torch.exp(0.5*logv)
        eps = torch.randn_like(std)
        return  mean + eps * std