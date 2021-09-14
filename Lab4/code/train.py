from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import torch
import random
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):

    def __init__(self, model, data_transformer, label, learning_rate, use_cuda, checkpoint_name="model.pt", loss_file="loss.npz",
                 teacher_forcing_ratio=1.0, kl_weight=0):

        self.model = model
        self.checkpoint_name = checkpoint_name
        self.loss_file = loss_file
        self.total_iter = 0
        
        # save list
        self.entropy = []
        self.kld = []
        self.kl_weight_list = []
        self.teacher_forcing_ratio_list = []
        self.score = []
        
        # init hyperparameters
        self.kl_weight = kl_weight
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.kl_weight_list.append(self.kl_weight)
        self.teacher_forcing_ratio_list.append(self.teacher_forcing_ratio)
        
        # record some information about dataset
        self.data_transformer = data_transformer
        self.label = label
        self.vocab_size = self.data_transformer.vocab_size
        self.PAD_ID = self.data_transformer.PAD_ID
        self.use_cuda = use_cuda
        
        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.optimizer= torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.PAD_ID, reduction='mean')

    def train(self, num_epochs, batch_size, pretrained=False):

        if pretrained:
            self.load_model()

        step = 0

        for epoch in range(0, num_epochs):
            mini_batches = self.data_transformer.mini_batches(batch_size=batch_size)
            for input_batch, target_batch, label_batch in mini_batches:
                self.total_iter += 1
                self.optimizer.zero_grad()
                self.model.decoder.teacher_forcing_ratio = self.teacher_forcing_ratio
                decoder_outputs, decoder_hidden, hidden_means, hidden_logv, cell_means, cell_logv = \
                    self.model(input_batch, target_batch, label_batch)

                # calculate the loss and back prop.
                cur_loss = self.get_loss(decoder_outputs, target_batch[0])
                kl_loss = self.kl_weight * self.get_kl_loss(hidden_means, hidden_logv)+\
                            self.kl_weight* self.get_kl_loss(cell_means, cell_logv)
                loss = cur_loss + kl_loss
                
                self.entropy.append(cur_loss.item())
                self.kld.append(kl_loss.item())
                
                # logging
                step += 1
                if step % 50 == 0:
                    print("Step:", step, "char-loss: ", loss.item())
                    print("KL_weight: ", self.kl_weight, "teacher_forcing_ratio: ", self.teacher_forcing_ratio)
                    self.save_model()
                loss.backward()

                # optimize
                self.optimizer.step()
                
                # update hyperparameters
                self.kl_weight = self.get_kl_weight(self.kl_weight)
                self.teacher_forcing_ratio = self.get_teacher_forcing_ratio(self.teacher_forcing_ratio)
                self.kl_weight_list.append(self.kl_weight)
                self.teacher_forcing_ratio_list.append(self.teacher_forcing_ratio)

        self.save_model()

    def get_loss(self, decoder_outputs, targets):
        b = decoder_outputs.size(1)
        t = decoder_outputs.size(0)
        targets = targets.contiguous().view(-1)  # S = (B*T)
        decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (B*T) x V
        return self.criterion(decoder_outputs, targets)
    
    def get_kl_loss(self, mean, logvar):
        result = -0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)
        return result.mean()
    
    def get_kl_weight(self, kl_weight):
#         return 0
        return min(0.4, kl_weight + 0.000001)

    def get_teacher_forcing_ratio(self, teacher_forcing_ratio):
        return teacher_forcing_ratio
#         return 1
#         return max(0, teacher_forcing_ratio - 0.0000000000001)

    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)
        np.savez(self.loss_file, entropy=self.entropy, kld=self.kld, kl_weight=self.kl_weight_list,\
                 teacher_forcing_ratio=self.teacher_forcing_ratio_list, score=self.score)
        print("Model has been saved as %s.\n" % self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name, map_location=device))
        load_file = np.load(self.loss_file)
        self.entropy = load_file['entropy'].tolist()
        self.kld = load_file['kld'].tolist()
        self.kl_weight_list = load_file['kl_weight'].tolist()
        self.teacher_forcing_ratio_list = load_file['teacher_forcing_ratio'].tolist()
        self.score = load_file['score'].tolist()
        print("Pretrained model has been loaded.\n")

    def evaluate(self, words, src_label, trg_label):
        # make sure that words is list
        if type(words) is not list:
            words = [words]

        # transform word to index-sequence
        eval_var = self.data_transformer.evaluation_batch(words=words)
        decoded_indices = self.model.evaluate(eval_var, src_label, trg_label)
        results = []
        for indices in decoded_indices:
            results.append(self.data_transformer.vocab.indices_to_sequence(indices))
        return results
    
    def compute_bleu(self, output, reference):
        cc = SmoothingFunction()
        if len(reference) == 3:
            weights = (0.33,0.33,0.33)
        else:
            weights = (0.25,0.25,0.25,0.25)
        return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)