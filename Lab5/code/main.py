from dataset import *
from evaluator import *
from torch.utils.data import DataLoader
from models import *
from utils import *

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # not use X window to show img

import tqdm
import argparse
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def main():
    parser = argparse.ArgumentParser(description='Lets play GANs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--load_model', action="store_true")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--path", type=str, default="./")
    
    args = parser.parse_args()
    
    if not args.train and not args.eval:
        sys.exit("Not choosen the mode. (--train or --eval)")
        
    if not args.path:
        sys.exit("Not assign the path. (--path)")
    else:
        print("Loading data...")
        try:
            test_labels = load_test(args.path, args.test_file)
        except:
            print("No test file. If you want then --test_file=FILENAME")
        train_loader = load_train(args.path, args.batch)

    # loss function
#     adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.BCELoss()

    # model 
    n_classes = 24
    latent_dim = 100
    img_shape = 64
    n_channels = 3
    load = False
    
    if args.load_model:
        generator = torch.load("./generator.pt", map_location=device)
        discriminator = torch.load("./discriminator.pt", map_location=device)
    else:
        generator = Generator(n_classes, latent_dim, img_shape, n_channels).to(device)
        discriminator = Discriminator(n_classes, img_shape).to(device)
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    # optimizer 
    lr = args.learning_rate
    b1 = 0.5
    b2 = 0.999
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    
    # training 
    epochs = args.epochs
    n_critic = 5 # number of training steps for discriminator per iter
    lambda_gp = 10 # Loss weight for gradient penalty
    lambda_cls = 5 # los weight for cls 
    save = True
    max_g_loss = np.inf
    max_d_loss = -np.inf

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    
    # ============================= training ============================= #
    if args.train:
        print("Training...")
        for epoch in tqdm.tqdm(range(epochs)):
            total_d_loss = 0
            total_g_loss = 0
            generator.train()

            for i, (real_imgs, labels) in enumerate(train_loader):
                batch_size = real_imgs.shape[0]

                real_imgs = real_imgs[:, :3].to(device)
                real_labels = labels.to(device)

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device) # shape [batch_size, latent_dim] with normal distribution       

                # Generate a batch of images
                gen_imgs = generator(z, real_labels) # shape [batch_size, 3, 64, 64]

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                 # Real images
                real_pred, real_aux = discriminator(real_imgs)
                d_real_loss = - torch.mean(real_pred)
                d_real_cls_loss = auxiliary_loss(real_aux, real_labels.float())

                # Fake images
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                d_fake_loss = torch.mean(fake_pred)
                d_fake_cls_loss = auxiliary_loss(fake_aux, real_labels.float())

                # gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, gen_imgs)

                 # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss + lambda_cls*d_real_cls_loss + lambda_gp*gradient_penalty
        #         print( "   [Epoch %d/%d] [Iter %d/%d] [d_real_loss %f] [d_fake_loss %f] [d_real_cls_loss %f] [d_fake_cls_loss %f]"
        #                 % (epoch+1, epochs, i, len(train_loader), d_real_loss, d_fake_loss, d_real_cls_loss, d_fake_cls_loss)
        #             )

                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                # Train the generator every n_critic steps
                if i % n_critic == 0:
                    optimizer_G.zero_grad()

                    # Loss measures generator's ability to fool the discriminator
                    gen_imgs = generator(z, real_labels) # shape [batch_size, 3, 64, 64]

                    # Loss measures generator's ability to fool the discriminator
                    fake_validity, pred_label = discriminator(gen_imgs)

                    g_loss_fake = -torch.mean(fake_validity)
                    g_loss_cls = auxiliary_loss(pred_label, real_labels.float())
                    g_loss = g_loss_fake + lambda_cls * g_loss_cls

                    g_loss.backward()
                    optimizer_G.step()

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()

            score = test_eval(test_labels, latent_dim, generator, epoch+1)
            total_d_loss /= len(train_loader)
            total_g_loss /= (len(train_loader)/n_critic)

            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f] [test score: %f]"
                % (epoch+1, epochs, total_d_loss, total_g_loss, score)
            )

            # loss save
            if save:
                save_acc("./loss.npz", total_g_loss, total_d_loss, score)

            if save:
                if max_g_loss > total_g_loss:
                    max_g_loss = total_g_loss
                    torch.save(generator, "./save_generator.pt")
                if max_d_loss < total_d_loss:
                    max_d_loss = total_d_loss
                    torch.save(discriminator, "./save_discriminator.pt")
                    
    if args.eval:
        print(test_eval(test_labels, latent_dim, generator, 0))
        
if __name__ == '__main__':
    main()