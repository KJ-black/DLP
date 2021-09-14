from dataset import *
from evaluator import *
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # not use X window to show img
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_test(path, test_filename):
    data = json.load(open(os.path.join(path,test_filename)))
    obj = json.load(open(os.path.join(path,'objects.json')))
    test_labels = data
    for i in range(len(test_labels)):
        for j in range(len(test_labels[i])):
            test_labels[i][j] = obj[test_labels[i][j]]
        tmp = np.zeros(len(obj))
        tmp[test_labels[i]] = 1
        test_labels[i] = tmp
    test_labels = torch.tensor(test_labels).to(device)
    return test_labels

def load_train(path, batch_size):
    image_size = 64

    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([image_size, image_size]),
                                transforms.CenterCrop([image_size, image_size]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)), # normalize to [-1, 1] for the last layer of generator is tanh()
                                ])

    # preprocessing size -> 64x64
    train = ICLEVRLoader(path, trans=trans, mode="train", preprocessing=None)

    train_loader = DataLoader(
        dataset=train, 
        batch_size=batch_size,
        num_workers = 4
    )
    
    return train_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def save_acc(filename, g_loss, d_loss, score):
    try:
        acc = np.load(filename)
        np_g_loss = acc['g_loss']
        np_d_loss = acc['d_loss']
        np_score = acc['score']
        np_g_loss = np.append(np_g_loss, g_loss)
        np_d_loss = np.append(np_d_loss, d_loss)
        np_score = np.append(np_score, score)
        np.savez(filename, g_loss=np_g_loss, d_loss=np_d_loss, score=np_score)
    except:
        g_loss = np.array(g_loss)
        d_loss = np.array(d_loss)
        score = np.array(score)
        np.savez(filename, g_loss=g_loss, d_loss=d_loss, score=score)
        
def test_eval(test_labels, latent_dim, generator, epoch):
    FloatTensor = torch.FloatTensor
    np.random.seed(8)    
    generator.eval()
    eval_model = evaluation_model()
    batch_size = test_labels.shape[0]
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device) # shape [batch_size, latent_dim]
    gen_imgs = generator(z, test_labels) # shape [batch_size, 3, 64, 64]
    if epoch % 10 == 0:
        show_image(gen_imgs)
    return eval_model.eval(gen_imgs, test_labels)

def show_image(gen_imgs):
    # step 1: convert it to [0 ,2]
    gen_imgs = gen_imgs +1
    
    # step 2: convert it to [0 ,1]
    gen_imgs = gen_imgs - gen_imgs.min()
    gen_imgs = gen_imgs / (gen_imgs.max() - gen_imgs.min())
    
    grid = make_grid(gen_imgs)
    plt.figure(figsize=(14, 14))
    plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
#     plt.show()
    plt.savefig("test_image.png")    
    
def compute_gradient_penalty(D, real_samples, fake_samples):
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    out, out_cls = D(interpolates)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=out,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty