from model import *
from utils import *
from Data import *
from train import *

import tqdm
import argparse
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))

def main():
    parser = argparse.ArgumentParser(description='Seq2Seq LSTM VAE')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--tense_test', action="store_true")
    parser.add_argument('--noise_generate', action="store_true")

    parser.add_argument('--load_model', action="store_true")
    parser.add_argument("--model_name", type=str, default="model.pt")
    
    args = parser.parse_args()
    
    ## ====================== load training data ====================== ##
    train_data = np.squeeze(pd.read_csv('train.txt', header=None))
    y_train = y_train_make(len(train_data))
    y_train = to_one_hot(y_train)
    train_data = split_data(train_data)
    train_loader = DataTransformer(train_data, y_train, use_cuda=True)
    
    ## ====================== load testing data ======================= ##
    test_data = np.squeeze(pd.read_csv('test.txt', header=None))
    test_data = split_data(test_data)
    test_data = np.array(test_data)
    src, trg = src_trg_split(test_data)
    test_src = []
    test_trg = []

    for word in src:
        test_src.append(train_loader.vocab.sequence_to_indices(word, add_eos=True))

    for word in trg:
        test_trg.append(train_loader.vocab.sequence_to_indices(word, add_eos=True))
    """
    sp -> p
    sp -> pg
    sp -> tp
    sp -> tp
    p  -> tp
    sp -> pg
    p  -> sp
    pg -> sp
    pg -> p
    pg -> tp
    """
    sp = 0
    tp = 1
    pg = 2
    p = 3
    test_c_src = np.array([sp, sp, sp, sp, p, sp, p, pg, pg, pg]).reshape(-1, 1)
    test_c_trg = np.array([p, pg, tp, tp, tp, pg, sp, sp, p, tp]).reshape(-1, 1)
    test_c_src = Variable(torch.LongTensor(to_one_hot(test_c_src))).to(device)
    test_c_trg = Variable(torch.LongTensor(to_one_hot(test_c_trg))).to(device)
    

    ## ====================== make model ====================== ##
    encoder = Encoder(vocab_size=train_loader.vocab_size,
                             embedding_size=256,
                             output_size=256,
                             lat_dim=32).to(device)

    decoder = Decoder(hidden_size=256,
                             output_size=train_loader.vocab_size,
                             lat_dim=32,
                             max_length=train_loader.max_length,
                             teacher_forcing_ratio=1.,
                             sos_id=train_loader.SOS_ID,
                             use_cuda=True).to(device)
    seq2seq = Seq2Seq(encoder=encoder,
                      decoder=decoder)
    if args.load_model:
        seq2seq.load_state_dict(torch.load(args.model_name, map_location=device))
    trainer = Trainer(seq2seq, train_loader, y_train, learning_rate=args.learning_rate, use_cuda=True, checkpoint_name="model.pt")
    
    if args.train:
        ## =========================training========================== ##
        for epoch in tqdm.tqdm(range(args.epochs)):
            trainer.train(num_epochs=10, batch_size=args.batch, pretrained=args.load_model)

            ## ===========================eval============================ ##
            print("========================================Evaluating========================================")  
            total_score = 0.0
            for i in range(len(test_src)):
                word = train_loader.vocab.indices_to_sequence(test_src[i])
                trg_true = train_loader.vocab.indices_to_sequence(test_trg[i])
                results = trainer.evaluate(word, test_c_src[i].view(1, -1), test_c_trg[i].view(1, -1))[0]
                score = trainer.compute_bleu(results, trg_true)
                print("Src_true: {:>12}".format(word), "\tTrg_true:{:>12}".format(trg_true), "\tPredict: {:>12}".format(results), "\tScore: {:>8.5f}".format(score))
                total_score += score
            total_score /= len(test_src)
            trainer.score.append(total_score)
            trainer.save_model()
            print("Total score:", total_score)
            print("==========================================================================================\n")
    
    if args.eval:
        ## BLEU-4
        if args.tense_test:
            total_score = 0.0
            for i in range(len(test_src)):
                word = train_loader.vocab.indices_to_sequence(test_src[i])
                trg_true = train_loader.vocab.indices_to_sequence(test_trg[i])
                results = trainer.evaluate(word, test_c_src[i].view(1, -1), test_c_trg[i].view(1, -1))[0]
                score = trainer.compute_bleu(results, trg_true)
                print("Src_true: {:>12}".format(word), "\tTrg_true:{:>12}".format(trg_true), "\tPredict: {:>12}".format(results), "\tScore: {:>8.5f}".format(score))
                total_score += score
            total_score /= len(test_src)
            print(total_score)
         
        ## noise generate
        if args.noise_generate:
            label = torch.LongTensor([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]).to(device)

            words = []
            for i in range(100):
                hidden_mean = torch.randn([1, 1, 32]).to(device)
                hidden_logv = torch.randn([1, 1, 32]).to(device)
                cell_mean = torch.randn([1, 1, 32]).to(device)
                cell_logv = torch.randn([1, 1, 32]).to(device)

                encoder_hidden = reparaterization_trick(hidden_mean, hidden_logv)
                encoder_hidden = decoder.latent2hidden(encoder_hidden)
                encoder_cell = reparaterization_trick(cell_mean, cell_logv)
                encoder_cell = decoder.latent2hidden(encoder_cell)

                tmp = []
                for i in range(4):
                    hidden = torch.cat([encoder_hidden, label[i].view(1, 1, 4)], dim=2)
                    cell = torch.cat([encoder_cell, label[i].view(1, 1, 4)], dim=2)
                    decoded_indices = decoder.evaluate(context_vector=hidden, decoder_cell=cell)
                    results = []
                    for indices in decoded_indices:
                        results.append(train_loader.vocab.indices_to_sequence(indices))
                    tmp.append(results[0])
                words.append(tmp)
            print(words)
            print("Total Gaussion score:", Gaussian_score(words))

if __name__ == '__main__':
    main()