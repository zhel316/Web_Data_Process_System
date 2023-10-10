from RE_model.data_loader import DataLoader
import RE_model.utils as utils
import argparse
import os
import torch
import RE_model.model.net as net
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def load_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='RE_model/data/SemEval2010_task8', help="Directory containing the dataset")
    parser.add_argument('--embedding_file', default='RE_model/data/embeddings/vector_50d.txt', help="Path to embeddings file.")
    parser.add_argument('--model_dir', default='RE_model/experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--gpu', default=-1, help="GPU device number, 0 by default, -1 means CPU.")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before training")

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    if torch.cuda.is_available():
        params.gpu = args.gpu
    else:
        params.gpu = -1
    return args,params

def load_dataloader(args,params):
    # Initialize the DataLoader
    data_loader = DataLoader(data_dir=args.data_dir,
                             embedding_file=args.embedding_file,
                             word_emb_dim=params.word_emb_dim,
                             max_len=params.max_len,
                             pos_dis_limit=params.pos_dis_limit,
                             pad_word='<pad>',
                             unk_word='<unk>',
                             other_label='Other',
                             gpu=params.gpu)
    # Load word embdding
    data_loader.load_embeddings_from_file_and_unique_words(emb_path=args.embedding_file,
                                                           emb_delimiter=' ',
                                                           verbose=True)
    metric_labels = data_loader.metric_labels  # relation labels to be evaluated
    # print(data_loader.word2idx)
    return data_loader,metric_labels



def load_model(data_loader,params):
    model = net.CNN(data_loader, params)
    model.eval()
    return model

def predict_re(args,data_loader,model,data):
    pre_data = data_loader.load_pre(data)

    label2idx = dict()

    labels_path = os.path.join(args.data_dir, 'labels.txt')
    with open(labels_path, 'r') as f:
        for i, line in enumerate(f):
            label2idx[i] = line.strip()

    batch_output = model.forward(pre_data)
    batch_output_labels = torch.max(batch_output, dim=1)[1]

    relation = label2idx[int(batch_output_labels[0])]
    return relation


if __name__ == '__main__':
    args,params = load_setting()
    data_loader,metric_labels = load_dataloader(args, params)
    model = load_model(data_loader,params)
    data = "WordPress"+'\t'+"Elegant Themes"+'\t'+"Powered by WordPress  Designed by Elegant Themes"
    key = 'clueweb12-0000tw-00-00004'

    print(predict_re(args,data_loader,model,data))