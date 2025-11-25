import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from pdarts_utils import *
from collections import OrderedDict
from torchviz import make_dot

os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--workers', type=int, default=0, help='number of workers to load dataset') #2
parser.add_argument('--batch_size', type=int, default=64, help='batch size') #96
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs') # P-DARTS: 25, Proposed: 25, DARTS: 100, PC-DARTS: 50
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels') # P-DARTS: 16, Proposed: 36, DARTS: 16, PC-DARTS: 16
parser.add_argument('--layers', type=int, default=14, help='total number of layers') # P-DARTS: 5, Proposed: 20, DARTS: 8, PC-DARTS: 8
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

args = parser.parse_args()
parameterSharing = False

args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000

import io
from zipfile import ZipFile
from PIL import Image
from torch.utils.data import Dataset

from torch.utils.data import get_worker_info

from zipfile import ZipFile

from torch.utils.data import Subset
import random

def get_subset(dataset, num_samples, seed=42):
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)
    return Subset(dataset, indices)


class ImageNetDataset(Dataset):
    def __init__(self, dataroot: str, train: bool = True, transform=None):
        self.zfpath = os.path.join(
            dataroot,
            f"{'train' if train else 'val'}_blurred.zip",
        )
        self.transform = transform
        self.zf = None

        # Avoid using ZipFile in `with` block
        zf = ZipFile(self.zfpath)
        self.imglist = [path for path in zf.namelist() if path.endswith(".jpg")]
        zf.close()

        with open(os.path.join(dataroot, "map_clsloc.txt")) as f:
            def parse_row(row): return row.split()[0], int(row.split()[1]) - 1
            self.classes = dict(parse_row(row) for row in f)

    def get_label(self, path: str) -> int:
        classname: str = path.split("/")[-2]
        return self.classes[classname]
        
    def __len__(self):
        return len(self.imglist)


    def _ensure_zip_open(self):
        if self.zf is None:
            self.zf = ZipFile(self.zfpath)

    def __getitem__(self, idx):
        self._ensure_zip_open()
        imgpath = self.imglist[idx]
        img = Image.open(io.BytesIO(self.zf.read(imgpath))).convert('RGB')
        label = self.get_label(imgpath)
        if self.transform:
            img = self.transform(img)
        return img, label



def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    
    imagenet_root = '/mimer/NOBACKUP/Datasets/ImageNet/Face-blurred_ILSVRC2012-2017'

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #train_data = ImageNetDataset(imagenet_root, train=True, transform=train_transform)
    train_data = ImageNetDataset(imagenet_root, train=True, transform=train_transform)
    train_data = get_subset(train_data, num_samples=10000)
    #valid_data = ImageNetDataset(imagenet_root, train=False, transform=valid_transform)
    valid_data = ImageNetDataset(imagenet_root, train=False, transform=valid_transform)
    valid_data = get_subset(valid_data, num_samples=2000)


    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    
    switches = []
    '''for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])'''
    for i in range(14):
        row = [True for j in range(len(PRIMITIVES))]
        # row[0] = False  # Set the first value to zero for none operation
        row[3] = False  # Set the second value to zero for skip connections
        switches.append(row)
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # Numbers to Keep
    num_to_keep = [6, 5, 4, 3, 2, 1] # Proposed
    # num_to_keep = [5, 3, 1] # P-DARTS
    #num_to_keep = [1] # DARTS
    
    # Numbers to Drops
    num_to_drop = [1, 1, 1, 1, 1, 1] # Proposed
    # num_to_drop = [3, 2, 2] # P-DARTS
    #num_to_drop = [7] # DARTS
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0, 0, 0, 0] # Proposed
        #add_width = [0, 0, 0] # P-DARTS
        #add_width = [0] # DARTS
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 0, 0, 0, 0, 0] # Proposed
        #add_layers = [0, 6, 12] # P-DARTS
        #add_layers = [0] # DARTS
    if len(args.dropout_rate) == 3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Proposed
        #drop_rate = [0.0, 0.0, 0.0] # P-DARTS
        #drop_rate = [0.0] # DARTS
    eps_no_archs = [10, 10, 10, 10, 10, 10]  # Proposed
    #eps_no_archs = [10, 10, 10] # P-DARTS    
    #eps_no_archs = [0] # DARTS
    for sp in range(len(num_to_keep)):
        model = Network(args.init_channels + int(add_width[sp]), CLASSES, args.layers + int(add_layers[sp]),
                        criterion, switches_normal=switches_normal, switches_reduce=switches_reduce,
                        p=float(drop_rate[sp]))
        model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("Model Parameters = %f Millions", utils.count_parameters_in_MB(model))
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)
        optimizer = torch.optim.SGD(
            network_params,
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                                       lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                       weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        for epoch in range(epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
                model.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.module.update_p()
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer,
                                             optimizer_a, lr, train_arch=False)
            else:
                model.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model.module.update_p()
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer,
                                             optimizer_a, lr, train_arch=True)

                # === Collect and save attention logs ===
                #all_logs = []
                #for module in model.modules():
                #    if isinstance(module, MixedOp):
                #        all_logs.extend(module.get_logged_attention_data())
                #        module.clear_logged_data()

                #write_logs_to_csv(all_logs, filename=args.save + '/sp' + str(sp) +'_attention_weights_epoch'+str(epoch)+'.csv')

            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
        ma = torch.cuda.memory_allocated()/(1024*1024*1024)
        mma = torch.cuda.max_memory_allocated()/(1024*1024*1024)
        logging.info("Memory Allocated = %f Giga, Max = %f Giga", ma, mma)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        logging.info('------Dropping %d paths------',num_to_drop[sp])
        # Save switches info for s-c refinement.
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()

        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        #normal_prob = torch.sigmoid(arch_param[0]).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        #reduce_prob = torch.sigmoid(arch_param[1]).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        #logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        #logging_switches(switches_reduce)

        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            #normal_prob = torch.sigmoid(arch_param[0]).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            #reduce_prob = torch.sigmoid(arch_param[1]).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])
                # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk = check_sk_number(switches_normal)
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)


def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above.
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if (step % args.report_freq == 0) and (step != 0):
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)