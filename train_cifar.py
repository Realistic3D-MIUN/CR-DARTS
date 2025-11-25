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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from torch.autograd import Variable
from model import NetworkCIFAR as Network
from calflops import calculate_flops


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='CR_DARTS_CIFAR', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--tmp_data_dir', type=str, default='tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--cifar100', action='store_true', default=False, help='if use cifar100')
# NASNet , AmoebaNet , DARTS_V1 , DARTS_V2 , PDARTS , PC_DARTS , PD_DARTS , CR_DARTS
args, unparsed = parser.parse_known_args()

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    
    genotype = eval("genotypes.%s" % args.arch)    
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    complexity_analysis(model)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        start_time = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)

        scheduler.step()
        #logging.info('Epoch: %d lr %e', epoch, scheduler.get_last_lr())
        #print('Epoch: ', epoch,' lr: ', scheduler.get_last_lr()[0])
        #logging.info('Train_acc: %f', train_acc)
        #print('Train_acc: ', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
            utils.save(model.module, os.path.join(args.save, args.arch + str(CIFAR_CLASSES) + '_weights_best.pt'))
        #logging.info('Valid_acc: %f', valid_acc)
        #print('Valid_acc: ', valid_acc)
        end_time = time.time()
        duration = end_time - start_time
        #print('Epoch time: ', duration )
        print('Epoch: ', epoch, ' lr: ', scheduler.get_last_lr()[0], ' Train_acc: ', train_acc, ' Valid_acc: ', valid_acc, ' Time: ', duration )
        utils.save(model.module, os.path.join(args.save, args.arch + str(CIFAR_CLASSES) + '_weights_last.pt'))

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg

def complexity_analysis(main_model):
    model = main_model
    model = model.eval()
    batch_size = 1
    input_shape = (batch_size, 3, 32, 32)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4, output_unit='G')
    print("FLOPs: %s   MACs: %s   Params: %s \n" % (flops, macs, params))
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print('Model parameters: ', total_params / 1000000, ' Millions')

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('File size: {:.8f} MB'.format(size_all_mb))

    input_tensor = torch.rand(1, 3, 32, 32, device='cuda')
    input_tensor = input_tensor.to('cuda')
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Using torch.utils.benchmark.Timer for measurement

    import torch.utils.benchmark as benchmark
    timer = benchmark.Timer(
        stmt='model(input_tensor)',
        globals={'model': model, 'input_tensor': input_tensor}
    )

    # Run the timer for a specified number of measurements
    # Warm-up
    measurement = timer.timeit(10)  # Run few times to warm up
    # Run the timer with 1000 repetitions
    measurement = timer.timeit(1000)  # Run to measure average inference time
    print(measurement)

    # Print average inference time (for readability in seconds)
    print(f'Average Inference Time= {measurement.mean:.6f}s')
    #sys.exit(0)


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: ', duration, 's')
    
