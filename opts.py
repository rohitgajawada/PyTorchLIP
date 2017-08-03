import argparse

def myargparser():
    parser = argparse.ArgumentParser(description='PyTorch Core Training')

    #data stuff
    parser.add_argument('--data', metavar='DIR', help='path to dataset', default='')
    parser.add_argument('--vdata', metavar='VDIR', help='path to val set')
    parser.add_argument('--dataset', metavar='dset', default='cifar100', help='chosen dataset')
    parser.add_argument('--testOnly', metavar='testOnly', default=False, help='run on validation set only')

    #default stuff
    parser.add_argument('--cachemode', metavar='cm', default=True, help='if cachemode')
    parser.add_argument('--cuda', metavar='cuda', default=True, help='if cuda is available')
    parser.add_argument('--manualSeed', metavar='seed', default=123, help='fixed seed for experiments')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--ngpus', metavar='ngpus', default=1, help='no. of gpus')
    parser.add_argument('--logdir', metavar='logdir', type=str, default='../logs/', help='log directory')
    parser.add_argument('--verbose', default=True)

    #other default stuff
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=12, type=int,metavar='N', help='mini-batch size (default: 128)')

    parser.add_argument('--nclasses', metavar='nc', help='number of classes', default=20)
    parser.add_argument('--printfreq', default=20, type=int, metavar='print_freq', help='print frequency (default: 10)')

    #optimizer/criterion stuff
    parser.add_argument('--decayinterval', metavar='decay', default=50, help='decays by a power of 2 in these epochs')
    parser.add_argument('-criterion', default='crossentropy', type=str, metavar='crit', help='criterion')
    parser.add_argument('-optimType', default='sgd', type=str, metavar='optim', help='optimizer')

    parser.add_argument('--lr', default=0.001, type=float,metavar='LR', help='initial learning rate')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--weightDecay', default=5e-4, type=float,metavar='weightDecay', help='weight decay (default: 1e-4)')

    parser.add_argument('--betas', default = (0.9, 0.999), type=float, metavar='betas',help='betas')
    parser.add_argument('--eps', default = 1e-06, type=float, metavar='eps',help='epsilon')
    parser.add_argument('--rho', default = 0.9, type=float, metavar='rho',help='rho')

    #model stuff
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str, metavar='PATH',help='path to storing checkpoints (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--pretrained_file', metavar='pretrained_file', default="")


    #extra model stuff
    parser.add_argument('--model_def', metavar='model_def', default='wrn')
    parser.add_argument('--name', default='deeplab', type=str,help='name of experiment')


    return parser
