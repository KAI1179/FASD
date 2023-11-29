'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from fine_grained_datasets import load_dataset
import os
import argparse
import numpy as np
from utils.AverageMeter import AverageMeter
from models import resnet18
from utils.metric import metric_ece_aurc_eaurc
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults


def seed_it(seed):
    random.seed(seed) #可以注释掉
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #这个懂吧
    torch.backends.cudnn.deterministic = True #确定性固定
    torch.backends.cudnn.benchmark = True #False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  #增加运行效率，默认就是True
    torch.manual_seed(seed)
seed_it(114514)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--alpha', default
=0.9, type=float, help='KD loss alpha')
parser.add_argument('--temperature', default=4, type=int, help='KD loss temperature')
parser.add_argument('--warmup', default=20, type=int, help='warm up epoch')
parser.add_argument("--root", type=str, default="./data")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--classes_num", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--cls", type=bool, default=True)
parser.add_argument('--dataset', default='CUB200', type=str, help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67')
parser.add_argument('--dataroot', default='/data2/xukai_data/dataset/CUB_200_2011/', type=str, help='data directory') # '.../CUB_200_2011/' | '.../StandFordDogs' | '/data2/xukai_data/dataset/'(MIT67)

parser.add_argument("--aug_nums", type=int, default=2)  #

args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

save_path = './checkpoint/FASD_R18_CUB'
save_path_pth = os.path.join(save_path, 'ckpt.pth')

if not args.cls:
    trainloader, valloader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)  ## CUB / MIT / StanFord
else:
    trainloader, valloader = load_dataset(args.dataset, args.dataroot, 'pair', batch_size=args.batch_size)  ## CUB / MIT / StanFord



num_class = trainloader.dataset.num_classes

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')


net = resnet18(num_classes=num_class)
net = net.cuda()

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.pth')
    checkpoint = torch.load(save_path_pth)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
bce_WL = nn.BCEWithLogitsLoss()
ls_loss = LabelSmoothing(smoothing=0.1)
L2Loss = nn.MSELoss()
def divide_features(feature, divide_init):
    global inh_index
    global exp_index
    if divide_init == False:
        length = feature.size(1)

        index = torch.randperm(length).cuda()
        inh_index = index[:int(length / 2)]
        exp_index = index[int(length / 2):]

        print('Inh Index: {}'.format(inh_index.tolist()))
        print('Exp Index: {}'.format(exp_index.tolist()))

    inh_feature = feature.index_select(dim=1, index=inh_index)
    exp_feature = feature.index_select(dim=1, index=exp_index)
    return inh_feature, exp_feature

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1.0 - alpha)

    return KD_loss

def loss_fn_kd_crd(outputs, teacher_outputs, temperature):
    p_s = F.log_softmax(outputs/temperature, dim=1)
    p_t = F.softmax(teacher_outputs/temperature, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (temperature**2) / outputs.shape[0]

    return loss

def gram_matrix(f1, f2):

    f1 = f1.view(f1.size(0), 1, -1)
    f2 = f2.view(f2.size(0), 1, -1)
    tmp = []
    tmp.append(f1)
    tmp.append(f2)
    tmp = torch.cat(tmp, dim=1)
    gram = torch.bmm(tmp, tmp.permute(0, 2, 1))
    return gram

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def hard_select(fea_vec, target):

    fea_vec = normalize(fea_vec, axis=-1)

    fea_vec_t = fea_vec.permute(1, 0)


    cos_dis = torch.matmul(fea_vec, fea_vec_t)
    cos_dis_np = cos_dis.detach().cpu().numpy()

    target_list = target.detach().cpu().numpy().tolist()
    for i in range(cos_dis.size(0)):
        for j in range(cos_dis.size(0)):
            if target_list[j] == target_list[i]:
                cos_dis_np[i][j] = 0

    index = []
    for i in range(cos_dis.size(0)):
        tmp = np.argmax(cos_dis_np[i])
        index.append(tmp)
    fea_vec_rank = fea_vec[index]
    exp_fea = fea_vec - fea_vec_rank
    return exp_fea


def intra_extract(fea_vec, target):
    global catagrey_centre
    global catagrey_num

    batch_catagrey_centre = torch.zeros((num_class, fea_vec.size(1)), device='cuda')
    batch_catagrey_num = torch.zeros((num_class, 1), device='cuda')

    batch_catagrey_num_over_2 = torch.zeros((num_class, 1), device='cuda') ## batch内样本大于2才相减

    batch_fea_centre = []



    target_list = target.detach().cpu().numpy().tolist()

    for i in range(len(target_list)):

        batch_catagrey_centre[target_list[i], :] += fea_vec[i]
        batch_catagrey_num[target[i], :] += 1


    batch_catagrey_centre = torch.div(batch_catagrey_centre, batch_catagrey_num + 1e-10)

    batch_catagrey_num_list = batch_catagrey_num.detach().cpu().numpy().tolist()

    for i in range(len(batch_catagrey_num_list)):
        if batch_catagrey_num_list[i][0] >= 2:
            batch_catagrey_num_over_2[i, :] = 1

    for i in range(len(target_list)):
        tmp_intra = batch_catagrey_centre[target_list[i], :] * batch_catagrey_num_over_2[target_list[i], :]
        batch_fea_centre.append(tmp_intra.reshape(1, -1))

    batch_fea_centre = torch.cat(batch_fea_centre, dim=0)

    fea_sub_intra = fea_vec - batch_fea_centre

    return fea_sub_intra

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)
# Training
save_change = False

catagrey_centre = torch.zeros([100, 512], device='cuda')  # 与数据集/网络 有关
catagrey_num = torch.zeros([100, 1], device='cuda') # 与数据集 有关

def train(epoch, pre_data):
    global save_change
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss_all = 0
    train_loss_sub = 0
    train_loss_intra = 0

    correct_sub = 0
    correct_all = 0
    correct_intra = 0

    total = 0
    for batch_idx, data in enumerate(trainloader):

        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        if pre_data != None:
            pre_inputs, pre_targets = pre_data
            if torch.cuda.is_available():
                pre_inputs = pre_inputs.cuda()
                pre_targets = pre_targets.cuda()

            inputs = torch.cat([inputs[:, 0, ...], pre_inputs[:, 1, ...]])
            targets = torch.cat([targets, pre_targets])
        else:
            inputs = inputs[:, 0, ...]
            targets = targets


        pre_data = data  # 用于下一轮的data

        out_all, exp_fea = net(inputs)

        exp_fea_sub = hard_select(exp_fea, targets)  ##减去类间共有特征
        out_sub = net.fc(exp_fea_sub)

        intra_fea_sub = intra_extract(exp_fea, targets)  ## 减去类内共有特征
        out_intra = net.fc(intra_fea_sub)


        loss_all = criterion(out_all, targets)


        loss_IB = loss_fn_kd(out_sub, targets, out_all.detach(), alpha=1, temperature=4)
        loss_intra = loss_fn_kd(out_intra, targets, out_all.detach(), alpha=1, temperature=4)

        loss = loss_all + loss_IB * 30 + loss_intra * 30  ## final loss: = 1.4 : 27 : 10




        loss = min((epoch+1) / args.warmup, 1.0) * loss

        loss.backward()
        optimizer.step()


        train_loss_all += loss_all.item()
        train_loss_sub += loss_IB.item()
        train_loss_intra += loss_intra.item()


        _, predicted_all = out_all.max(1)
        _, predicted_sub = out_sub.max(1)
        _, predicted_intra = out_intra.max(1)

        total += targets.size(0)

        correct_all += predicted_all.eq(targets).sum().item()
        correct_sub += predicted_sub.eq(targets).sum().item()
        correct_intra += predicted_intra.eq(targets).sum().item()


    epoch_loss_all = train_loss_all / (batch_idx + 1)
    epoch_loss_sub = train_loss_sub / (batch_idx + 1)
    epoch_loss_intra = train_loss_intra / (batch_idx + 1)

    epoch_acc_all = correct_all / total
    epoch_acc_sub = correct_sub / total
    epoch_acc_intra = correct_intra / total


    # print('Train Loss_1: {:.4f} Acc: {:.4f}'.format(epoch_loss_1, epoch_acc_1))
    # print('Train Loss_2: {:.4f} Acc: {:.4f}'.format(epoch_loss_2, epoch_acc_2))
    print('Train Loss_ce: {:.4f} Acc: {:.4f}'.format(epoch_loss_all, epoch_acc_all))
    print('Train epoch_loss_sub: {:.4f} Acc: {:.4f}'.format(epoch_loss_sub, epoch_acc_sub))
    print('Train epoch_loss_intra: {:.4f} Acc: {:.4f}'.format(epoch_loss_intra, epoch_acc_intra))
    print('-' * 20)
    # print('Train Loss_l2: {:.6f}'.format(epoch_loss_l2))

    return pre_data


def test(epoch):
    global best_acc
    global save_change
    net.eval()

    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_losses = AverageMeter()

    targets_list = []
    confidences = []

    correct_all = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # for ECE, AURC, EAURC
            targets_numpy = targets.cpu().numpy()
            targets_list.extend(targets_numpy.tolist())

            # out_all= net(inputs)
            out_all, _ = net(inputs)

            # for ECE, AURC, EAURC
            softmax_predictions = F.softmax(out_all, dim=1)
            softmax_predictions = softmax_predictions.cpu().numpy()
            for values_ in softmax_predictions:
                confidences.append(values_.tolist())


            _, predicted_all = out_all.max(1)
            total += targets.size(0)
            correct_all += predicted_all.eq(targets).sum().item()

            loss_all = criterion(out_all, targets)
            val_losses.update(loss_all.item(), inputs.size(0))

            # Top1, Top5 Err
            err1, err5 = accuracy(out_all.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))

        if is_main_process():
            ece, aurc, eaurc = metric_ece_aurc_eaurc(confidences,
                                                     targets_list,
                                                     bin_size=0.1)

        print('[Epoch {}] [val_loss {:.3f}] [val_top1_acc {:.3f}] [val_top5_acc {:.3f}] [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}] [correct/total {}/{}]'.format(
                epoch,
                val_losses.avg,
                val_top1.avg,
                val_top5.avg,
                ece,
                aurc,
                eaurc,
                correct_all,
                total))

    # Save checkpoint.
    acc = val_top1.avg
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path_pth)
        best_acc = acc
        save_change = True
        print('save change!')


if __name__ == '__main__':
    pre_data = None
    for epoch in range(start_epoch, start_epoch+240):
        pre_data = train(epoch, pre_data)
        test(epoch)
        scheduler.step()
