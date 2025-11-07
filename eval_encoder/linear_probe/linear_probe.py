import math

import warnings

import numpy as np
import torch

from sklearn.metrics import average_precision_score



warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_sharing_strategy('file_system')



def log(msg):
    print(msg)


@torch.no_grad()
def np2pt(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)


@torch.no_grad()
def dense2onehot(arr, depth):
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    arr = torch.nn.functional.one_hot(arr, num_classes=depth)
    return arr


@torch.no_grad()
def float2long(arr):
    arr = arr.long()
    return arr


def search(
    x_train,
    y_train,
    x_test,
    y_test,
    debug,
    num_classes,
    metric = "acc"
    ):
    list_score = []
    list_linear_model = []

    for i in range(5):
        score, linear_model = search_one(x_train, y_train, x_test, y_test, debug, num_classes, metric)
        list_score.append(score)
        list_linear_model.append(linear_model)
    max_idx = int(np.argmax(list_score))
    return list_score[max_idx], list_linear_model[max_idx]


def search_one(
    x_train,
    y_train,
    x_test,
    y_test,
    debug,
    num_classes,
    metric="acc"
    ):

    x_train = np2pt(x_train)
    y_train = np2pt(y_train)
    x_test = np2pt(x_test)
    y_test = np2pt(y_test)

    if metric == "map":
        x_train = x_train.float().cuda()
        x_test = x_test.float().cuda()
        y_train = y_train.long().cuda()
        y_test = y_test.long().cuda()

        if len(y_train.size()) < 2:
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)

            if num_classes > 1:
                y_train = dense2onehot(y_train, num_classes)
                y_test = dense2onehot(y_test, num_classes)
            else:
                y_train = y_train.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

    elif metric == "acc":
        x_train = x_train.float().cuda()
        x_test = x_test.float().cuda()
        y_train = y_train.long().cuda()
        y_test = y_test.long().cuda()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    if debug:
        print(y_train.size())
        print(y_test.size())

    # rough search--narrow down the search
    best_acc = 0
    best_idx_step_1 = None
    best_c = None
    list_lamda_1 = [10**-14, 10**-12, 10**-10, 10**-8, 10**-6, 10**-4, 10**-2, 1, 10**2, 10**4, 10**6]

    for idx, c in enumerate(list_lamda_1):
        acc, _ = train_val(x_train, y_train, x_test, y_test, num_classes , c)
        if acc >= best_acc:
            best_acc = acc
            best_c = list_lamda_1[idx]
            best_idx_step_1 = idx
        if debug:
            log(f"c: {c :<30}, metric: {float(acc)}")

    if best_idx_step_1 is None:
        best_idx_step_1 = 1
        best_c = list_lamda_1[best_idx_step_1]

    left = max(0, best_idx_step_1 - 1)
    c_init = list_lamda_1[left]

    # fine search
    for i in range(8 * 2):
        acc, _ = train_val(
            x_train,
            y_train,
            x_test,
            y_test,
            num_classes,
            c_init,
            metric,
        )
        if debug:
            log(f"c: {c_init :<30}, metric: {float(acc)}")
        if acc > best_acc:
            best_acc = acc
            best_c = c_init

        c_init *= math.log(96, 10)  # step: log(x,base)

    # Finally, evaluation in test
    acc, model_linear = train_val(
            x_train,
            y_train,
            x_test,
            y_test,
            num_classes,
            best_c,
            metric,
    )

    return acc, model_linear


class LogisticRegressionGPU(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


def train_val(x_train, y_train, x_test, y_test, num_classes, C, metirc="acc"):
    embedding_size = x_train.size(1)
    # model
    model_lr_gpu = LogisticRegressionGPU(embedding_size, num_classes)

    model_lr_gpu.train()
    model_lr_gpu.cuda()

    # loss
    if metirc == "map":
        cross_entropy = torch.nn.BCEWithLogitsLoss()
        y_train = y_train.float()
    elif metirc == "acc":
        cross_entropy = torch.nn.CrossEntropyLoss()
    cross_entropy.cuda()

    # optim
    lbfgs = torch.optim.LBFGS(model_lr_gpu.parameters(), max_iter=1000,)

    def l2_regularization(model, l2_alpha):
        l2_loss = []
        for module in model.modules():
            if type(module) is torch.nn.Linear:
                l2_loss.append((module.weight**2).sum() / 2.0)
        return l2_alpha * torch.sqrt(sum(l2_loss))

    def closure():
        lbfgs.zero_grad()
        chunk_size = 32768
        for i in range(len(x_train) // chunk_size + 1):
            end = min(len(x_train), i * chunk_size + chunk_size)
            predict = model_lr_gpu(x_train[i * chunk_size: end])
            ce_loss = cross_entropy(predict, y_train[i * chunk_size: end])
            l2loss = l2_regularization(model_lr_gpu, C)
            loss = ce_loss + l2loss
            loss.backward()
        return loss

    try:
        lbfgs.step(closure)
    except RuntimeError:
        return 0, None

    # eval
    model_lr_gpu.eval()
    x_predict = model_lr_gpu(x_test)
    # print(x_predict)
    # check NAN
    if torch.any(torch.isnan(x_predict)):
        return 0, None

    with torch.no_grad():
        if metirc == "map":
            x_predict.sigmoid_()
            score = metric_map(x_predict.cpu().numpy(), y_test.cpu().numpy())
        elif metirc == "acc":
            score = metric_acc(x_predict.cpu(), y_test.cpu())
        else:
            raise RuntimeError("Only support BCE")

    return score, model_lr_gpu


@torch.no_grad()
def metric_map(probs: np.ndarray, gts: np.ndarray):
    gts = gts.astype(np.int32)

    mAP = []
    for i in range(probs.shape[1]):
        sum_gt = np.sum(gts[:, i])
        if sum_gt < 1:
            continue
        else:
            ap = average_precision_score(gts[:, i], probs[:, i])
            mAP.append(ap)
    score = np.mean(mAP) * 100
    return score


@torch.no_grad()
def metric_acc(probs, gts, topk=(1,)):
    maxk = max(topk)
    batch_size = gts.size(0)
    _, pred = probs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(gts.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res[0]
