import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import gc

import torch_geometric
import yaml
import mlp
from optimizer import construct_optimizer
import data_proccess
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from sklearn import metrics
import random
from torch_geometric.loader import DataLoader as PyG_DataLoader
from torch_geometric.loader import DataListLoader as PyG_ListLoader
import graph_dataset
from graph_dataset import TwoGraphsData
import xl_transforms
import gnn
import wandb
import temperature_scaling
import sys
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser')
import general_utils
import cross_link


torch.multiprocessing.set_sharing_strategy('file_system')
# matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = "cpu"


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_cm(y_true, y_pred_prob):
    y_pred_tags = np.argmax(y_pred_prob, axis=1).astype(np.int32)
    th = ['0 - 12', '12 - 20', '20 - 28', '28 - 45']
    general_utils.plot_confusion_matrix(y_true.tolist(), y_pred_tags, th, title="Confusion Matrix (Validation Set)")
    general_utils.plot_confusion_matrix(y_true.tolist(), y_pred_tags, th, normalize=True,
                                        title="Normalized Confusion Matrix (Validation Set)")
    plt.show()


def load_config(args=None, cfg_path='/cs/labs/dina/seanco/xl_mlp_nn/configs/gnn_separate_39f_angles.yaml'):
   # Setup cfg.
    if args is None:
        cfg_file = cfg_path
    else:
        cfg_file = args.cfg_file
    # with open(args.cfg_file, "r") as f:
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

        return cfg


def analyze_wrong_predicted_features(features, distances):
    x = features[:, -1]
    x[x == 0.103] = 0
    x[x == 0.114] = 1
    x[x == 0.334] = 2
    plt.hist(x, bins=[0, 1, 2])
    plt.title("Spacer length in wrongly predicted features")
    # plt.show()
    plt.hist(distances, 10)
    plt.title("XL Distances of wrongly predicted features")
    # plt.show()


def plot_res(train_loss, val_loss):
    train_loss = np.asarray(train_loss)
    val_loss = np.asarray(val_loss)
    x = np.arange(train_loss.shape[0])
    plt.plot(x, train_loss)
    # plt.show()

    plt.plot(np.arange(val_loss.shape[0]), val_loss)
    # plt.show()


def plot_roc_curve(fpr, tpr, auc, cfg, i=0):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    out_dir = '/cs/labs/dina/seanco/xl_mlp_nn/plots/'
    fig_name = out_dir + cfg['name'] + '_roc_' + str(i) + '.png'
    # plt.show()

    plt.savefig(fig_name)


def calc_precision_recall_roc(confusion_matrix, n_classes, pred_prob, true_y, cfg, i):
    if n_classes > 2:
        return 0, 0, 0
    tp = confusion_matrix[1, 1]
    fn = confusion_matrix[1, 0]
    fp = confusion_matrix[0, 1]
    tn = confusion_matrix[0, 0]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_prob)
    auc = metrics.roc_auc_score(true_y, pred_prob)
    # plot_roc_curve(fpr, tpr, auc, cfg, i)
    return auc, recall, precision


def loss_corn(logits, y_train):
    num_classes = logits.shape[1] + 1
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred)*train_labels + (F.logsigmoid(pred) - pred)*(1-train_labels))
        losses += loss
    return losses/num_examples


def label_from_corn_logits(logits):
    """ Converts logits to class labels.
    This is function is specific to CORN.
    """
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def prediction2label(pred):
    """Convert ordinal predictions to class labels, e.g.
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    pred = torch.round(pred)
    pred = (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1
    pred[pred < 0] += 1
    return pred


def ordinal_regression(predictions, targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""
    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)
    predictions = torch.round(predictions)
    modified_pred = torch.zeros_like(predictions)
    for i in range(1, predictions.shape[1]):
        modified_pred[:, i] = predictions[:, i-1] * predictions[:, i]
    # predictions = (predictions > 0.5).cumprod(axis=1)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target+1] = 1

    # return torch.nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1)
    return torch.mean(torch.nn.MSELoss(reduction='none')(modified_pred, modified_target).sum(axis=1))
    # return criterion(predictions, modified_target)


def record_angles_accuracy(outputs, labels, wandb_, epoch, cfg):
    omega_acc, theta_acc, phi_acc = None, None, None
    if cfg['omega_pred']:
        omega_pred_prob = torch.softmax(outputs[0], dim=1)
        _, o_pred_tags = torch.max(omega_pred_prob, dim=1)
        correct_pred = (o_pred_tags == labels[0][:, 0]).float()
        omega_acc = correct_pred.sum() / len(labels[0][:, 0])
    if cfg['theta_pred']:
        theta_pred_prob_a = torch.softmax(outputs[1][:, :24], dim=1)
        theta_pred_prob_b = torch.softmax(outputs[1][:, 24:], dim=1)
        _, t_pred_tags = torch.max(theta_pred_prob_a, dim=1)
        correct_pred_a = (t_pred_tags == labels[1][:, 0]).float()
        _, t_pred_tags = torch.max(theta_pred_prob_b, dim=1)
        correct_pred_b = (t_pred_tags == labels[1][:, 2]).float()
        theta_acc = (correct_pred_a.sum() + correct_pred_b.sum()) / (len(labels[1]) * 2)
    if cfg['phi_pred']:
        phi_pred_prob_a = torch.softmax(outputs[2][:, :12], dim=1)
        phi_pred_prob_b = torch.softmax(outputs[2][:, 12:], dim=1)
        _, p_pred_tags = torch.max(phi_pred_prob_a, dim=1)
        correct_pred_a = (p_pred_tags == labels[2][:, 0]).float()
        _, p_pred_tags = torch.max(phi_pred_prob_b, dim=1)
        correct_pred_b = (p_pred_tags == labels[2][:, 2]).float()
        phi_acc = (correct_pred_a.sum() + correct_pred_b.sum()) / (len(labels[2]) * 2)
    acc = [omega_acc, theta_acc, phi_acc]
    title_lst = ['Omega', 'Theta', 'Phi']
    for i in range(len(acc)):
        if acc[i] is not None:
            wandb_.log({'epoch': epoch, f"{title_lst[i]} Angle Accuracy / Train": acc[i]})


def calc_accuracy(output, labels, confusion_matrix=None, n_classes=2, distances=None, loss_type='bce', temp_s=False):
    if loss_type == 'ordinal':
        y_pred_tags = prediction2label(output)
        y_pred_prob = output
    elif loss_type == 'corn':
        y_pred_tags = label_from_corn_logits(output)
        y_pred_prob = output
    elif n_classes > 2:
        y_pred_prob = torch.softmax(output, dim=1)
        _, y_pred_tags = torch.max(y_pred_prob, dim=1)
    else:
        y_pred_prob = torch.sigmoid(output)
        y_pred_tags = torch.round(y_pred_prob)
    if temp_s:
        y_pred_prob = output
    correct_pred = (y_pred_tags == labels).float()
    # if distances is not None:
    #     distances = distances[~correct_pred.bool().flatten()]
    if confusion_matrix is not None:
        for t, p in zip(labels, y_pred_tags):
            confusion_matrix[t.long(), p.long()] += 1
    return correct_pred.sum() / len(labels), y_pred_prob, distances, correct_pred.bool().flatten()


def calculate_loss_multi_head_bins(data, model, criterion, loss_weights, losses):
    inputs = data.to(device, non_blocking=True)
    outputs = model(inputs)
    labels = [inputs.y_ca, inputs.y_cb, inputs.y_omega, inputs.y_theta, inputs.y_phi]
    cur_total_loss = 0
    batch_size = outputs[0].shape[0]

    for k in range(len(criterion)):
        if criterion[k] is not None:
            if k < 2:
                tmp_loss = criterion[k](outputs[k], labels[k])
            elif k == 2:
                labels[k] = torch.reshape(labels[k], (batch_size, labels[k].shape[0] // batch_size))
                tmp_loss = criterion[k](outputs[k], labels[k][:, 0].long())
            else:
                labels[k] = torch.reshape(labels[k], (batch_size, labels[k].shape[0] // batch_size))
                h = outputs[k].shape[1] // 2
                tmp_loss = (criterion[k](outputs[k][:, :h], labels[k][:, 0].long()) +
                            criterion[k](outputs[k][:, h:], labels[k][:, 2].long())) / 2
            losses[k] += tmp_loss
            tmp_loss = loss_weights[k] * tmp_loss
            cur_total_loss += tmp_loss
    return cur_total_loss, inputs, outputs, labels


def calculate_loss_multi_head(data, model, criterion, loss_weights, losses, is_sample_weights=False, is_eval=False):
    inputs = data.to(device, non_blocking=True)
    outputs = model(inputs)
    labels = [inputs.y_ca, inputs.y_cb, inputs.y_omega, inputs.y_theta, inputs.y_phi]
    cur_total_loss = 0

    for i in range(2, len(criterion)):
        if criterion[i] is not None:
            labels[i] = torch.reshape(labels[i], outputs[i].shape).float()
    for k in range(len(criterion)):
        if criterion[k] is not None:
            tmp_loss = criterion[k](outputs[k], labels[k])
            if k == 0 and is_sample_weights:
                if not is_eval:
                    # tmp_loss = (tmp_loss * inputs.ca_error).mean()
                    tmp_loss = (tmp_loss * ((10 * inputs.xl_type[:, 1]) + 1)).mean()
                else:
                    tmp_loss = tmp_loss.mean()
            losses[k] += tmp_loss
            tmp_loss = loss_weights[k] * tmp_loss
            cur_total_loss += tmp_loss
    return cur_total_loss, inputs, outputs, labels


def calculate_loss(is_gnn, data, model, criterion, n_classes, loss_weights, cfg, is_connected=True, losses=None,
                   is_eval=False):
    if is_connected is False:
        if cfg['angles'] == 'bins':
            return calculate_loss_multi_head_bins(data, model, criterion, loss_weights, losses)
        return calculate_loss_multi_head(data, model, criterion, loss_weights, losses, cfg['sample_weight'], is_eval)
    if is_gnn:
        inputs = data.to(device)
        outputs = model(inputs)
        labels = inputs.y
    else:
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)

    if n_classes == 2:
        labels = torch.reshape(labels, outputs.shape)
        labels = labels.float()
    else:
        labels = labels.long()

    cur_loss = criterion(outputs, labels)
    return cur_loss, inputs, outputs, labels


def record_losses(losses, wandb_, len_data, epoch, phase):
    if losses is None:
        return
    losses /= len_data
    title_lst = ['Ca', 'Cb', 'Omega', 'Theta', 'Phi']
    for i in range(len(losses)):
        if losses[i] != 0:
            wandb_.log({'epoch': epoch, f"{title_lst[i]} loss / {phase}": losses[i]})


def eval_epoch(model, val_loader, criterion, epoch, val_loss, wandb_, is_regression, n_classes, acc,
               max_epoch, cfg, is_gnn=True, temp_scaling=False, record_in_wandb=True):
    is_connected = cfg['is_connected']
    is_ca = 'ca_pred' not in cfg or cfg['ca_pred']
    is_cb = cfg['cb_pred']
    loss_weights = None
    losses = None
    per_class_acc = [0, 0]
    if is_connected is False:
        losses = torch.zeros(5, device=device)
        loss_weights = cfg['loss_head_weights']
    model.eval()
    with torch.no_grad():
        predicted_probabilities = list()
        true_labels = list()
        confusion_matrix = torch.zeros(n_classes, n_classes)
        confusion_matrix_cb = torch.zeros(n_classes, n_classes)
        total_loss = 0
        ca_pred, cb_pred, ca_labels, cb_labels = [], [], [], []

        # data_loader = tqdm(val_loader, position=0, leave=True)
        for data in val_loader:
            # with torch.cuda.amp.autocast():
            cur_loss, inputs, outputs, labels = calculate_loss(is_gnn, data, model, criterion,
                                                               n_classes, loss_weights, cfg,
                                                               is_connected, losses, True)
            total_loss += cur_loss
            if is_connected is False:
                ca_pred.append(outputs[0])
                ca_labels.append(labels[0])
                # tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs[0].to('cpu'), labels[0].to('cpu'),
                #                                                          confusion_matrix, n_classes)
                if is_cb:
                    cb_pred.append(outputs[1])
                    cb_labels.append(labels[1])
                    # tmp_acc_cb, y_prob_cb, _, _ = calc_accuracy(outputs[1].to('cpu'), labels[1].to('cpu'),
                    #                                             None, n_classes)
                    # accuracy_cb += tmp_acc_cb
            else:
                ca_pred.append(outputs)
                ca_labels.append(labels)
                # tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs.to('cpu'), labels.to('cpu'),
                #
            # data_loader.set_postfix({'loss': cur_loss.item(), "Epoch": epoch})

        total_loss = total_loss / len(val_loader)
        if not is_regression:
            if is_ca:
                accuracy, y_prob, wrong_dist, correct_idx = calc_accuracy(torch.cat(ca_pred, dim=0).to('cpu', non_blocking=True),
                                                                          torch.cat(ca_labels, dim=0).to('cpu', non_blocking=True),
                                                                          confusion_matrix, n_classes, None, cfg['loss_type'],
                                                                          temp_s=temp_scaling)
                acc.append(accuracy)
                per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
                if record_in_wandb:
                    wandb_.log({'epoch': epoch, 'Accuracy / Validation': accuracy})
                    wandb_.log({'epoch': epoch, "Per-Class-Accuracy / Validation":
                               {ii: ac for ii, ac in enumerate(per_class_acc)}})
                print(confusion_matrix)
                print('per class acc: ', per_class_acc)
                if epoch == max_epoch:
                    if not temp_scaling:
                        predicted_probabilities = np.asarray(y_prob.to('cpu', non_blocking=True)).flatten()
                        true_labels = np.asarray(torch.cat(ca_labels, dim=0).int().to('cpu')).flatten()
                    else:
                        predicted_probabilities = y_prob.to('cpu', non_blocking=True)
                        true_labels = torch.cat(ca_labels, dim=0).int().to('cpu')
            if is_cb:
                accuracy_cb, y_prob_cb, _, _ = calc_accuracy(torch.cat(cb_pred, dim=0).to('cpu', non_blocking=True),
                                                             torch.cat(cb_labels, dim=0).to('cpu', non_blocking=True),
                                                             confusion_matrix_cb, n_classes, None, cfg['loss_type'])
                per_class_acc_cb = confusion_matrix_cb.diag() / confusion_matrix_cb.sum(1)
                if record_in_wandb:
                    wandb_.log({'epoch': epoch, 'Accuracy_CB / Validation': accuracy_cb})
                    wandb_.log({'epoch': epoch, "Per-Class-Accuracy_CB / Validation":
                               {ii: ac for ii, ac in enumerate(per_class_acc_cb)}})

        val_loss.append(total_loss)
        if record_in_wandb:
            wandb_.log({'epoch': epoch, 'Loss / Validation': total_loss})
            record_losses(losses.to('cpu', non_blocking=True), wandb_, len(val_loader), epoch, 'Validation')
        print("Finished Evaluation epoch:", epoch, " loss: ", total_loss)
        return per_class_acc, predicted_probabilities, true_labels, confusion_matrix


def train_epoch(model, train_loader, criterion, _optimizer, epoch, train_loss, wandb_, scheduler, acc,
                n_classes, max_epoch, cfg, is_gnn=True):
    torch.backends.cudnn.benchmark = True
    is_connected = cfg['is_connected']
    is_cb = cfg['cb_pred']
    is_ca = 'ca_pred' not in cfg or cfg['ca_pred']
    losses = None
    loss_weights = None
    if is_connected is False:
        loss_weights = cfg['loss_head_weights']
        losses = torch.zeros(5, device=device)
    model.train()
    total_loss = 0
    ca_pred, cb_pred, ca_labels, cb_labels = [], [], [], []
    omega_pred, omega_labels, theta_pred, theta_labels, phi_pred, phi_labels = [], [], [], [], [], []
    confusion_matrix = torch.zeros(n_classes, n_classes)
    inter_or_intra = [0, 0]
    # data_loader = tqdm(train_loader, position=0, leave=True)
    for data in train_loader:
        inter_or_intra[0] += (data.xl_type.T[0] == 0).sum().item()
        inter_or_intra[1] += (data.xl_type.T[0] == 1).sum().item()
        _optimizer.zero_grad(set_to_none=True)
        # with torch.cuda.amp.autocast():
        cur_loss, inputs, outputs, labels = calculate_loss(is_gnn, data, model, criterion,
                                                           n_classes, loss_weights, cfg, is_connected,
                                                           losses, False)
        cur_loss.backward()
        _optimizer.step()

        # running_loss += cur_loss.item()
        total_loss += cur_loss
        if is_connected is False:
            ca_pred.append(outputs[0].to('cpu', non_blocking=True))
            ca_labels.append(labels[0].to('cpu', non_blocking=True))
            # tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs[0].to('cpu'), labels[0].to('cpu'),
            #                                                          confusion_matrix, n_classes)
            if is_cb:
                cb_pred.append(outputs[1].to('cpu', non_blocking=True))
                cb_labels.append(labels[1].to('cpu', non_blocking=True))
                # tmp_acc_cb, y_prob_cb, _, _ = calc_accuracy(outputs[1].to('cpu'), labels[1].to('cpu'),
                #                                             None, n_classes)
                # accuracy_cb += tmp_acc_cb
        else:
            ca_pred.append(outputs)
            ca_labels.append(labels)
            # tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs.to('cpu'), labels.to('cpu'),
            #                                                  confusion_matrix, n_classes)
        if cfg['angles'] == 'bins':
            omega_pred.append(outputs[2])
            omega_labels.append(labels[2])
            theta_pred.append(outputs[3])
            theta_labels.append(labels[3])
            phi_pred.append(outputs[4])
            phi_labels.append(outputs[4])
        # if dist_in_wrong_pred is None:
        #     dist_in_wrong_pred = wrong_dist
        # else:
        #     dist_in_wrong_pred = torch.cat((dist_in_wrong_pred, wrong_dist))
        # if epoch == max_epoch:
        #     if wrong_pred_feat is None:
        #         wrong_pred_feat = inputs.y.to('cpu')[~correct_idx]
        #     else:
        #         wrong_pred_feat = torch.cat((wrong_pred_feat, inputs.y.to('cpu')[~correct_idx]), dim=0)
        # accuracy += tmp_acc

        # data_loader.set_postfix({'loss': cur_loss.item()})
        # if (i + 1) % 20 == 0:
        #     running_loss /= 20
        #     print(f"[Train] Epoch: {epoch} Batch: {i+1} Loss: {running_loss:.3f}")
        #     running_loss = 0.0

    if scheduler is not None:
        scheduler.step()
    total_loss /= len(train_loader)
    train_loss.append(total_loss)
    wandb_.log({'epoch': epoch, 'Loss / Train': total_loss})
    if is_ca:
        accuracy, y_prob, wrong_dist, correct_idx = calc_accuracy(torch.cat(ca_pred, dim=0).to('cpu', non_blocking=True),
                                                                  torch.cat(ca_labels, dim=0).to('cpu', non_blocking=True),
                                                                  confusion_matrix, n_classes, None, cfg['loss_type'])
        acc.append(accuracy)
        wandb_.log({'epoch': epoch, 'Accuracy / Train': accuracy})
        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        wandb_.log({'epoch': epoch, "Per-Class-Accuracy / Train": {ii: ac for ii, ac in enumerate(per_class_acc)}})
        print(confusion_matrix)
        print('per class acc Train: ', per_class_acc)
        print("Finished epoch:", epoch, " loss: ", total_loss, " accuracy: ", accuracy)
    if is_cb:
        accuracy_cb, y_prob_cb, _, _ = calc_accuracy(torch.cat(cb_pred, dim=0).to('cpu', non_blocking=True),
                                                     torch.cat(cb_labels, dim=0).to('cpu', non_blocking=True),
                                                     None, n_classes, None, cfg['loss_type'])
        # accuracy_cb = accuracy_cb / len(train_loader)
        wandb_.log({'epoch': epoch, 'Accuracy_CB / Train': accuracy_cb})
    record_losses(losses.to('cpu', non_blocking=True), wandb_, len(train_loader), epoch, 'Train')
    if cfg['angles'] == 'bins':
        angles_out = [torch.cat(omega_pred, dim=0).to('cpu', non_blocking=True),
                      torch.cat(theta_pred, dim=0).to('cpu', non_blocking=True),
                      torch.cat(phi_pred, dim=0).to('cpu', non_blocking=True)]
        angles_labels = [torch.cat(omega_labels, dim=0).to('cpu', non_blocking=True),
                         torch.cat(theta_labels, dim=0).to('cpu', non_blocking=True),
                         torch.cat(phi_labels, dim=0).to('cpu', non_blocking=True)]
        record_angles_accuracy(angles_out, angles_labels, wandb_, epoch, cfg)
    print(f"inter samples: {inter_or_intra[0]}")
    print(f"intra samples: {inter_or_intra[1]}")


def update_loss_weights(cfg):
    weights = cfg['loss_head_weights']
    weights[2] *= 1.5
    weights[3] *= 1.5
    weights[4] *= 3
    if cfg['num_classes'] <= 5:
        scale_mult = cfg['num_classes'] / 2
    else:
        scale_mult = cfg['num_classes'] / 3
    for i in range(2, 5):
        weights[i] *= scale_mult
    cfg['loss_head_weights'] = weights


def get_multi_head_criterion(cfg, weights):
    ca_crit, cb_crit, omega_crit, theta_crit, phi_crit = None, None, None, None, None
    calc_weights = [None, None]
    if weights is not None:
        for i in range(len(weights)):
            if cfg['num_classes'] == 2:
                calc_weights[i] = (torch.as_tensor(weights[i][0] / weights[i][1], dtype=torch.float))
            else:
                total_sum = np.sum(weights[i])
                calc_weights[i] = (torch.from_numpy((total_sum - weights[i]) / weights[i]).float().to(
                    device))
    if 'ca_pred' in cfg and cfg['ca_pred']:
        if cfg['num_classes'] > 2:
            if cfg['loss_type'] == 'ordinal':
                ca_crit = ordinal_regression
            elif cfg['loss_type'] == 'corn':
                ca_crit = loss_corn
            elif cfg['sample_weight']:
                ca_crit = torch.nn.CrossEntropyLoss(reduction='none', weight=calc_weights[0])
            else:
                ca_crit = torch.nn.CrossEntropyLoss(reduction='mean', weight=calc_weights[0])
            if cfg['cb_pred']:
                if cfg['loss_type'] == 'ordinal':
                    cb_crit = ordinal_regression
                elif cfg['loss_type'] == 'corn':
                    cb_crit = loss_corn
                else:
                    cb_crit = torch.nn.CrossEntropyLoss(reduction='mean', weight=calc_weights[1])
        else:
            ca_crit = torch.nn.BCEWithLogitsLoss(pos_weight=calc_weights[0])
            if cfg['cb_pred']:
                cb_crit = torch.nn.BCEWithLogitsLoss(pos_weight=calc_weights[1])
    if cfg['omega_pred']:
        if cfg['angles'] == 'bins':
            omega_crit = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            omega_crit = torch.nn.MSELoss()
    if cfg['theta_pred']:
        if cfg['angles'] == 'bins':
            theta_crit = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            theta_crit = torch.nn.MSELoss()
    if cfg['phi_pred']:
        if cfg['angles'] == 'bins':
            phi_crit = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            phi_crit = torch.nn.MSELoss()
    return [ca_crit, cb_crit, omega_crit, theta_crit, phi_crit], False


def get_criterion(cfg, data=None):
    if data is not None and (cfg['weight'] is True):
        if cfg['model_type'] == 'gnn':
            weights = data.dataset.get_label_count(data.indices)
            if not cfg['is_connected']:
                return get_multi_head_criterion(cfg, weights)
        else:
            weights = data.get_label_count()
        if cfg['num_classes'] == 2:
            weights = torch.as_tensor(weights[0] / weights[1], dtype=torch.float)
        else:
            total_sum = np.sum(weights)
            weights = torch.from_numpy((total_sum - weights) / weights).float().to(device)
    else:
        weights = None
    if not cfg['is_connected']:
        return get_multi_head_criterion(cfg, weights)
    if cfg['loss_type'] == 'mse':
        is_regression = True
        criterion = torch.nn.MSELoss()
    else:
        is_regression = False
        if cfg['num_classes'] > 2:
            criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weights)
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    return criterion, is_regression


def get_scheduler(cfg, _optimizer):
    if cfg['scheduler'] == 'steps':
        steps = cfg['steps']
        gama = cfg['gamma']
        scheduler = MultiStepLR(_optimizer, steps, gama)
    else:
        scheduler = None
    return scheduler


def add_hparams_to_tb(writer, cfg, loss, total_accuracy, per_class_acc, auc, recall, precision):
    writer.add_hparams({'optimizer': cfg['optimizing_method'],
                        'lr': cfg['base_lr'],
                        'weight_decay': cfg['weight_decay'],
                        'scheduler': cfg['scheduler'],
                        # 'class_th': cfg['MODEL']['DISTANCE_TH_CLASSIFICATION'],
                        'dropout': cfg['dropout'],
                        'batch_size': cfg['batch_size'],
                        'n_hidden_layers': cfg['hidden_layers'],
                        'class_weight': cfg['weight']},
                       {'hparam/loss': loss,
                        'hparam/accuracy': total_accuracy,
                        'hparam/acc_class_0': per_class_acc[0],
                        'hparam/acc_class_1': per_class_acc[1],
                        'hparam/AUC': auc,
                        'hparam/recall': recall,
                        'hparam/precision': precision})


def load_data(cfg, train_data, eval_data=None, shuffle=True):
    val_loader = None
    train_sampler = None
    if cfg['loss_type'] == 'corn':
        train_sampler = graph_dataset.ImbalancedDatasetSampler(dataset=train_data,
                                                               callback_get_label=graph_dataset.XlPairGraphDataset.get_ca_labels)
        shuffle = False
    elif cfg['sample'] and train_data != eval_data and cfg['xl_type_feature']:
        inter_idx = (train_data.dataset.data.xl_type.T[0] == 0).nonzero(as_tuple=True)[0]
        train_inter_idx = list(set(inter_idx.tolist()).intersection(train_data.indices.tolist()))
        train_inter_idx = torch.tensor([(train_data.indices == i).nonzero(as_tuple=True)[0] for i in train_inter_idx])
        weights = torch.ones(len(train_data.indices))
        weights[train_inter_idx] = 15
        generator = torch.Generator().manual_seed(3407)
        train_sampler = WeightedRandomSampler(weights, len(weights), generator=generator)
        shuffle = False
    if cfg['debug']:
        num_workers = 1
    else:
        num_workers = len(os.sched_getaffinity(0)) - 1
    # num_workers = 1
    if cfg['model_type'] == 'gnn':
        if cfg['is_connected']:
            train_loader = PyG_DataLoader(train_data, cfg['batch_size'], shuffle=shuffle,
                                          num_workers=num_workers, pin_memory=True, sampler=None)
            if eval_data is not None:
                val_loader = PyG_DataLoader(eval_data, cfg['batch_size'], shuffle=shuffle,
                                            num_workers=num_workers, pin_memory=True)
        else:
            if torch.cuda.device_count() > 1 and cfg['multi_gpu'] and eval_data is not None:
                train_loader = PyG_ListLoader(train_data, cfg['batch_size'], shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True, sampler=train_sampler)
                val_loader = PyG_ListLoader(eval_data, cfg['batch_size'], shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=True)
            else:
                train_loader = PyG_DataLoader(train_data, cfg['batch_size'], shuffle=shuffle,
                                              num_workers=num_workers, follow_batch=['x_a', 'x_b'],
                                              pin_memory=True, sampler=train_sampler)
                if eval_data is not None:
                    val_loader = PyG_DataLoader(eval_data, cfg['batch_size'], shuffle=shuffle,
                                                num_workers=num_workers, follow_batch=['x_a', 'x_b'],
                                                pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(eval_data, batch_size=cfg['batch_size'], shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_labels_th(cfg):
    n_classes = cfg['num_classes']
    th = cfg['distance_th_classification']
    if th == 'manual':
        if n_classes == 2:
            th = [random.choice([12, 14, 17, 18, 20])]
        elif n_classes == 3:
            th = random.choice([[12, 18], [13, 18], [13, 20], [12, 20], [13, 21]])
        elif n_classes == 4:
            # th = random.choice([[12, 18, 22], [11, 17, 20], [11, 16, 21], [10, 17, 23], [13, 19, 23]])
            th = [11.98239517, 16.39906025, 22.83931446]
        elif n_classes == 5:
            # th = random.choice([[10, 14, 18, 22], [11, 17, 21, 25], [11, 14, 17, 22],
            #                      [10, 15, 18, 23], [10, 14, 19, 23]])
            th = [8, 12, 16, 30]
        elif n_classes == 6:
            th = [8, 12, 18, 25, 32]
        elif n_classes == 7:
            th = random.choice([[9, 13, 15, 17, 20, 25], [11, 15, 17, 19, 21, 25], [9, 12, 14, 16, 18, 22],
                                 [10, 13, 17, 20, 23, 28], [12, 17, 19, 21, 25, 29]])
        elif n_classes == 18:
            th = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
        elif n_classes == 12:
            th = [7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]
        else:
            th = random.sample(range(8, 32), n_classes)
            th.sort()
    return th


def split_inter_and_intra_dataset(data, cfg):
    generator = torch.Generator().manual_seed(3407)
    inter_idx = (data.data.xl_type.T[0] == 0).nonzero(as_tuple=True)[0]
    inter_idx = inter_idx[torch.randperm(len(inter_idx), generator=generator)]
    intra_idx = (data.data.xl_type.T[0] == 1).nonzero(as_tuple=True)[0]
    intra_idx = intra_idx[torch.randperm(len(intra_idx), generator=generator)]
    intra_lengths = get_datasets_length(cfg, intra_idx, 0.1, 0.1)
    inter_lengths = get_datasets_length(cfg, inter_idx, 0.2, 0.1)
    prev_ter, prev_tra, accumulate_ter, accumulate_tra = 0, 0, 0, 0
    subsets = []
    for i in range(len(intra_lengths)):
        accumulate_tra += intra_lengths[i]
        accumulate_ter += inter_lengths[i]
        idx = torch.cat([intra_idx[prev_tra: accumulate_tra], inter_idx[prev_ter: accumulate_ter]])
        subsets.append(torch.utils.data.Subset(data, indices=idx))
        prev_tra = accumulate_tra
        prev_ter = accumulate_ter
    return subsets


def get_datasets_length(cfg, data, eval_fraction, test_fraction):
    if cfg['debug']:
        eval_len, test_len = int(len(data) * 0.01), int(len(data) * 0.98)
    elif cfg['data_type'] == 'predict':
        eval_len, train_len, test_len = int(len(data)), 0, 0
    else:
        eval_len, test_len = int(len(data) * eval_fraction), int(len(data) * test_fraction)
    train_len = int(len(data) - (eval_len + test_len))
    return train_len, eval_len, test_len


def split_dataset(cfg, data, eval_fraction, test_fraction):
    generator = torch.Generator().manual_seed(3407)
    train_len, eval_len, test_len = get_datasets_length(cfg, data, eval_fraction, test_fraction)
    train_data, eval_data, test_data = random_split(data, [train_len, eval_len, test_len], generator)
    return train_data, eval_data, test_data


def get_dataset(cfg, wandb_=None):
    th = get_labels_th(cfg)
    transform = None
    if cfg['transform']:
        transform = xl_transforms.FlipTransform()
    if cfg['model_type'] == 'gnn':
        if cfg['is_connected']:
            data = graph_dataset.XlGraphDataset(cfg, th)
        else:
            data = graph_dataset.XlPairGraphDataset(cfg, th, transform=transform)
        if cfg['xl_type_feature']:
            train_data, eval_data, test_data = split_inter_and_intra_dataset(data, cfg)
        else:
            train_data, eval_data, test_data = split_dataset(cfg, data, 0.1, 0.1)
        if cfg['data_type'] == 'test':
            return test_data
        elif cfg['data_type'] == 'predict':
            return eval_data, eval_data
    elif cfg['dataset'] == "linker_as_label":
        train_data = data_proccess.LinkerAsLabelDS(cfg, is_train=True)
        eval_data = data_proccess.LinkerAsLabelDS(cfg, is_train=False)
    else:
        train_data = data_proccess.FeatDataset(cfg, is_train=True)
        eval_data = data_proccess.FeatDataset(cfg, is_train=False)
    if wandb_ is not None:
        wandb_.log({"label distance th": cfg['distance_th_classification']})
    return train_data, eval_data


def get_model(cfg, config=None):
    if cfg['model_type'] == 'gnn':
        return gnn.create_model(cfg)
    return mlp.create_model(cfg, config)


def temperature_scaling_calibration(model, cfg, wandb_, epoch, _optimizer, loss, val_loader=None, criterion=None):
    print('model calibration')

    if val_loader is None:
        train_data, eval_data = get_dataset(cfg, wandb_)
        train_loader, val_loader = load_data(cfg, train_data, eval_data)
        criterion = get_criterion(cfg, train_data)[0][0]
    per_class_acc, predicted_probabilities, true_labels, confusion_matrix = eval_epoch(model,
                                                                                       val_loader, [criterion],
                                                                                       cfg['epochs'], [], wandb_,
                                                                                       False, cfg['num_classes'], [],
                                                                                       cfg['epochs'], cfg,
                                                                                       temp_scaling=True)
    ece = temperature_scaling._ECELoss()
    ece_score_before = ece(predicted_probabilities, true_labels)

    scaled_model = temperature_scaling.ModelWithTemperature(model)
    scaled_model.set_temperature(val_loader)

    per_class_acc, predicted_probabilities, true_labels, confusion_matrix = eval_epoch(scaled_model,
                                                                                       val_loader, [criterion],
                                                                                       cfg['epochs'], [], wandb_,
                                                                                       False, cfg['num_classes'], [],
                                                                                       cfg['epochs'], cfg,
                                                                                       temp_scaling=True)
    ece_score_after = ece(predicted_probabilities, true_labels)
    print(f"ECE before calibrating: {ece_score_before}, after: {ece_score_after}")
    torch.save({'epoch': epoch, 'model_state_dict': scaled_model.state_dict(),
                'optimizer_state_dict': _optimizer.state_dict(), 'loss': loss},
               f"models/{cfg['name']}_scaled")
    return scaled_model


def fine_tune_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.GNN_A.convs[-1].parameters():
        param.requires_grad = True
    for param in model.GNN_B.convs[-1].parameters():
        param.requires_grad = True
    for param in model.lin1.parameters():
        param.requires_grad = True
    for param in model.lin2.parameters():
        param.requires_grad = True
    for param in model.lin3.parameters():
        param.requires_grad = True


def stats_of_inter_xl_performance(model, eval_data, criterion, wandb_, cfg, i, out_size):
    inter_idx = (eval_data.dataset.data.xl_type.T[0] == 0).nonzero(as_tuple=True)[0]
    val_inter_idx = list(set(inter_idx.tolist()).intersection(eval_data.indices.tolist()))
    val_inter_data = torch.utils.data.Subset(eval_data.dataset, val_inter_idx)
    data_loader, _ = load_data(cfg, val_inter_data, val_inter_data)
    val_loss, eval_accuracy = list(), list()
    per_class_acc, predicted_probabilities, true_labels, confusion_matrix = eval_epoch(model, data_loader, criterion,
                                                                                       cfg['epochs'] - 1, val_loss, wandb_,
                                                                                       False, cfg['num_classes'], eval_accuracy,
                                                                                       cfg['epochs'] - 1, cfg, record_in_wandb=False)
    wandb_.log({'Inter_xl_accuracy': eval_accuracy[-1]})
    log_results(cfg, wandb_, confusion_matrix, predicted_probabilities, true_labels, i, out_size, 'Inter Xl ')


def log_results(cfg, wandb_, confusion_matrix, predicted_probabilities, true_labels, i, out_size, title=''):
    if 'ca_pred' in cfg and cfg['ca_pred']:
        auc_roc, recall, precision = calc_precision_recall_roc(confusion_matrix, cfg['num_classes'],
                                                               predicted_probabilities, true_labels, cfg, i)
        if cfg['num_classes'] == 2:
            wandb_.log({"pr": wandb.plot.pr_curve(true_labels,
                                                  np.reshape(predicted_probabilities,
                                                             (true_labels.shape[0], out_size)), classes_to_plot=0)})
            prob_complete = -predicted_probabilities + 1
            reshaped_probas = np.reshape(predicted_probabilities, (predicted_probabilities.shape[0], 1))
            prob_complete = np.reshape(prob_complete, (prob_complete.shape[0], 1))
            probas = np.concatenate((prob_complete, reshaped_probas), axis=1)
        elif cfg['loss_type'] != 'corn':
            probas = np.reshape(predicted_probabilities, (true_labels.shape[0], out_size))
        th = list(cfg['distance_th_classification'])
        if th is None or th == 'None':
            th = [11.897181510925293, 16.312454223632812, 22.885560989379883]
        th.append(45)
        th.insert(0, 0)
        th = [(round(th_i, 1)) for th_i in th]
        class_names = [f"{th[j]}-{th[j + 1]}" for j in range(len(th) - 1)]
        if cfg['loss_type'] != 'corn':
            wandb_.log({f"{title}roc": wandb.plot.roc_curve(true_labels, probas, classes_to_plot=0, title=f"{title}ROC")})
            cm = wandb.plot.confusion_matrix(y_true=true_labels.tolist(), probs=probas,
                                             class_names=class_names, title=f"{title}Confusion Matrix (Validation)")
        else:
            preds = np.reshape(predicted_probabilities, (true_labels.shape[0], out_size - 1))
            preds = label_from_corn_logits(torch.from_numpy(preds))
            cm = wandb.plot.confusion_matrix(y_true=true_labels.tolist(), preds=preds.tolist(),
                                             class_names=class_names, title=f"{title}Confusion Matrix (Validation)")
        wandb_.log({f"{title}conf_mat": cm})
        wandb_.log({f"{title}recall": recall})
        wandb_.log({f"{title}precision": precision})
        wandb_.log({f"{title}auc": auc_roc})
        fig = general_utils.plot_roc(true_labels, probas, "ROC curve", cfg['num_classes'])
        wandb_.log({f"{title}roc_plot_multiclass": fig})
        # plot_cm(true_labels, probas)


def train(cfg=None, i=0, wandb_=None):
    if wandb_ is None:
        wandb_ = wandb.init(project="xl_gnn", entity="seanco", config=cfg)
    print(cfg, flush=True)
    if cfg['num_classes'] == 2:
        out_size = 1
    else:
        out_size = cfg['num_classes']
    model = get_model(cfg)
    model.float()
    if torch.cuda.device_count() > 1 and cfg['multi_gpu']:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch_geometric.nn.DataParallel(model)
    model.to(device)
    if 'fine_tune' in cfg and cfg['fine_tune']:
        fine_tune_model(model)
    _optimizer = construct_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(cfg, _optimizer)
    initial_epoch = 0
    if 'exist_model' in cfg and cfg['exist_model'] is not None:
        if cfg['exist_model'].split('_')[-1] == 'scaled':
            model = temperature_scaling.ModelWithTemperature(model)
            model.to(device)
        checkpoint = torch.load(f"/cs/labs/dina/seanco/xl_mlp_nn/models/{cfg['exist_model']}")
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        _optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']
        if cfg['data_type'] != 'Train' or cfg['fine_tune']:
            initial_epoch = 0
        loss = checkpoint['loss']
        if cfg['calibrate_model'] == 'start':
            return temperature_scaling_calibration(model, cfg, wandb_, initial_epoch, _optimizer, loss)
            # return temperature_scaling_calibration(model, cfg, wandb_, initial_epoch, _optimizer, 1)
    train_loss, val_loss, eval_accuracy, train_accuracy, per_class_acc = list(), list(), list(), list(), list()
    train_data, eval_data = get_dataset(cfg, wandb_)
    # graph_dataset.test_pair_graph_dataloader(train_data)
    criterion, is_regression = get_criterion(cfg, train_data)
    train_loader, val_loader = load_data(cfg, train_data, eval_data)
    # writer = SummaryWriter()
    epochs = cfg['epochs']
    # update_loss_weights(cfg)
    for epoch in range(initial_epoch, epochs):
        try:
            if cfg['data_type'] == 'Train':
                train_epoch(model, train_loader, criterion, _optimizer, epoch, train_loss, wandb_, scheduler,
                            train_accuracy, cfg['num_classes'], cfg['epochs'] - 1, cfg)
            # update_lr(optimizer, epoch, cfg)

            # Evaluate the model.
            per_class_acc, predicted_probabilities, true_labels, confusion_matrix = eval_epoch(model,
                                    val_loader, criterion, epoch, val_loss, wandb_,
                                    is_regression, cfg['num_classes'], eval_accuracy,
                                    cfg['epochs'] - 1, cfg)

        except Exception as e:
            print(e)
            raise e
        except SystemExit as e:
            print(e)
            raise e

    print("Finished Training")
    log_results(cfg, wandb_, confusion_matrix, predicted_probabilities, true_labels, i, out_size)
    if cfg['xl_type_feature']:
        stats_of_inter_xl_performance(model, eval_data, criterion, wandb_, cfg, i, out_size)
    if cfg['name'] != 'None':
        if cfg['calibrate_model'] == 'end':
            temperature_scaling_calibration(model, cfg, wandb_, epoch, _optimizer, train_loss[-1], val_loader,
                                            criterion[0])
        else:
            print("saving model")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': _optimizer.state_dict(), 'loss': train_loss[-1]},
                       f"models/{cfg['name']}")


def parameters_tune(args, cfg=None, sweeps=False):
    print("Start hyper parameter tune\n", flush=True)
    i = 0
    if cfg is None:
        cfg = load_config(args)
    if sweeps:
        return run_with_sweeps(cfg)
        # return train_with_sweeps()
    if not cfg['hparam_tune']:
        return train(cfg=cfg)
    lrs = [0.0005, 0.0001]
    # schedulers = ['steps', None]
    # batch_sizes = [32, 64]
    l1s = [256, 512]
    l2s = [256, 128, 64, 32]
    l3s = [128, 64, 32, 16, 8, 4]
    thresholds = [18, 20]
    hidden_layers = [3, 4]

    # for opt in optimizers:
    for lr in lrs:
        # for th in thresholds:
        for l1 in l1s:
            for l2 in l2s:
                for l3 in l3s:
                    layers = [l1, l2, l3]

                    cfg['base_lr'] = lr
                    cfg['layers'] = layers
                    print('train with params: opt, lr, drop, class_th: ', lr)
                    wandb_run = wandb.init(reinit=True, project="xl_gnn", entity="seanco", config=cfg)
                    train(None, args=args, cfg=cfg, i=i, wandb_=wandb_run)
                    i += 1


def parse_args():
    parser = TrainingParser()
    args = parser.parse_args()
    TrainingParser.validate_args(args)
    return args


class TrainingParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        super(TrainingParser, self).__init__(**kwargs)
        self.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the config file",
            default="/cs/labs/dina/seanco/xl_mlp_nn/configs/gnn_separate_44f_3a.yaml",
            # default="/cs/labs/dina/seanco/xl_mlp_nn/configs/eval_inter_44f_3a.yaml",
            # default="/cs/labs/dina/seanco/xl_mlp_nn/configs/debug_config.yaml",
            # default="/cs/labs/dina/seanco/xl_mlp_nn/configs/sweep_gat_44.yaml",
            type=str,
        )
        self.add_argument(
            "--lr",
            help="Base learning rate",
            type=float,
        )
        self.add_argument(
            "--epochs",
            help="Number of epochs",
            type=int
        )

    @staticmethod
    def validate_args(args):
        if args.cfg_file is None or not os.path.isfile(args.cfg_file):
            raise argparse.ArgumentTypeError(f"Invalid config file path: {args.cfg_file}")

    def parse_args(self, args=None, namespace=None):
        """ Parse the input arguments """
        args = super(TrainingParser, self).parse_args(args, namespace)
        return args


def train_with_sweeps():
    # cfg = load_config(None)
    # with wandb.init(config=cfg, project='xl_gnn', entity='seanco') as wandb_:
    with wandb.init() as wandb_:
        config = wandb_.config
        print(config)
        # cfg_file_path = open('/cs/labs/dina/seanco/xl_mlp_nn/configs/gnn_separate_39f_angles.yaml', 'r')
        # cfg = yaml.safe_load(cfg_file_path)
        # cfg = upd_cfg_by_sweeps(cfg, config)
        train(config, wandb_=wandb_)
    gc.collect()


def run_with_sweeps(config):
    wandb.login()
    # sweep_id = wandb.sweep(config, entity='seanco', project='xl_gnn')
    # wandb.agent(sweep_id, project='xl_gnn', entity='seanco', function=train_with_sweeps)
    wandb.agent('gfu40i73', project='xl_gnn', entity='seanco', function=train_with_sweeps)


if __name__ == "__main__":
    xl_objects = cross_link.get_upd_and_filter_processed_objects_pipeline('processed_xl_objects_intra_lys')
    graph_dataset.generate_graph_data(xl_objects, data_name='44f_3A_intra_lys_dataset', edge_dist_th=3)
    print("Start main\n", flush=True)
    seed_everything()
    args = parse_args()
    parameters_tune(args, None, False)
