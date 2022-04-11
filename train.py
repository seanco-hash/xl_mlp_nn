import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import yaml
from tqdm import tqdm
import mlp
from optimizer import construct_optimizer
import data_proccess
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn import metrics
import random
from torch_geometric.loader import DataLoader as PyG_DataLoader
import graph_dataset
from graph_dataset import TwoGraphsData
import gnn
import wandb


torch.multiprocessing.set_sharing_strategy('file_system')
matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "cpu":
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False


def ray_load_config():
    with open('/cs/labs/dina/seanco/xl_mlp_nn/configs/processed_xl.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_config(args):
   # Setup cfg.
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # cfg.set_new_allowed(True)  # Allow new attributes

        if args.lr is not None:
            cfg['SOLVER']['BASE_LR'] = args.lr

        if args.epochs is not None:
            cfg['TRAIN']['EPOCHS'] = args.epochs
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
    fig_name = out_dir + cfg['NAME'] + '_roc_' + str(i) + '.png'
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
    plot_roc_curve(fpr, tpr, auc, cfg, i)
    return auc, recall, precision


def calc_accuracy(output, labels, confusion_matrix=None, n_classes=2, distances=None):
    if n_classes > 2:
        y_pred_prob = torch.softmax(output, dim=1)
        _, y_pred_tags = torch.max(y_pred_prob, dim=1)
    else:
        y_pred_prob = torch.sigmoid(output)
        y_pred_tags = torch.round(y_pred_prob)
    correct_pred = (y_pred_tags == labels).float()
    # if distances is not None:
    #     distances = distances[~correct_pred.bool().flatten()]
    if confusion_matrix is not None:
        for t, p in zip(labels, y_pred_tags):
            confusion_matrix[t.long(), p.long()] += 1
    return correct_pred.sum() / len(labels), y_pred_prob, distances, correct_pred.bool().flatten()


def calculate_loss_multi_head(data, model, criterion, n_classes, loss_weights, cfg, losses):
    inputs = data.to(device)
    outputs = model(inputs)
    labels = [inputs.y_ca, inputs.y_cb, inputs.y_omega, inputs.y_theta, inputs.y_phi]
    cur_total_loss = 0

    # if n_classes <= 2:
    #     labels[0] = torch.reshape(labels[0], outputs[0].shape()).float()
    #     if cfg['MODEL']['CB_PRED']:
    #         labels[1] = torch.reshape(labels[1], outputs[1].shape).float()
    # else:
    #     labels[0] = labels[0].long()
    #     if cfg['MODEL']['CB_PRED']:
    #         labels[1] = labels[1].long()

    for i in range(2, len(criterion)):
        if criterion[i] is not None:
            labels[i] = torch.reshape(labels[i], outputs[i].shape).float()
    for k in range(len(criterion)):
        if criterion[k] is not None:
            tmp_loss = (loss_weights[k] * criterion[k](outputs[k], labels[k]))
            cur_total_loss += tmp_loss
            losses[k] += tmp_loss.to('cpu')
    return cur_total_loss, inputs, outputs, labels


def calculate_loss(is_gnn, data, model, criterion, n_classes, loss_weights, cfg, is_connected=True, losses=None):
    if is_connected is False:
        return calculate_loss_multi_head(data, model, criterion, n_classes, loss_weights, cfg, losses)
    if is_gnn:
        inputs = data.to(device)
        outputs = model(inputs)
        labels = inputs.y
    else:
        inputs, labels = data
        inputs = inputs.to(device).float()
        outputs = model(inputs)
        labels = labels.to(device)

    if n_classes == 2:
        labels = torch.reshape(labels, outputs.shape)
        labels = labels.float()
    else:
        labels = labels.long()

    cur_loss = criterion(outputs, labels)
    return cur_loss, inputs, outputs, labels


def record_losses(losses, wandb_, len_data, epoch):
    if losses is None:
        return
    losses /= len_data
    title_lst = ['Ca', 'Cb', 'Omega', 'Theta', 'Phi']
    for i in range(len(losses)):
        if losses[i] != 0:
            wandb_.log({'epoch': epoch, f"{title_lst[i]} loss / Validation": losses[i]})


def eval_epoch(model, val_loader, criterion, epoch, val_loss, wandb_, is_regression, n_classes, acc,
               max_epoch, cfg, is_gnn=True):
    is_connected = cfg['DATA']['IS_CONNECTED']
    is_cb = cfg['MODEL']['CB_PRED']
    loss_weights = None
    losses = None
    if is_connected is False:
        losses = torch.zeros(5)
        loss_weights = cfg['MODEL']['LOSS_HEAD_WEIGHTS']
    model.eval()
    with torch.no_grad():
        predicted_probabilities = list()
        true_labels = list()
        confusion_matrix = torch.zeros(n_classes, n_classes)
        confusion_matrix_cb = torch.zeros(n_classes, n_classes)
        total_loss = 0
        accuracy = 0
        accuracy_cb = 0
        data_loader = tqdm(val_loader, position=0, leave=True)
        for data in data_loader:
            cur_loss, inputs, outputs, labels = calculate_loss(is_gnn, data, model, criterion,
                                                               n_classes, loss_weights, cfg, is_connected, losses)
            total_loss += cur_loss.item()
            if is_connected is False:
                tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs[0].to('cpu'), labels[0].to('cpu'),
                                                                         confusion_matrix, n_classes)
                if is_cb:
                    tmp_acc_cb, y_prob_cb, _, _ = calc_accuracy(outputs[1].to('cpu'), labels[1].to('cpu'),
                                                                confusion_matrix_cb, n_classes)
                    accuracy_cb += tmp_acc_cb
            else:
                tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs.to('cpu'), labels.to('cpu'),
                                                                         confusion_matrix, n_classes)
            accuracy += tmp_acc

            if epoch == max_epoch:
                predicted_probabilities.append(np.asarray(y_prob.to('cpu')).flatten())
                true_labels.append(np.asarray(labels[0].to('cpu')).flatten())
            # Print statistics.
            data_loader.set_postfix({'loss': cur_loss.item(), "Epoch": epoch})

        total_loss = total_loss / len(val_loader)
        if not is_regression:
            accuracy = accuracy / len(val_loader)
            acc.append(accuracy)
            # writer.add_scalar('Accuracy / Validation', accuracy, epoch)
            wandb_.log({'epoch': epoch, 'Accuracy / Validation': accuracy})
            per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
            # writer.add_scalars('Per-Class-Accuracy / Validation',
            #                    {'0': per_class_acc[0], '1': per_class_acc[1],
            #                     '2': per_class_acc[2]}, epoch)
            wandb_.log({'epoch': epoch, "Per-Class-Accuracy / Validation":
                      {ii: ac for ii, ac in enumerate(per_class_acc)}})
            # writer.add_scalars('Per-Class-Accuracy / Validation',
            #                    {'0': per_class_acc[0], '1': per_class_acc[1]}, epoch)
            print(confusion_matrix)
            print('per class acc: ', per_class_acc)
            if epoch == max_epoch:
                predicted_probabilities = np.concatenate(predicted_probabilities)
                true_labels = np.concatenate(true_labels)
            if is_cb:
                accuracy_cb = accuracy_cb / len(val_loader)
                wandb_.log({'epoch': epoch, 'Accuracy_CB / Validation': accuracy_cb})
                per_class_acc_cb = confusion_matrix_cb.diag() / confusion_matrix_cb.sum(1)
                wandb_.log({'epoch': epoch, "Per-Class-Accuracy_CB / Validation":
                    {ii: ac for ii, ac in enumerate(per_class_acc)}})

        val_loss.append(total_loss)
        wandb_.log({'epoch': epoch, 'Loss / Validation': total_loss})
        record_losses(losses, wandb_, len(data_loader), epoch)
        # writer.add_scalar('Loss / Validation', total_loss, epoch)
        print("Finished Evaluation epoch:", epoch, " loss: ", total_loss)
        return per_class_acc, predicted_probabilities, true_labels, confusion_matrix


def train_epoch(model, train_loader, criterion, _optimizer, epoch, train_loss, wandb_, scheduler, acc,
                n_classes, max_epoch, cfg, is_gnn=True):
    is_connected = cfg['DATA']['IS_CONNECTED']
    is_cb = cfg['MODEL']['CB_PRED']
    losses = None
    loss_weights = None
    if is_connected is False:
        loss_weights = cfg['MODEL']['LOSS_HEAD_WEIGHTS']
        losses = torch.zeros(5)
    model.train()
    # wrong_pred_feat = None
    # start_time = time.time()
    running_loss = 0
    total_loss = 0
    accuracy = 0
    accuracy_cb = 0
    # dist_in_wrong_pred = None
    confusion_matrix = torch.zeros(n_classes, n_classes)
    confusion_matrix_cb = torch.zeros(n_classes, n_classes)
    data_loader = tqdm(train_loader, position=0, leave=True)
    model = model.float()
    for i, data in enumerate(data_loader, 0):
        _optimizer.zero_grad()
        cur_loss, inputs, outputs, labels = calculate_loss(is_gnn, data, model, criterion,
                                                           n_classes, loss_weights, cfg, is_connected, losses)
        cur_loss.backward()
        _optimizer.step()
        running_loss += cur_loss.item()
        total_loss += cur_loss.item()
        if is_connected is False:
            tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs[0].to('cpu'), labels[0].to('cpu'),
                                                                     confusion_matrix, n_classes)
            if is_cb:
                tmp_acc_cb, y_prob_cb, _, _ = calc_accuracy(outputs[1].to('cpu'), labels[1].to('cpu'),
                                                            None, n_classes)
                accuracy_cb += tmp_acc_cb
        else:
            tmp_acc, y_prob, wrong_dist, correct_idx = calc_accuracy(outputs.to('cpu'), labels.to('cpu'),
                                                             confusion_matrix, n_classes)
        # if dist_in_wrong_pred is None:
        #     dist_in_wrong_pred = wrong_dist
        # else:
        #     dist_in_wrong_pred = torch.cat((dist_in_wrong_pred, wrong_dist))
        # if epoch == max_epoch:
        #     if wrong_pred_feat is None:
        #         wrong_pred_feat = inputs.y.to('cpu')[~correct_idx]
        #     else:
        #         wrong_pred_feat = torch.cat((wrong_pred_feat, inputs.y.to('cpu')[~correct_idx]), dim=0)
        accuracy += tmp_acc

        data_loader.set_postfix({'loss': cur_loss.item()})
        if (i + 1) % 20 == 0:
            running_loss /= 20
            print(f"[Train] Epoch: {epoch} Batch: {i+1} Loss: {running_loss:.3f}")
            running_loss = 0.0

    if scheduler is not None:
        scheduler.step()
    total_loss /= len(train_loader)
    train_loss.append(total_loss)
    wandb_.log({'epoch': epoch, 'Loss / Train': total_loss})
    accuracy = accuracy / len(train_loader)
    acc.append(accuracy)
    wandb_.log({'epoch': epoch, 'Accuracy / Train': accuracy})
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    wandb_.log({'epoch': epoch, "Per-Class-Accuracy / Train": {ii: ac for ii, ac in enumerate(per_class_acc)}})
    if is_cb:
        accuracy_cb = accuracy_cb / len(train_loader)
        wandb_.log({'epoch': epoch, 'Accuracy_CB / Train': accuracy_cb})
    record_losses(losses, wandb_, len(data_loader), epoch)
    print(confusion_matrix)
    print('per class acc Train: ', per_class_acc)
    print("Finished epoch:", epoch, " loss: ", total_loss, " accuracy: ", accuracy)


def get_multi_head_criterion(cfg, weights):
    cb_crit, omega_crit, theta_crit, phi_crit = None, None, None, None
    calc_weights = []
    if weights is not None:
        for i in range(len(weights)):
            if cfg['MODEL']['NUM_CLASSES'] == 2:
                calc_weights.append(torch.as_tensor(weights[i][0] / weights[i][1], dtype=torch.float))
            else:
                total_sum = np.sum(weights[i])
                calc_weights.append(torch.from_numpy((total_sum - weights[i]) / weights[i]).float().to(
                    device))
    if cfg['MODEL']['NUM_CLASSES'] > 2:
        ca_crit = torch.nn.CrossEntropyLoss(reduction='mean', weight=calc_weights[0])
        if cfg['MODEL']['CB_PRED']:
            cb_crit = torch.nn.CrossEntropyLoss(reduction='mean', weight=calc_weights[1])
    else:
        ca_crit = torch.nn.BCEWithLogitsLoss(pos_weight=calc_weights[0])
        if cfg['MODEL']['CB_PRED']:
            cb_crit = torch.nn.BCEWithLogitsLoss(pos_weight=calc_weights[1])
    if cfg['MODEL']['OMEGA_PRED']:
        omega_crit = torch.nn.MSELoss()
    if cfg['MODEL']['THETA_PRED']:
        theta_crit = torch.nn.MSELoss()
    if cfg['MODEL']['PHI_PRED']:
        phi_crit = torch.nn.MSELoss()
    return [ca_crit, cb_crit, omega_crit, theta_crit, phi_crit], False


def get_criterion(cfg, data=None):
    if data is not None and cfg['LOSS']['WEIGHT'] is True:
        if cfg['MODEL']['TYPE'] == 'gnn':
            weights = data.dataset.get_label_count()
            if not cfg['DATA']['IS_CONNECTED']:
                return get_multi_head_criterion(cfg, weights)
        else:
            weights = data.get_label_count()
        if cfg['MODEL']['NUM_CLASSES'] == 2:
            weights = torch.as_tensor(weights[0] / weights[1], dtype=torch.float)
        else:
            total_sum = np.sum(weights)
            weights = torch.from_numpy((total_sum - weights) / weights).float().to(device)
    else:
        weights = None
    if cfg['LOSS']['TYPE'] == 'mse':
        is_regression = True
        criterion = torch.nn.MSELoss()
    else:
        is_regression = False
        if cfg['MODEL']['NUM_CLASSES'] > 2:
            criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weights)
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    return criterion, is_regression


def get_scheduler(cfg, _optimizer):
    if cfg['SOLVER']['SCHEDULER'] == 'steps':
        steps = cfg['SOLVER']['STEPS']
        gama = cfg['SOLVER']['GAMMA']
        scheduler = MultiStepLR(_optimizer, steps, gama)
    else:
        scheduler = None
    return scheduler


def add_hparams_to_tb(writer, cfg, loss, total_accuracy, per_class_acc, auc, recall, precision):
    writer.add_hparams({'optimizer': cfg['SOLVER']['OPTIMIZING_METHOD'],
                        'lr': cfg['SOLVER']['BASE_LR'],
                        'weight_decay': cfg['SOLVER']['WEIGHT_DECAY'],
                        'scheduler': cfg['SOLVER']['SCHEDULER'],
                        # 'class_th': cfg['MODEL']['DISTANCE_TH_CLASSIFICATION'],
                        'dropout': cfg['MODEL']['DROPOUT'],
                        'batch_size': cfg['TRAIN']['BATCH_SIZE'],
                        'n_hidden_layers': cfg['MODEL']['HIDDEN_LAYERS'],
                        'class_weight': cfg['LOSS']['WEIGHT']},
                       {'hparam/loss': loss,
                        'hparam/accuracy': total_accuracy,
                        'hparam/acc_class_0': per_class_acc[0],
                        'hparam/acc_class_1': per_class_acc[1],
                        'hparam/AUC': auc,
                        'hparam/recall': recall,
                        'hparam/precision': precision})


def load_data(cfg, train_data, eval_data):
    # num_workers = 1
    if cfg['MODEL']['DEBUG']:
        num_workers = 1
    else:
        num_workers = len(os.sched_getaffinity(0)) - 1
    # num_workers = 1
    if cfg['MODEL']['TYPE'] == 'gnn':
        if cfg['DATA']['IS_CONNECTED']:
            train_loader = PyG_DataLoader(train_data, cfg['TRAIN']['BATCH_SIZE'], shuffle=True,
                                          num_workers=num_workers)
            val_loader = PyG_DataLoader(eval_data, cfg['TRAIN']['BATCH_SIZE'], shuffle=True,
                                        num_workers=num_workers)
        else:
            train_loader = PyG_DataLoader(train_data, cfg['TRAIN']['BATCH_SIZE'], shuffle=True,
                                          num_workers=num_workers,follow_batch=['x_a', 'x_b'])
            val_loader = PyG_DataLoader(eval_data, cfg['TRAIN']['BATCH_SIZE'], shuffle=True,
                                        num_workers=num_workers, follow_batch=['x_a', 'x_b'])
    else:
        train_loader = DataLoader(train_data, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True,
                                  num_workers=num_workers)
        val_loader = DataLoader(eval_data, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True,
                                num_workers=num_workers)
    return train_loader, val_loader


def get_dataset(cfg):
    if cfg['MODEL']['TYPE'] == 'gnn':
        if cfg['DATA']['IS_CONNECTED']:
            data = graph_dataset.XlGraphDataset(cfg)
        else:
            data = graph_dataset.XlPairGraphDataset(cfg)
        generator = torch.Generator().manual_seed(3407)
        if cfg['MODEL']['DEBUG']:
            eval_len, test_len = int(len(data) * 0.01), int(len(data) * 0.98)
        else:
            eval_len, test_len = int(len(data) * 0.1), int(len(data) * 0.1)
        train_len = int(len(data) - (eval_len + test_len))
        train_data, eval_data, test_data = random_split(data, [train_len, eval_len, test_len], generator)
    elif cfg['DATA']['DATASET'] == "linker_as_label":
        train_data = data_proccess.LinkerAsLabelDS(cfg, is_train=True)
        eval_data = data_proccess.LinkerAsLabelDS(cfg, is_train=False)
    else:
        train_data = data_proccess.FeatDataset(cfg, is_train=True)
        eval_data = data_proccess.FeatDataset(cfg, is_train=False)
    return train_data, eval_data


def get_model(cfg, config=None):
    if cfg['MODEL']['TYPE'] == 'gnn':
        return gnn.create_model(cfg)
    return mlp.create_model(cfg, config)


def train(cfg=None, i=0, checkpoint_dir=None, wandb_=None):
    if wandb_ is None:
        wandb_ = wandb.init(project="xl_gnn", entity="seanco", config=cfg)
    if cfg['MODEL']['NUM_CLASSES'] == 2:
        cfg['MODEL']['OUTPUT_SIZE'] = 1
    else:
        cfg['MODEL']['OUTPUT_SIZE'] = cfg['MODEL']['NUM_CLASSES']
    model = get_model(cfg)
    model.to(device)
    _optimizer = construct_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(cfg, _optimizer)
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        _optimizer.load_state_dict(optimizer_state)
    train_loss, val_loss, eval_accuracy, train_accuracy, per_class_acc = list(), list(), list(), list(), list()
    train_data, eval_data = get_dataset(cfg)
    # graph_dataset.test_pair_graph_dataloader(train_data)
    criterion, is_regression = get_criterion(cfg, train_data)
    train_loader, val_loader = load_data(cfg, train_data, eval_data)
    # writer = SummaryWriter()
    epochs = cfg['TRAIN']['EPOCHS']
    for epoch in range(epochs):
        try:
            train_epoch(model, train_loader, criterion, _optimizer, epoch, train_loss, wandb_, scheduler,
                        train_accuracy, cfg['MODEL']['NUM_CLASSES'], cfg['TRAIN']['EPOCHS'] - 1, cfg)
            # update_lr(optimizer, epoch, cfg)

            # Evaluate the model.
            per_class_acc, predicted_probabilities, true_labels, confusion_matrix = eval_epoch(model,
                                    val_loader, criterion, epoch, val_loss, wandb_,
                                    is_regression, cfg['MODEL']['NUM_CLASSES'], eval_accuracy,
                                    cfg['TRAIN']['EPOCHS'] - 1, cfg)

        except Exception as e:
            print(e)
            raise e
        except SystemExit as e:
            print(e)
            raise e

    print("Finished Training")
    auc_roc, recall, precision = calc_precision_recall_roc(confusion_matrix, cfg['MODEL']['NUM_CLASSES'],
                                                           predicted_probabilities, true_labels, cfg, i)
    wandb_.log({"pr": wandb.plot.pr_curve(true_labels,
                                np.reshape(predicted_probabilities, (true_labels.shape[0],
                                                                     cfg['MODEL']['OUTPUT_SIZE'])),
                                          classes_to_plot=0)})
    if cfg['MODEL']['NUM_CLASSES'] == 2:
        prob_complete = -predicted_probabilities + 1
        reshaped_probas = np.reshape(predicted_probabilities, (predicted_probabilities.shape[0], 1))
        prob_complete = np.reshape(prob_complete, (prob_complete.shape[0], 1))
        probas = np.concatenate((reshaped_probas, prob_complete), axis=1)
    else:
        probas = np.reshape(predicted_probabilities, (true_labels.shape[0],
                                                                     cfg['MODEL']['OUTPUT_SIZE']))
    wandb_.log({"roc": wandb.plot.roc_curve(true_labels,
                                            probas,
                                            classes_to_plot=0)})
    cm = wandb.plot.confusion_matrix(y_true=true_labels.tolist(), probs=probas)
    wandb_.log({"conf_mat": cm})
    wandb_.log({'recall': recall, 'precision': precision, 'auc': auc_roc})
    # add_hparams_to_tb(wandb_, cfg, train_loss[-1], eval_accuracy[-1], per_class_acc, auc_roc, recall, precision)
    # plot_res(train_loss, val_loss)


def parameters_tune(args, cfg=None, sweeps=False):
    print("Start hyper parameter tune\n", flush=True)
    i = 0
    if cfg is None:
        cfg = load_config(args)
    if sweeps:
        return run_with_sweeps(cfg)
    if not cfg['TRAIN']['HPARAM_TUNE']:
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
                    # for opt in optimizers:
                    # cfg['SOLVER']['OPTIMIZING_METHOD'] = opt
                    cfg['SOLVER']['BASE_LR'] = lr
                    cfg['MODEL']['LAYERS'] = layers
                    # cfg['SOLVER']['SCHEDULER'] = sc
                    # cfg['MODEL']['HIDDEN_LAYERS'] = hl
                    # cfg['MODEL']['DISTANCE_TH_CLASSIFICATION'] = th
                    # cfg['SOLVER']['OPTIMIZING_METHOD'] = opt
                    # cfg['SOLVER']['WEIGHT_DECAY'] = wd
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
            default="/cs/labs/dina/seanco/xl_mlp_nn/configs/gnn_separate_39f_angles.yaml",
            # default="/cs/labs/dina/seanco/xl_mlp_nn/configs/processed_xl.yaml",
            # default="/cs/labs/dina/seanco/xl_mlp_nn/configs/linker_as_label.yaml",
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


def upd_cfg_by_sweeps(cfg, sweeps_cfg):
    cfg['MODEL']['NUM_CLASSES'] = sweeps_cfg['num_classes']
    cfg['MODEL']['DROPOUT'] = sweeps_cfg['dropout']
    cfg['MODEL']['DISTANCE_TH_CLASSIFICATION'] = sweeps_cfg['distance_th_classification']
    cfg['MODEL']['HIDDEN_LAYERS'] = sweeps_cfg['hidden_layers']
    cfg['MODEL']['LAYERS'] = sweeps_cfg['layers']
    cfg['MODEL']['SAMPLE'] = sweeps_cfg['sample']
    cfg['MODEL']['POOL_RATIO'] = sweeps_cfg['pool_ratio']
    cfg['MODEL']['POOL_POLICY'] = sweeps_cfg['pool_policy']
    cfg['MODEL']['CB_PRED'] = sweeps_cfg['cb_pred']
    cfg['MODEL']['OMEGA_PRED'] = sweeps_cfg['omega_pred']
    cfg['MODEL']['THETA_PRED'] = sweeps_cfg['theta_pred']
    cfg['MODEL']['LOSS_HEAD_WEIGHTS'] = sweeps_cfg['loss_head_weights']
    cfg['SOLVER']['BASE_LR'] = sweeps_cfg['base_lr']
    cfg['SOLVER']['MOMENTUM'] = sweeps_cfg['momentum']
    cfg['SOLVER']['WEIGHT_DECAY'] = sweeps_cfg['weight_decay']
    cfg['SOLVER']['OPTIMIZING_METHOD'] = sweeps_cfg['optimizing_method']
    cfg['SOLVER']['SCHEDULER'] = sweeps_cfg['scheduler']
    cfg['SOLVER']['STEPS'] = sweeps_cfg['steps']
    cfg['SOLVER']['GAMMA'] = sweeps_cfg['gamma']
    cfg['TRAIN']['BATCH_SIZE'] = sweeps_cfg['batch_size']
    cfg['TRAIN']['EPOCHS'] = sweeps_cfg['epochs']
    cfg['LOSS']['WEIGHT'] = sweeps_cfg['weight']
    return cfg


def train_with_sweeps():
    wandb_ = wandb.init()
    config = wandb_.config
    print(config)
    cfg_file_path = open('/cs/labs/dina/seanco/xl_mlp_nn/configs/gnn_separate_39f_angles.yaml', 'r')
    cfg = yaml.safe_load(cfg_file_path)
    cfg = upd_cfg_by_sweeps(cfg, config)
    train(cfg, wandb_=wandb_)


def run_with_sweeps(config):
    wandb.login()
    sweep_id = wandb.sweep(config)
    wandb.agent(sweep_id, function=train_with_sweeps, count=1)


if __name__ == "__main__":
    print("Start main\n", flush=True)
    seed_everything()
    args = parse_args()
    parameters_tune(args, None, False)
