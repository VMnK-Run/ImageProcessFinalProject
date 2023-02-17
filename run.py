import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from model import VGG19
import logging
import colorlog
import argparse
from tqdm import tqdm, trange
from loadData import FERDataSet
import utils
import os
import csv

warnings.filterwarnings('ignore')

USE_GPU = True
LOG_TO_FILE = False
logger = logging.getLogger(__name__)

log_colors_config = {
    'DEBUG': 'white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(filename='./log/run.log', mode='a', encoding='utf8')

logger.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.INFO)

# 日志输出格式
file_formatter = logging.Formatter(
    fmt='[%(asctime)s] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S'
)
console_formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s[%(asctime)s] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S',
    log_colors=log_colors_config
)

console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler) if LOG_TO_FILE else 0

console_handler.close()
file_handler.close()

transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

train_acc_history = []
eval_acc_history = []


def train(args, model, train_dataset, device):
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    args.max_steps = args.epochs * len(train_dataset)  # 最大步长

    learning_rate_decay_start = 80
    learning_rate_decay_every = 5
    learning_rate_decay_rate = 0.9

    best_acc = 0.0

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    history_path = './log/' + args.model + '_' + args.dataset + '.csv'
    with open(history_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_acc", "eval_acc"])
    for epoch in range(args.epochs):
        if epoch > learning_rate_decay_start >= 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = args.learning_rate * decay_factor
            utils.set_lr(optimizer, current_lr)  # set the decayed rate
        train_loss = 0
        train_num = 0
        total_num = 0
        correct_num = 0
        train_acc = 0.0
        eval_acc = 0.0
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(bar):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            model.train()

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            loss, outputs = model(inputs, targets)
            loss.backward()

            utils.clip_gradient(optimizer=optimizer, grad_clip=0.1)
            optimizer.step()
            train_loss += loss.item()
            train_num += 1
            avg_loss = round(train_loss / train_num, 5)

            _, predicted = torch.max(outputs.data, 1)
            total_num += targets.size(0)
            correct_num += predicted.eq(targets.data).cpu().sum()
            train_acc = correct_num / total_num

            bar.set_description("epoch {} loss {} train_acc {}".format(epoch, avg_loss, train_acc))
        results = evaluate(args, model, device)
        if results['eval_acc'] > best_acc:
            best_acc = results['eval_acc']
            logger.info("  " + "*" * 20)
            logger.info("  Best accuracy:%s", round(best_acc, 4))
            logger.info("  " + "*" * 20)
            model_dir = './models/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_name = model_dir + args.model + '_' + args.dataset
            model_to_save = model.module if hasattr(model, 'module') else model
            state = {
                'model': model_to_save.state_dict() if USE_GPU else model_to_save,
                'acc': best_acc,
                'epoch': epoch
            }
            torch.save(state, model_name + '.t7')
        eval_acc = results['eval_acc']
        train_acc_history.append(train_acc)
        eval_acc_history.append(eval_acc)
        with open(history_path, 'a+', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_acc, eval_acc])


def evaluate(args, model, device):
    if args.dataset == "FER2013":
        eval_dataset = FERDataSet(mode="PublicTest", transform=transform_test)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    else:
        eval_dataset = None
        eval_dataloader = None

    # Eval!
    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        inputs = batch[0].to(device)
        label = batch[1].to(device)
        inputs, label = Variable(inputs), Variable(label)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')

    result = {
        "eval_acc": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, device):
    if args.dataset == "FER2013":
        test_dataset = FERDataSet(mode="PrivateTest", transform=transform_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    else:
        test_dataset = None
        test_dataloader = None

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    y_trues = []
    for batch in test_dataloader:
        inputs = batch[0].to(device)
        label = batch[1].to(device)
        inputs, label = Variable(inputs), Variable(label)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')

    result = {
        "eval_acc": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg", required=True,
                        help="model to train and evaluate")
    parser.add_argument("--dataset", type=str, default="FER2013", required=True,
                        help="dataset to train and evaluate")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on dev test")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to run eval on dev test")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--epochs", default=250, type=int,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", default=0.01, type=float,
                        help="learning rate")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

    logger.info("Training/evaluation parameters: %s", args)
    logger.info("device: %s", device)

    if args.model == "vgg":
        model = VGG19()
    else:
        model = None

    if args.do_train:
        if args.dataset == "FER2013":
            train_dataset = FERDataSet("Training", transform=transform_train)
            train(args, model, train_dataset, device)

    if args.do_eval:
        model_dir = './models/' + args.model + '_' + args.dataset + '_.bin'
        model.load_state_dict(torch.load(model_dir)['model'])
        model.to(device)
        result = evaluate(args, model, device)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        model_dir = './models/' + args.model + '_' + args.dataset + '.bin'
        model.load_state_dict(torch.load(model_dir)['model'])
        model.to(device)
        result = evaluate(args, model, device)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))


if __name__ == '__main__':
    main()
