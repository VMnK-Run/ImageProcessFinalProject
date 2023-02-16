import torch
import torch.nn as nn
from torchvision import transforms
from model import VGG19
import logging
import colorlog
import argparse
from tqdm import tqdm, trange
from loadData import FERDataSet

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


def train(args, model, device):
    pass


def evaluate(args, model, device):
    pass


def test(args, model, device):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg", required=True,
                        help="model to train and evaluate")

    args = parser.parse_args()
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)


if __name__ == '__main__':
    main()
