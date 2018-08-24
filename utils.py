import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import json
from transforms import ColorJitter, Lighting
import torchvision.transforms as transforms

def get_model_params(network_config):
    model_params = {}
    model_params["p"] = network_config["p"]
    model_params["num_classes"] = network_config["num_classes"]
    return model_params


def create_optimizer(optimizer_config, params):

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
            step_size=optimizer_config["schedule"]["params"]["step_size"],
            gamma=optimizer_config["schedule"]["params"]["gamma"])
    else:
        raise KeyError("unrecognized schedule type {}".format(optimizer_config["schedule"]["type"]))

    return optimizer, scheduler



def create_transforms(input_config):

    train_transforms = []

    if input_config["scale_and_random_crop_train"]:
        train_transforms += [transforms.Resize(input_config["smallest_side_scale_train"])]
        train_transforms += [transforms.RandomCrop(input_config["random_crop_train"])]
    else:
        train_transforms += [transforms.RandomResizedCrop(
                                 size=input_config["randomresized_crop_train"], 
                                 scale=tuple(input_config["randomresized_scale_train"]), 
                                 ratio=tuple(input_config["randomresized_ratio_train"]))]    
    
    if input_config["horizontal_flip_train"]: train_transforms += [transforms.RandomHorizontalFlip()]
    
    train_transforms += [transforms.ToTensor()]
    
    if input_config["color_jitter_train"]: train_transforms += [ColorJitter()]
    if input_config["lighting_train"]: train_transforms += [Lighting()]
    
    train_transforms += [transforms.Normalize(mean=input_config["mean"], std=input_config["std"])]


    val_transforms = []

    if input_config["scale_and_center_crop_val"]:
        val_transforms += [transforms.Resize(size=(input_config["smallest_size_scale_val"]))]
        val_transforms += [transforms.CenterCrop(input_config["center_crop_val"])]
    else: 
        val_transforms += [transforms.Resize(size=(input_config["squared_crop_val"],input_config["squared_crop_val"]))]

    val_transforms += [transforms.ToTensor()]
    val_transforms += [transforms.Normalize(mean=input_config["mean"], std=input_config["std"])]

    return train_transforms, val_transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_config(config_file):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    return config
