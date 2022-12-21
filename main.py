import torch
import torchvision
from pytorch_lightning import Trainer
from datasets import train_test_loader

from models import EightTrigrams

# 2 classes; Only target class or background
num_classes = 4
batch_size = 1
# # parameters
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


def get_transform():
    custom_transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.432, 0.432, 0.374), (0.275, 0.273, 0.268))]
    return torchvision.transforms.Compose(custom_transforms)


train_data_dir = 'data/flower/train'
train_coco = 'data/flower/train/_annotations.coco.json'
train_loader = train_test_loader(train_data_dir, train_coco, get_transform())

train_data_dir = 'data/flower/test'
train_coco = 'data/flower/test/_annotations.coco.json'
test_loader = train_test_loader(train_data_dir, train_coco, get_transform())

if __name__ == "__main__":
    trainer = Trainer(accelerator='gpu', devices=1, limit_train_batches=100, max_epochs=1000)
    model = EightTrigrams(640, batch_size, num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)
    # move model to the right device

    model.to(device)
    model.hparams.lr = 0.001
    # trainer.tune(model)
    trainer.fit(model)
