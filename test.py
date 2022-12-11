import torch
import torchvision
from torchvision.models import yolo

# 定义YOLOv3模型
model = yolo.YOLOv3()

# 定义输入图像
image = torch.randn(3, 224, 224)

# 经过YOLOv3的neck部分，得到特征图
features = model.neck(image)
print(features)