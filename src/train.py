import os
from ultralytics import YOLO
from multiprocessing import freeze_support
import torch.nn as nn
from transformers import SwinModel

# 获取当前文件的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 训练数据路径
DATA_CONFIG_PATH = os.path.join(BASE_DIR, "../dataset/data.yaml")

# 预训练模型路径
PRETRAINED_MODEL_PATH = os.path.join(BASE_DIR, "../models/yolov8n.pt")

# 训练结果存储目录
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

# 自动创建 results 目录（如果不存在）
os.makedirs(RESULTS_DIR, exist_ok=True)


# 自定义 Swin Transformer Backbone
class SwinBackbone(nn.Module):
    def __init__(self, model_path=os.path.join(BASE_DIR, "../models/swin-tiny-model")):
        super(SwinBackbone, self).__init__()
        self.swin = SwinModel.from_pretrained(model_path, use_safetensors=True)

    def forward(self, x):
        return self.swin(x).last_hidden_state


# 自定义检测头
class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DecoupledHead, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4, kernel_size=1),
        )

    def forward(self, x):
        return self.cls_head(x), self.reg_head(x)


# 训练主函数
def train():
    model = YOLO(PRETRAINED_MODEL_PATH)

    # 替换 Swin Transformer 作为 Backbone
    model.model.backbone = SwinBackbone()

    # 替换 Head
    model.model.head = DecoupledHead(in_channels=512, num_classes=80)

    results = model.train(
        data=DATA_CONFIG_PATH,
        epochs=50,
        imgsz=1280,
        batch=8,
        device="0",
        name=os.path.join(RESULTS_DIR, "yolov8_citypersons_improved"),  # 训练结果存储路径
        workers=8,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        save_period=10,
        save_json=True,
        save_hybrid=True,
        plots=True,
    )

    print(f"✅ 训练完成，结果已保存至 {RESULTS_DIR}")
    print(results)


if __name__ == "__main__":
    freeze_support()
    train()
