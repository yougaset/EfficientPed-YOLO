import os
import torch
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from ultralytics import YOLO

# 获取项目根目录（`src` 的上一级目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 训练完成的模型路径（确保指向 `results/` 根目录）
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "results", "yolov8_citypersons_improved", "weights", "best.pt"))

# 量化后模型的保存路径（存放在 `results/models/`）
QUANTIZED_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "results", "models", "best_quantized.pt"))

# 确保 `results/models/` 目录存在
os.makedirs(os.path.dirname(QUANTIZED_MODEL_PATH), exist_ok=True)


# 量化感知训练模型
class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedModel, self).__init__()
        self.quant = QuantStub()  # 量化输入
        self.model = model  # 原 YOLO 模型
        self.dequant = DeQuantStub()  # 反量化

    def forward(self, x):
        x = self.quant(x)  # 量化输入
        x = self.model(x)  # 通过模型
        x = self.dequant(x)  # 反量化输出
        return x


# 量化流程
def quantize_model():
    """加载 YOLO 训练模型，并进行 QAT 量化"""

    # ✅ 确保 `best.pt` 存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 错误：找不到模型文件 {MODEL_PATH}，请先运行 train.py 进行训练！")

    print(f"✅ 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 包装为量化模型
    quantized_model = QuantizedModel(model.model)
    quantized_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

    # **修正 QAT 设置，确保 Conv2d & Linear 模块参与量化**
    for name, module in quantized_model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.qconfig = torch.quantization.default_qat_qconfig

    # 准备量化感知训练（QAT）
    prepare_qat(quantized_model, inplace=True)
    print("📌 量化感知训练已准备完成，开始转换...")

    # 转换为量化模型
    quantized_model = convert(quantized_model, inplace=True)
    print("✅ 量化转换完成！")

    # 保存量化模型
    torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)
    print(f"✅ 量化后的模型已保存至: {QUANTIZED_MODEL_PATH}")


if __name__ == "__main__":
    quantize_model()
