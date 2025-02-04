# EfficientPed-YOLO
### **🚀 YOLOv8 行人检测 - 使用教程**  
本项目基于 **YOLOv8 + Swin Transformer** 进行 **行人检测**，支持 **模型训练、评估、量化及推理**。  

---

## **📌 目录**
- [1. 环境安装](#1-环境安装)
- [2. 数据准备](#2-数据准备)
- [3. 训练模型](#3-训练模型)
- [4. 评估模型](#4-评估模型)
- [5. 量化模型](#5-量化模型)
- [6. 模型推理](#6-模型推理)

---

## **1️⃣ 环境安装**
### **1.1 创建 Python 虚拟环境**
推荐使用 **conda** 或 **venv** 进行环境管理：
```bash
# conda 创建虚拟环境
conda create -n yolo_env python=3.9 -y
conda activate yolo_env
```
或
```bash
# venv 创建虚拟环境
python -m venv yolo_env
source yolo_env/bin/activate  # Linux/macOS
yolo_env\Scripts\activate  # Windows
```

### **1.2 安装依赖**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **1.3 检查 YOLOv8 是否安装成功**
```bash
python -c "from ultralytics import YOLO; print(YOLO('yolov8n.pt'))"
```
如果成功加载 `yolov8n.pt` 说明安装成功。

---

## **2️⃣ 数据准备**
本项目默认支持 **COCO 格式** 和 **YOLO 格式** 的数据集，你需要：
1. **准备数据集**（如 `CityPersons`）
2. **将数据转换为 YOLO 格式**
3. **创建 `dataset/data.yaml` 配置文件**

### **2.1 数据集目录结构**
确保数据集格式如下：
```plaintext
dataset/
│── images/
│   │── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │── test/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│── labels/
│   │── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │── test/
│   │   ├── img1.txt
│   │   ├── img2.txt
│── data.yaml
```

### **2.2 创建 `dataset/data.yaml`**
编辑 `dataset/data.yaml`：
```yaml
train: ../dataset/images/train
val: ../dataset/images/test
nc: 1  # 类别数量（1 代表行人检测）
names: ["pedestrian"]
```

---

## **3️⃣ 训练模型**
运行以下命令进行模型训练：
```bash
python src/train.py
```
**训练完成后，模型将保存在**：
```plaintext
results/yolov8_citypersons_improved/weights/best.pt
```

### **可选参数**
- `epochs=50` 训练轮数（可调整）
- `batch=8` 训练批次大小
- `imgsz=1280` 输入图像尺寸
- `device="0"` 指定 GPU 训练

---

## **4️⃣ 评估模型**
**确保训练完成后**，运行：
```bash
python src/evaluate.py
```
**评估完成后，会输出模型指标，例如：**
```plaintext
✅ 加载模型: results/yolov8_citypersons_improved/weights/best.pt
🔍 测试集评估结果：
📌 mAP50: 0.642
📌 mAP50-95: 0.380
📌 Precision: 0.752
📌 Recall: 0.492
```

---

## **5️⃣ 量化模型**
量化后可以加快推理速度，降低模型大小：
```bash
python src/quantization.py
```
量化后的模型将保存在：
```plaintext
results/models/best_quantized.pt
```

---

## **6️⃣ 模型推理**
你可以使用训练好的模型进行行人检测：
```python
from ultralytics import YOLO

# 加载模型
model = YOLO("results/yolov8_citypersons_improved/weights/best.pt")

# 进行推理
results = model("test.jpg")

# 显示检测结果
results.show()
```

如果使用 **量化模型**：
```python
model = YOLO("results/models/best_quantized.pt")
results = model("test.jpg")
results.show()
```

---

## **📌 总结**
| **步骤** | **命令** | **输出** |
|---------|--------|---------|
| **1. 训练模型** | `python src/train.py` | `results/yolov8_citypersons_improved/weights/best.pt` |
| **2. 评估模型** | `python src/evaluate.py` | mAP, Precision, Recall |
| **3. 量化模型** | `python src/quantization.py` | `results/models/best_quantized.pt` |
| **4. 推理测试** | `model("test.jpg")` | 显示行人检测结果 |
