import os
from ultralytics import YOLO

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 训练完成的模型路径
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "results", "yolov8_citypersons_improved", "weights", "best.pt"))

# 训练数据路径
DATA_CONFIG_PATH = os.path.normpath(os.path.join(BASE_DIR, "dataset/data.yaml"))

# 确保 `results/` 目录存在
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def evaluate():
    """评估 YOLO 训练模型"""

    # ✅ 确保 `best.pt` 存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 错误：找不到模型文件 {MODEL_PATH}，请先运行 train.py 进行训练！")

    print(f"✅ 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 运行模型评估
    test_results = model.val(
        data=DATA_CONFIG_PATH,
        split="test",
        imgsz=1280,
        batch=8,
        device="0",
        name="yolov8_citypersons_test",
        iou=0.5,
        conf=0.1,
        augment=False,
    )

    # 计算评估指标
    precision = test_results.box.mp  # Mean Precision
    recall = test_results.box.mr  # Mean Recall
    map50 = test_results.box.map50  # mAP50
    map50_95 = test_results.box.map  # mAP50-95

    # 输出评估结果
    print("\n🔍 **测试集评估结果：**")
    print(f"📌 **mAP50:** {map50:.4f}")
    print(f"📌 **mAP50-95:** {map50_95:.4f}")
    print(f"📌 **Precision:** {precision:.4f}")
    print(f"📌 **Recall:** {recall:.4f}")


if __name__ == "__main__":
    evaluate()
