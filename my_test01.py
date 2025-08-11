from ultralytics import YOLO

def main():
    """
    主函数，用于执行 YOLO 模型的训练和评估。
    """
    # 加载一个预训练的 YOLOv11n 模型
    # 如果 yolo11n.pt 不存在，它会自动从 Ultralytics 的服务器下载
    model = YOLO("yolo11n.pt")

    # 在 COCO8 数据集上训练模型 100 个周期
    # coco8.yaml 会在首次使用时自动下载
    train_results = model.train(
        data="coco8.yaml",  # 数据集配置文件路径
        epochs=100,         # 训练周期数
        imgsz=640,          # 训练图像尺寸
        device="0",         # 运行设备（'cpu', '0', 或 [0,1,2,3]）
        # workers=0         # 如果问题仍然存在，可以尝试将 workers 设置为 0
    )

    # 评估模型在验证集上的性能
    print("开始评估模型性能...")
    metrics = model.val()
    print("模型评估指标:", metrics)

if __name__ == '__main__':
    # 这行代码是关键，它确保了只有在直接运行此脚本时，
    # 才会执行 main() 函数中的内容。
    # 这可以防止在 Windows 上进行多进程数据加载时出现递归错误。
    main()