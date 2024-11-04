import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob

from src.data.dataset import FFDIDataset
from src.data.transforms import get_val_transforms
from src.models.model import DualBranchModel
from configs.config import PredictConfig

def load_models(model_paths, device, num_classes=2):
    """
    加载多个模型
    """
    models = []
    for path in model_paths:
        model = DualBranchModel.create_model(num_classes=num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        models.append(model)
    return models

def predict_batch(models, batch, weights, device):
    """
    对一个batch进行预测，支持模型集成
    """
    img, freq, high_freq, _ = batch
    img = img.to(device)
    freq = freq.to(device)
    high_freq = high_freq.to(device)
    
    total_probs = None
    
    with torch.no_grad():
        for model, weight in zip(models, weights):
            outputs = model(img, freq, high_freq)
            probs = torch.softmax(outputs, dim=1)
            
            if total_probs is None:
                total_probs = weight * probs
            else:
                total_probs += weight * probs
    
    predictions = torch.argmax(total_probs, dim=1)
    return predictions.cpu().numpy(), total_probs.cpu().numpy()

def main():
    config = PredictConfig()
    
    # 获取测试图像路径
    test_images = glob(os.path.join(config.img_folder, "*.jpg")) + \
                 glob(os.path.join(config.img_folder, "*.png"))
    
    # 创建数据集和数据加载器
    test_dataset = FFDIDataset(
        img_paths=test_images,
        transform=get_val_transforms(config.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 加载模型
    models = load_models(config.models, config.device)
    
    # 归一化权重
    weights = np.array(config.weights)
    weights = weights / weights.sum()
    
    # 预测
    all_predictions = []
    all_probabilities = []
    all_image_paths = []
    
    print("Starting prediction...")
    for batch in tqdm(test_loader, desc="Predicting"):
        predictions, probabilities = predict_batch(models, batch, weights, config.device)
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        all_image_paths.extend(batch[-1])
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'image_path': all_image_paths,
        'prediction': all_predictions,
        'probability_class_0': [prob[0] for prob in all_probabilities],
        'probability_class_1': [prob[1] for prob in all_probabilities]
    })
    
    # 保存结果
    output_path = 'predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # 打印统计信息
    print("\nPrediction Statistics:")
    print(f"Total images processed: {len(results_df)}")
    print("\nPrediction distribution:")
    print(results_df['prediction'].value_counts())

if __name__ == '__main__':
    main()