import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def compute_metrics(targets, predictions, probabilities=None):
    """
    计算各种评估指标
    
    Args:
        targets: 真实标签
        predictions: 预测标签
        probabilities: 预测概率（可选）
    
    Returns:
        dict: 包含各种指标的字典
    """
    metrics = {}
    
    # 计算分类报告
    report = classification_report(targets, predictions, output_dict=True)
    metrics['classification_report'] = report
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions)
    metrics['confusion_matrix'] = cm
    
    # 如果提供了概率，计算AUC
    if probabilities is not None:
        auc = roc_auc_score(targets, probabilities[:, 1])
        metrics['auc'] = auc
    
    return metrics

def print_metrics(metrics):
    """打印评估指标"""
    print("\nClassification Report:")
    for label in ['0', '1']:
        print(f"\nClass {label}:")
        print(f"Precision: {metrics['classification_report'][label]['precision']:.4f}")
        print(f"Recall: {metrics['classification_report'][label]['recall']:.4f}")
        print(f"F1-score: {metrics['classification_report'][label]['f1-score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    if 'auc' in metrics:
        print(f"\nAUC: {metrics['auc']:.4f}")