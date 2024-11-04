import os
import random
import numpy as np
import torch
import logging
from datetime import datetime
import cv2
from typing import Union, Tuple

def seed_everything(seed: int = 42) -> None:
    """
    设置整个训练过程的随机种子
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir: str) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_dir (str): 日志保存目录
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime('%Y%m%d_%H%M%S.log')
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('DeepFakeDetection')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def to_freq_domain(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    将图像转换到频域，并提取幅度谱
    
    Args:
        img_tensor (torch.Tensor): 输入图像张量 [B, C, H, W]
        
    Returns:
        torch.Tensor: 频域特征张量
    """
    img_np = img_tensor.cpu().numpy()
    freq_features = []
    
    for img in img_np:
        freq_channels = []
        for channel in img:
            # 应用FFT
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            
            # 对数变换以增强视觉效果
            magnitude_spectrum = np.log1p(magnitude_spectrum)
            
            # 归一化
            magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
                               (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
            
            freq_channels.append(magnitude_spectrum)
        
        freq_features.append(freq_channels)
    
    return torch.from_numpy(np.array(freq_features)).float()

def high_pass_filter(img_tensor: torch.Tensor, 
                    cutoff: float = 0.1,
                    order: int = 2) -> torch.Tensor:
    """
    使用Butterworth高通滤波器提取高频噪声特征
    
    Args:
        img_tensor (torch.Tensor): 输入图像张量 [B, C, H, W]
        cutoff (float): 截止频率 (0到1之间)
        order (int): 滤波器阶数
        
    Returns:
        torch.Tensor: 高频特征张量
    """
    img_np = img_tensor.cpu().numpy()
    filtered_images = []
    
    for img in img_np:
        channels = []
        for channel in img:
            # 创建频率网格
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            u = np.arange(rows)
            v = np.arange(cols)
            u, v = np.meshgrid(u, v, indexing='ij')
            
            # 计算到中心的距离
            d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2) / crow
            
            # Butterworth高通滤波器
            h = 1 / (1 + (cutoff / (d + 1e-8)) ** (2 * order))
            
            # 应用滤波器
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            filtered = fshift * h
            filtered_channel = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
            
            # 归一化
            filtered_channel = (filtered_channel - filtered_channel.min()) / \
                             (filtered_channel.max() - filtered_channel.min() + 1e-8)
            
            channels.append(filtered_channel)
        
        filtered_images.append(channels)
    
    return torch.from_numpy(np.array(filtered_images)).float()

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self, name: str, fmt: str = ':f'):
        """
        Args:
            name (str): 度量名称
            fmt (str): 显示格式
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计值
        
        Args:
            val (float): 当前值
            n (int): 当前批次大小
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """字符串表示"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state: dict, 
                   is_best: bool, 
                   checkpoint_dir: str, 
                   filename: str = 'checkpoint.pth.tar') -> None:
    """
    保存检查点
    
    Args:
        state (dict): 要保存的状态字典
        is_best (bool): 是否是最佳模型
        checkpoint_dir (str): 保存目录
        filename (str): 文件名
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)

def load_checkpoint(checkpoint_path: str, 
                   model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None) -> Tuple[int, float]:
    """
    加载检查点
    
    Args:
        checkpoint_path (str): 检查点路径
        model (torch.nn.Module): 模型
        optimizer (torch.optim.Optimizer, optional): 优化器
        
    Returns:
        Tuple[int, float]: (开始轮次, 最佳验证损失)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    return checkpoint['epoch'], checkpoint['best_val_loss']

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    获取当前学习率
    
    Args:
        optimizer (torch.optim.Optimizer): 优化器
        
    Returns:
        float: 当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']