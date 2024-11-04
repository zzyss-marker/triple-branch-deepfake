import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Union
from pathlib import Path

class FFDIDataset(Dataset):
    """
    Fake Face Detection and Identification Dataset
    用于假脸检测的数据集类
    """
    def __init__(self, 
                 img_paths: Union[List[str], List[Path]],
                 img_labels: Optional[List[int]] = None,
                 transform = None,
                 return_path: bool = False):
        """
        初始化数据集
        
        Args:
            img_paths: 图像路径列表
            img_labels: 标签列表（训练模式下需要）
            transform: 图像变换
            return_path: 是否返回图像路径
        """
        self.img_paths = [str(p) for p in img_paths]  # 确保路径是字符串格式
        self.img_labels = img_labels
        self.transform = transform
        self.return_path = return_path
        self.is_train = img_labels is not None

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                              Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
        """
        获取一个数据样本
        
        Args:
            index: 索引值
            
        Returns:
            训练模式: (image, frequency, high_frequency, label)
            测试模式: (image, frequency, high_frequency, image_path)
        """
        # 读取图像
        img_path = self.img_paths[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回一个随机的相同大小的图像作为替代
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # 应用变换
        if self.transform is not None:
            img = self.transform(img)

        # 转换为张量并添加批次维度
        img = img.unsqueeze(0)  # [1, C, H, W]

        # 获取频域特征
        freq = self._to_freq_domain(img)
        
        # 获取高频特征
        high_freq = self._high_pass_filter(img)

        # 移除批次维度
        img = img.squeeze(0)
        freq = freq.squeeze(0)
        high_freq = high_freq.squeeze(0)

        if self.is_train:
            label = torch.tensor(self.img_labels[index]).long()
            return img, freq, high_freq, label
        
        if self.return_path:
            return img, freq, high_freq, img_path
        return img, freq, high_freq

    @staticmethod
    def _to_freq_domain(img_tensor: torch.Tensor) -> torch.Tensor:
        """
        将图像转换到频域
        
        Args:
            img_tensor: 输入图像张量 [B, C, H, W]
            
        Returns:
            频域特征张量
        """
        img_np = img_tensor.numpy()
        freq_features = []
        
        for img in img_np:
            freq_channels = []
            for channel in img:
                # 应用FFT
                f = np.fft.fft2(channel)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = np.abs(fshift)
                
                # 对数变换
                magnitude_spectrum = np.log1p(magnitude_spectrum)
                
                # 归一化
                magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
                                   (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
                
                freq_channels.append(magnitude_spectrum)
            
            freq_features.append(freq_channels)
        
        return torch.from_numpy(np.array(freq_features)).float()

    @staticmethod
    def _high_pass_filter(img_tensor: torch.Tensor, cutoff: float = 0.1) -> torch.Tensor:
        """
        应用高通滤波器
        
        Args:
            img_tensor: 输入图像张量 [B, C, H, W]
            cutoff: 截止频率
            
        Returns:
            高频特征张量
        """
        img_np = img_tensor.numpy()
        N, C, H, W = img_np.shape
        
        # 创建高通滤波器掩码
        rows, cols = H, W
        crow, ccol = rows // 2, cols // 2
        
        mask = np.ones((H, W), dtype=np.float32)
        center_radius = int(min(crow, ccol) * cutoff)
        
        y, x = np.ogrid[:H, :W]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= center_radius ** 2
        mask[mask_area] = 0
        
        # 应用滤波器到每个通道
        filtered = np.zeros_like(img_np)
        for i in range(N):
            for j in range(C):
                f = np.fft.fft2(img_np[i, j])
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * mask
                f_filtered = np.fft.ifftshift(fshift_filtered)
                filtered[i, j] = np.abs(np.fft.ifft2(f_filtered))
        
        # 归一化
        filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-8)
        
        return torch.from_numpy(filtered).float()

    @classmethod
    def from_dataframe(cls, 
                      df: pd.DataFrame,
                      img_dir: str,
                      img_col: str = 'image',
                      label_col: Optional[str] = 'label',
                      transform = None) -> 'FFDIDataset':
        """
        从DataFrame创建数据集
        
        Args:
            df: 包含图像信息的DataFrame
            img_dir: 图像目录
            img_col: 图像列名
            label_col: 标签列名
            transform: 图像变换
            
        Returns:
            FFDIDataset实例
        """
        img_paths = [os.path.join(img_dir, img_name) for img_name in df[img_col]]
        labels = df[label_col].values if label_col in df.columns else None
        return cls(img_paths, labels, transform)

    @classmethod
    def from_directory(cls,
                      data_dir: str,
                      transform = None,
                      valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> 'FFDIDataset':
        """
        从目录创建数据集
        
        Args:
            data_dir: 数据目录
            transform: 图像变换
            valid_extensions: 有效的文件扩展名
            
        Returns:
            FFDIDataset实例
        """
        img_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_paths.append(os.path.join(root, file))
        return cls(img_paths, transform=transform)

def create_dataloaders(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      img_dir: str,
                      train_transform,
                      val_transform,
                      batch_size: int = 32,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        train_df: 训练数据DataFrame
        val_df: 验证数据DataFrame
        img_dir: 图像目录
        train_transform: 训练数据变换
        val_transform: 验证数据变换
        batch_size: 批次大小
        num_workers: 工作进程数
        
    Returns:
        训练数据加载器和验证数据加载器的元组
    """
    train_dataset = FFDIDataset.from_dataframe(
        train_df, img_dir, transform=train_transform
    )
    
    val_dataset = FFDIDataset.from_dataframe(
        val_df, img_dir, transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader