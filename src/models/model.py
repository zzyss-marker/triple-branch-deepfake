import torch
import torch.nn as nn
import timm

class DualBranchModel(nn.Module):
    def __init__(self, base_model_spatial, base_model_freq, base_model_highfreq, num_classes=2):
        super(DualBranchModel, self).__init__()
        
        # 空间域分支
        self.spatial_branch = base_model_spatial
        
        # 频域分支
        self.freq_branch = base_model_freq
        
        # 高频分支
        self.highfreq_branch = base_model_highfreq
        
        # 获取特征维度
        self.feature_dim = base_model_spatial.num_features
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x_spatial, x_freq, x_highfreq):
        # 提取空间特征
        spatial_features = self.spatial_branch(x_spatial)
        
        # 提取频域特征
        freq_features = self.freq_branch(x_freq)
        
        # 提取高频特征
        highfreq_features = self.highfreq_branch(x_highfreq)
        
        # 特征融合
        combined_features = torch.cat([
            spatial_features, 
            freq_features, 
            highfreq_features
        ], dim=1)
        
        # 最终分类
        output = self.fusion(combined_features)
        return output

    @classmethod
    def create_model(cls, num_classes=2, model_name='efficientnet_b5'):
        # 创建三个基础模型
        base_model_spatial = timm.create_model(model_name, pretrained=True)
        base_model_spatial.classifier = nn.Identity()
        
        base_model_freq = timm.create_model(model_name, pretrained=True)
        base_model_freq.classifier = nn.Identity()
        
        base_model_highfreq = timm.create_model(model_name, pretrained=True)
        base_model_highfreq.classifier = nn.Identity()
        
        return cls(base_model_spatial, base_model_freq, base_model_highfreq, num_classes)