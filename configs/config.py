import torch

class TrainConfig:
    # Data
    train_data_path = "data/train/phase1/trainset_label.txt"
    val_data_path = "data/train/phase1/valset_label.txt"
    train_img_dir = "data/train/phase1/trainset/"
    val_img_dir = "data/train/phase1/valset/"
    
    # Training parameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 1e-4
    max_lr = 1e-3
    weight_decay = 1e-4
    early_stop_patience = 10
    label_smoothing = 0.1
    
    # Model parameters
    img_size = (256, 256)
    num_classes = 2
    model_name = 'efficientnet_b5'
    
    # Hardware
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    model_save_path = "./checkpoints/"
    log_dir = "./logs/"

class PredictConfig:
    img_folder = "/testdata"
    batch_size = 64
    img_size = (256, 256)
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = [
        'checkpoints/best_model.pt',
    ]
    weights = [1.0]