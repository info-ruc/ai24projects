import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # ResNet要求输入224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet预训练模型的标准化参数
])

# 在transform定义之后添加类别映射
GAME_CATEGORIES = {
    'MOBA游戏': 0,
    'RPG游戏': 1,
    'RTS游戏': 2,
    '体育竞技游戏': 3,
    '卡牌游戏': 4,
    '射击游戏': 5,
    '音乐游戏': 6
}

def get_model(num_classes=7):  # 默认7个类别
    """
    获取预训练的ResNet50模型并修改最后的全连接层
    """
    # 加载预训练的ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 修改最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

# 加载数据集
def load_data(data_path, batch_size=32, val_split=0.2):
    """
    加载数据集并划分训练集和验证集
    
    Args:
        data_path: 数据集路径
        batch_size: 批次大小
        val_split: 验证集比例
    """
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    # 验证类别映射
    print("\n数据集类别映射:")
    for idx, class_name in enumerate(dataset.classes):
        print(f"{idx}: {class_name} -> GAME_CATEGORIES[{class_name}] = {GAME_CATEGORIES[class_name]}")
        # 确保映射一致
        assert GAME_CATEGORIES[class_name] == idx, f"类别映射不一致: {class_name} 在数据集中为 {idx}，在GAME_CATEGORIES中为 {GAME_CATEGORIES[class_name]}"
    
    # 获取类别数量
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")
    print("Classes:", dataset.classes)
    
    # 获取数据集大小
    n_samples = len(dataset)
    print(f"Total samples: {n_samples}")
    
    # 生成索引并划分训练集和验证集
    indices = list(range(n_samples))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_split,
        random_state=42,
        stratify=[dataset.targets[i] for i in indices]
    )
    
    # 创建训练集和验证集的Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, num_classes

# 添加评估函数
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return val_loss / len(val_loader), accuracy

# 修改训练函数
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_dir='checkpoints'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                train_loss = running_loss/len(pbar)
                train_acc = 100.*correct/total
                pbar.set_postfix({
                    'loss': f'{train_loss:.4f}',
                    'acc': f'{train_acc:.2f}%'
                })
        
        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 只保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
    
    # 保存训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return history

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载并划分数据集
    train_loader, val_loader, num_classes = load_data('data', batch_size=32, val_split=0.2)
    
    # 初始化模型
    model = get_model(num_classes).to(device)
    
    # 只训练最后的全连接层
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 创建保存目录（使用时间戳）
    save_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # 开始训练
    history = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, save_dir=save_dir) 