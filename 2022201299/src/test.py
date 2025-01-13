import torch
from torchvision import transforms
from PIL import Image
import os
from train import get_model, transform, GAME_CATEGORIES
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

def load_model(model_path):
    """
    加载训练好的模型
    """
    model = get_model()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_image(model, image_path, device, show_plot=True):
    """
    预测单张图片，并可选择显示结果
    """
    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 获取所有类别的概率
        probs_dict = {}
        idx_to_class = {v: k for k, v in GAME_CATEGORIES.items()}
        for idx, prob in enumerate(probabilities[0]):
            class_name = idx_to_class[idx]
            prob_value = prob.item()
            probs_dict[class_name] = prob_value
            print(f"{class_name}: {prob_value:.2%}")
            
        probability, predicted = torch.max(probabilities, 1)
    
    predicted_class = idx_to_class[predicted.item()]
    
    # 如果需要显示图片和预测结果
    if show_plot:
        plt.figure(figsize=(12, 6))
        
        # 显示图片
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'预测结果: {predicted_class}\n置信度: {probability.item():.2%}')
        plt.axis('off')
        
        # 显示概率条形图
        plt.subplot(1, 2, 2)
        classes = list(probs_dict.keys())
        probs = list(probs_dict.values())
        
        # 创建水平条形图
        bars = plt.barh(classes, probs)
        plt.xlabel('概率')
        plt.title('各类别预测概率')
        
        # 在条形上添加概率值
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2%}', 
                    ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    return predicted_class, probability.item()



if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model_path = "E:/temp_classification/游戏/checkpoints/20241222_095745/best_model.pth"
    model = load_model(model_path).to(device)
    
    # 打印模型使用的类别映射
    print("\n模型类别映射:")
    for class_name, idx in GAME_CATEGORIES.items():
        print(f"{idx}: {class_name}")
    
    # 单张图片预测示例
    image_path = "data/RTS游戏/t01afef2814c7dc2cb4.jpg"
    if os.path.exists(image_path):
        print("\n单张图片预测:")
        predicted_class, confidence = predict_image(model, image_path, device, show_plot=True)
        print(f"Image: {image_path}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
    
