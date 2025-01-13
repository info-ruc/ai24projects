import torch
import torch.nn
import torch.optim
try:
    from .interface import DataAugmentorInterface
    from ..exceptions import ModelNotTrainedError
except:
    class DataAugmentorInterface(object):
        def __init__(self):
            pass
    
    ModelNotTrainedError = RuntimeError

class Generator(torch.nn.Module):
    def __init__(self, 
                 latent_dim, 
                 hidden_layers, 
                 activate_function, 
                 data_dim):
        super(Generator, self).__init__()
        layers = []
        in_features = latent_dim
        for out_features in hidden_layers:
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(activate_function())
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, data_dim))  
        self.model = torch.nn.Sequential(*layers)
        self.__checkpoint = False
        return
 
    def forward(self, z):
        return self.model(z)
 
class Discriminator(torch.nn.Module):
    def __init__(self, 
                 hidden_layers, 
                 activate_function, 
                 data_dim):
        super(Discriminator, self).__init__()
        layers = []
        in_features = data_dim
        for out_features in hidden_layers:
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(activate_function())
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, 1))
        layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*layers)
        self.__checkpoint = False
        return
 
    def forward(self, x):
        return self.model(x)
    
class GANDataAugmentor(DataAugmentorInterface):
    """
    GANDataAugmentor 类
    
    使用生成对抗网络（GAN）进行数据扩充的数据增强器。
    
    参数:
    
    方法:
    - fit(data:torch.Tensor, augmentation_factor:float=1.0) -> 'GANDataAugmentor': 接收数据并进行拟合，返回拟合后的对象本身。
    - predict() -> torch.Tensor: 生成扩充的数据。
    - fit_predict(self, data:torch.Tensor, augmentation_factor:float=1.0) -> torch.Tensor: 结合fit和predict方法，先拟合数据然后生成扩充后的数据。
    """
    def __init__(self, 
                 data:torch.Tensor,
                 latent_dim:int=128, 
                 generator_hidden_layers:list[int]=[16, 32], 
                 discriminator_hidden_layers:list[int]=[32, 16],
                 activate_function:torch.nn.Module=torch.nn.ReLU,
                 criterion:torch.nn.Module=torch.nn.BCELoss
                 ) -> None:
        """
        参数:
        - latent_dim (int): 用于生成扩展样本的高斯噪声维度
        - generator_hidden_layers (list[int]): 生成器隐藏层结构，列表每个元素代表生成器每个隐藏层神经元数量
        - discriminator_hidden_layers (list[int]): 判别器隐藏层结构，列表每个元素代表判别器每个隐藏层神经元数量
        - activate_function (torch.nn.Module): 生成器激活函数，必须继承自torch.nn.Module
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.criterion = criterion()
        self.generator = Generator(latent_dim, generator_hidden_layers, activate_function, data.shape[1])
        self.discriminator = Discriminator(discriminator_hidden_layers, activate_function, data.shape[1])
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.__data__ = data

        return
    
    def fit(self, augmentation_factor:float=2.0) -> 'GANDataAugmentor':
        """
        接收数据并进行拟合。
 
        参数:
        - augmentation_factor (float): 扩充因子，用于控制数据扩充的程度。默认值为1.0。
        - data (torch.Tensor): 要进行拟合的数据。
 
        返回:
        self: 返回已经拟合的当前GANDataAugmentor实例。
        """
        num_epochs = int(augmentation_factor * 100)
        self.__augmentation_factor = augmentation_factor
        batch_size = 64
        
        for epoch in range(num_epochs):
            for i in range(0, len(self.__data__), batch_size):
                #print(i)
                batch_data = self.__data__[i:i+batch_size]
                batch_size_current = batch_data.shape[0]
                
                z = torch.randn(batch_size_current, self.latent_dim)
                fake_data = self.generator(z)
                real_labels = torch.ones(batch_size_current, 1)
                fake_labels = torch.zeros(batch_size_current, 1)
                
                d_loss_real = self.criterion(self.discriminator(batch_data), real_labels)
                d_loss_fake = self.criterion(self.discriminator(fake_data.detach()), fake_labels)
                d_loss = d_loss_real + d_loss_fake
                
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                z = torch.randn(batch_size_current, self.latent_dim)
                g_loss = self.criterion(self.discriminator(self.generator(z)), real_labels)
                
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
        
        self.generator.__checkpoint = True
        self.discriminator.__checkpoint = True
        
        return self
    
    def predict(self) -> torch.Tensor:
        """
        生成并返回扩充的数据。
 
        返回:
        torch.Tensor: 扩充的数据。
        """
        if self.generator.__checkpoint and self.discriminator.__checkpoint:
            z = torch.randn(int((self.__augmentation_factor - 1) * self.__data__.shape[0]), self.latent_dim)
            generated_data = self.generator(z)

            return generated_data
        else:
            raise ModelNotTrainedError('还没拟合呢！')

    def fit_predict(self, data:torch.Tensor, augmentation_factor:float=1.0) -> torch.Tensor:
        """
        结合fit和predict方法，先拟合数据然后生成扩充的数据。

        参数:
        - augmentation_factor (float): 扩充因子，用于控制数据扩充的程度。默认值为1.0。
        - data (torch.Tensor): 要进行拟合和预测的数据。
 
        返回:
        torch.Tensor: 扩充的数据。
        """
        return self.fit(data, augmentation_factor).predict()
    
    def __str__(self):
        return f"GANDataAugmentor(latent_dim={self.latent_dim})"
    

def main() -> None:
    #raise RuntimeError('该方法仅作调试用，该脚本不可调用。')
    test_data = torch.randn(100, 1)
    gan_augmentor = GANDataAugmentor(test_data)
    gan_augmentor.fit()
    augmented_data = gan_augmentor.predict()
 
    #打印原始数据和扩充数据的形状以验证结果
    print("Original data shape:", test_data.shape)
    print("Augmented data shape:", augmented_data.shape)
 
    #可以打印一些生成的数据样本
    print("Some samples of augmented data:")
    print(augmented_data[:5]) 
    return


if __name__ == '__main__':
    main()

