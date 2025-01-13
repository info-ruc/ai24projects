import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAReducer:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() 

    def fit(self, data):
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)
        self.pca.fit(scaled_data)

    def transform(self, data):
        scaled_data = self.scaler.transform(data)
        reduced_data = self.pca.transform(scaled_data)
        return reduced_data

    def inverse_transform(self, reduced_data):
        scaled_reconstructed_data = self.pca.inverse_transform(reduced_data)
        reconstructed_data = self.scaler.inverse_transform(scaled_reconstructed_data)
        return reconstructed_data


if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.rand(100, 50)
    n_components = 10
    pca_reducer = PCAReducer(n_components=n_components)
    pca_reducer.fit(data)
    reduced_data = pca_reducer.transform(data)
    print("Reduced data shape:", reduced_data.shape) 

    # 如果需要，可以将降维后的数据逆变换回原始空间（尽管会有信息损失）
    reconstructed_data = pca_reducer.inverse_transform(reduced_data)
    print("Reconstructed data shape:", reconstructed_data.shape)  