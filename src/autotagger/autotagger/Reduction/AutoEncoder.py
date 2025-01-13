import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()  
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def fit_predict(data, hidden_dim, epochs=50, batch_size=32, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = data.shape[1]
    model = Autoencoder(input_dim, hidden_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for batch_data in dataloader:
            batch_data = batch_data[0].to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(data):
        model.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32).to(device)
            return model.encoder(data).cpu().numpy()
    
    return predict

if __name__ == "__main__":
    sample_data = torch.randn(100, 20)  # 100 samples, 20 features each
    hidden_dimension = 2
    
    predict_fn = fit_predict(sample_data, hidden_dimension, epochs=50, batch_size=16, learning_rate=1e-3)
    
    predictions = predict_fn(sample_data)
    print(predictions)

    x = predictions[:, 0]
    y = predictions[:, 1]
 
    plt.scatter(x, y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Tensor Scatter Plot')
    plt.grid(True)
    plt.show()