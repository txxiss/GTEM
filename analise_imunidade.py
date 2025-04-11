import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model  # Importação correta
from tensorflow.keras.layers import Input, Dense  # Importação correta
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Configurações iniciais
DEVICE = torch.device('cpu')
BATCH_SIZE = 1024
N_CLUSTERS = 5
FREQ_BANDS = [25e6, 50e6, 1e9]  # Ajuste conforme necessário

class EUTDataset(Dataset):
    """Dataset para carregamento eficiente de dados"""
    def __init__(self, dataframe):
        self.X = torch.tensor(dataframe[['x', 'y', 'z']].values, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(dataframe['max_dBm'].values, dtype=torch.float32).to(DEVICE)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FusionModel(nn.Module):
    """Modelo de fusão de eixos com PyTorch"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

def load_and_process_data():
    """Carrega e processa dados mantendo todas as entradas originais"""
    # 1. Carregar dados sem alterações
    df_x = pd.read_csv('dados_eixo_x.csv').rename(columns={'dB/m': 'x'})
    df_y = pd.read_csv('dados_eixo_y.csv').rename(columns={'dB/m': 'y'})
    df_z = pd.read_csv('dados_eixo_z.csv').rename(columns={'dB/m': 'z'})

    # 2. Merge mantendo todas as linhas de todos os datasets
    df = pd.merge(
        left=pd.merge(df_x, df_y, on='frequencia', how='outer'),
        right=df_z,
        on='frequencia',
        how='outer'
    )

    # 3. Ordenar e marcar origens
    df = df.sort_values('frequencia')
    
    # 4. Preencher valores ausentes
    df[['x', 'y', 'z']] = (
    df[['x', 'y', 'z']]
    .interpolate(method='linear')  # Interpolação linear primeiro
    .bfill()                       # Preenchimento para trás (backfill)
)
    
    # 5. Calcular métricas
    df['max_dBm'] = df[['x', 'y', 'z']].max(axis=1)
    df['rms_dBm'] = np.sqrt((df[['x', 'y', 'z']]**2).mean(axis=1))
    
    # 6. Adicionar faixas
    df['faixa'] = pd.cut(df['frequencia'], bins=FREQ_BANDS)
    
    return df

def train_pytorch_model(df):
    """Treina o modelo de fusão com PyTorch"""
    dataset = EUTDataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = FusionModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Treinamento rápido (ajuste epochs conforme necessário)
    model.train()
    for epoch in range(50):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
    
    # Predição
    with torch.no_grad():
        df['predicted_dBm'] = model(dataset.X).cpu().numpy().flatten()
    
    return df

def detect_anomalies(df):
    """Detecção de anomalias com Autoencoder"""
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['x', 'y', 'z']])
    
    # Modelo Autoencoder
    input_dim = X_scaled.shape[1]
    encoding_dim = 2
    
    inputs = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(inputs)
    decoded = Dense(input_dim, activation='linear')(encoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Treinamento
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=BATCH_SIZE, verbose=0)
    
    # Cálculo de anomalias
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
    df['anomaly_score'] = mse
    df['anomaly'] = df['anomaly_score'] > np.percentile(mse, 95)
    
    return df

def visualize_results(df):
    """Visualização interativa dos resultados"""
    fig = go.Figure()
    
    # Dados originais
    for axis in ['x', 'y', 'z']:
        fig.add_trace(go.Scattergl(
            x=df['frequencia'], 
            y=df[axis], 
            name=f'Eixo {axis.upper()}',
            mode='markers',
            marker=dict(size=2)
        ))
    
    # Predições e anomalias
    #fig.add_trace(go.Scattergl(
    #    x=df['frequencia'],
    #    y=df['predicted_dBm'],
    #    name='Fusão Neural',
    #    line=dict(color='black', width=1)
    #))
    
    fig.add_trace(go.Scattergl(
        x=df[df['anomaly']]['frequencia'],
        y=df[df['anomaly']]['predicted_dBm'],
        mode='markers',
        name='Anomalias',
        marker=dict(color='red', size=5)
    ))
    
    fig.update_layout(
        title='Análise Completa de Imunidade Radiada',
        xaxis_title='Frequência (Hz)',
        yaxis_title='dB/m',
        template='plotly_white'
    )
    fig.show()

def main():
    # Pipeline principal
    df = load_and_process_data()
    df = train_pytorch_model(df)
    df = detect_anomalies(df)
    
    # Clusterização com K-Means
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[['x', 'y', 'z']])
    df['cluster'] = KMeans(n_clusters=N_CLUSTERS).fit_predict(X_cluster)
    
    visualize_results(df)
    
    # Salvar resultados
    df.to_csv('resultados_completos.csv', index=False)
if __name__ == "__main__":
    main()