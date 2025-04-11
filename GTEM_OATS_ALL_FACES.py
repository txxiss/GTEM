import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import plotly.graph_objects as go

# Função para carregar os dados de cada face do CSV
def load_data(filename, col_dB):
    df = pd.read_csv(filename)
    freq = df['freq_MHz'].values  # Frequências em Hz (mesmo que a coluna se chame freq_MHz)
    valores = df[col_dB].values   # Valores de dBuV/m
    return freq, valores

# Carregar os dados dos arquivos CSV
freq_x, dBuV_x = load_data('face_x.csv', 'dBuV_x')
freq_y, dBuV_y = load_data('face_y.csv', 'dBuV_y')
freq_z, dBuV_z = load_data('face_z.csv', 'dBuV_z')

# Determinar a faixa de frequências comum entre as faces
# Usando o máximo do menor valor e o mínimo do maior valor
freq_start = max(freq_x[0], freq_y[0], freq_z[0])
freq_end   = min(freq_x[-1], freq_y[-1], freq_z[-1])

# Definir o passo de frequência a partir dos dados (por exemplo, mediana dos incrementos da face x)
delta_x = np.median(np.diff(freq_x))
print(f"Passo calculado (face x): {delta_x}")
# Se preferir, pode usar um valor fixo – mas o importante é que seja compatível com a amostragem

# Criar o vetor de referência com a faixa comum e o passo adequado
freq_ref = np.arange(freq_start, freq_end + delta_x, delta_x)

# Função para alinhar os dados ao vetor de referência
def align_data(freq_data, values, freq_ref):
    freq_data_reshaped = freq_data.reshape(-1, 1)
    freq_ref_reshaped = freq_ref.reshape(-1, 1)
    
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(freq_data_reshaped)
    distances, indices = neigh.kneighbors(freq_ref_reshaped)
    
    aligned_values = values[indices.flatten()]
    return aligned_values

# Alinhar os valores para cada face usando a faixa comum
dBuV_x_aligned = align_data(freq_x, dBuV_x, freq_ref)
dBuV_y_aligned = align_data(freq_y, dBuV_y, freq_ref)
dBuV_z_aligned = align_data(freq_z, dBuV_z, freq_ref)

# Converter os dados alinhados para tensores do PyTorch
tensor_x = torch.tensor(dBuV_x_aligned, dtype=torch.float32)
tensor_y = torch.tensor(dBuV_y_aligned, dtype=torch.float32)
tensor_z = torch.tensor(dBuV_z_aligned, dtype=torch.float32)

# Para cada ponto de frequência, extrair o maior valor dentre as três faces
max_xy = torch.max(tensor_x, tensor_y)
unified_tensor = torch.max(max_xy, tensor_z)
unified_values = unified_tensor.numpy()

# Plotar os gráficos com Plotly usando scattergl para os pontos e linhas para melhor visualização
fig = go.Figure()

# Gráfico de pontos (scattergl) dos valores unificados
fig.add_trace(go.Scattergl(
    x = freq_ref / 1e6,  # Converter para MHz para visualização
    y = unified_values,
    mode = 'markers',
    name = 'EUT pontos (máximo entre x,y,z)',
    marker=dict(color='red', size=5)
))

# Linha unificada dos valores máximos
fig.add_trace(go.Scatter(
    x = freq_ref / 1e6,
    y = unified_values,
    mode = 'lines',
    name = 'EUT linha (máximo entre x,y,z)',
    line=dict(color='black')
))

# Plotando as curvas alinhadas de cada face
fig.add_trace(go.Scatter(
    x = freq_ref / 1e6,
    y = dBuV_x_aligned,
    mode = 'lines',
    name = 'Face x',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x = freq_ref / 1e6,
    y = dBuV_y_aligned,
    mode = 'lines',
    name = 'Face y',
    line=dict(color='orange')
))

fig.add_trace(go.Scatter(
    x = freq_ref / 1e6,
    y = dBuV_z_aligned,
    mode = 'lines',
    name = 'Face z',
    line=dict(color='green')
))

fig.update_layout(
    title='Comportamento do EUT - Valor Máximo de dBuV/m (faces x, y, z)',
    xaxis_title='Frequência (MHz)',
    yaxis_title='dBuV/m',
    # Atenção: se utilizar escala logarítmica certifique-se que os dados sejam positivos e consistentes
    yaxis_type='log',
    xaxis_type='log',
    template='plotly_white'
)

fig.show()
