import pandas as pd
import numpy as np

# Carregar CSVs garantindo o range correto
df_x = pd.read_csv('face_x.csv', names=['Freq_MHz', 'dBuV_x'])
df_y = pd.read_csv('face_y.csv', names=['Freq_MHz', 'dBuV_y'])
df_z = pd.read_csv('face_z.csv', names=['Freq_MHz', 'dBuV_z'])

freqs_unicas = np.unique(np.concatenate([df_x['Freq_MHz'], df_y['Freq_MHz'], df_z['Freq_MHz']]))
df_unified = pd.DataFrame({'Freq_MHz': freqs_unicas})

# Unificar usando inner join para manter apenas frequências presentes em todos
df_unified = pd.merge(pd.merge(df_x, df_y, on='Freq_MHz', how='inner'), df_z, on='Freq_MHz', how='inner')
df_unified['dBuV_max'] = df_unified[['dBuV_y', 'dBuV_x', 'dBuV_z']].max(axis=1)

df_unified.to_csv('resultados_2.csv', index=False)

import plotly.graph_objects as go

fig = go.Figure()

# Adicionar marcadores brutos
fig.add_trace(go.Scattergl(
    x=df_unified['Freq_MHz'], 
    y=df_unified['dBuV_max'],
    #y=df_unified['dBuV_x'],
    mode='markers',
    marker=dict(
        color='black',
        size=3,           # Tamanho dos marcadores
        opacity=0.5
    ),
    name='Dados Brutos'
))

fig.add_trace(go.Scattergl(
    x=df_unified['Freq_MHz'], 
    y=df_unified['dBuV_x'],
    mode='markers',
    marker=dict(
        color='blue',
        size=3,           # Tamanho dos marcadores
        opacity=0.5
    ),
    name='Face x'
))

fig.add_trace(go.Scattergl(
    x=df_unified['Freq_MHz'], 
    y=df_unified['dBuV_y'],
    mode='markers',
    marker=dict(
        color='green',
        size=3,           # Tamanho dos marcadores
        opacity=0.5
    ),
    name='Face y'
))

fig.add_trace(go.Scattergl(
    x=df_unified['Freq_MHz'], 
    y=df_unified['dBuV_z'],
    mode='markers',
    marker=dict(
        color='red',
        size=3,           # Tamanho dos marcadores
        opacity=0.5
    ),
    name='Face z'
))

fig.update_layout(
    title='Teste de Imunidade Radiada (Apenas Marcadores)',
    xaxis_title='Frequência (MHz)',
    yaxis_title='dBµV/m',
    yaxis_type='log',
    xaxis_type='log',
    template='plotly_white',
    showlegend=True
)

fig.show()
