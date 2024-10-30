import pandas as pd
from load_data import load_and_preprocess_data
from model import create_lstm_model_pre_processed, load_glove_embeddings, load_custom_embeddings
from model2 import create_lstm_model  # Importa a função do model2
from train import train_model
from evaluate import evaluate_model
from plot import plot_training_history, plot_confusion_matrix
from create_csv import save_predictions_to_csv
import os
import urllib.request
import zipfile
import subprocess

# Variáveis de configuração
embeddings_dir = 'embeddings'
model_choice = 'model2'  # model2 usa embeddings treinados com o modelo, model1 usa pre treinado
embedding_type = 'glove'  # Pode ser 'glove', 'custom_30', 'custom_50', ou 'custom_70', para o caso de pre treinado

##### Configuracoes exigidas para simulacoes
'''
- embedding 50 dim, LSTM 50, descrição 10 palavras
- embedding 50 dim, LSTM 100, descrição 10 palavras
- embedding 30 dim, LSTM 50, descrição 10 palavras
- embedding 70 dim, LSTM 50, descrição 10 palavras
- embedding 50 dim, LSTM 50, descrição 8 palavras
- embedding 50 dim, LSTM 50, descrição 12 palavras
'''

embedding_dim = 50 # Camada de Embedding
lstm_neurons = 50 # Camada LSTM com a quantidade de neuronios especificada
max_len = 12 # Descricao com X palavras
###########################################

# Carregar e preprocessar os dados
X_train, X_val, y_train, y_val, X_test, y_test, tokenizer, desc_test = load_and_preprocess_data('factoryReports.csv', max_len)

#X_train, X_val, y_train, y_val, X_test, y_test, tokenizer = load_and_preprocess_data('factoryReports.csv', max_len)

# Model 1 utiliza embeddings pré treinadas, glove ou usando word2vec
if model_choice == 'model1':
    # embedding de 50 dimensoes glove
    glove_file = os.path.join(embeddings_dir, 'glove.6B.50d.txt')

    # Diretório e download de embeddings, apenas se 'model' for escolhido
    if model_choice == 'model' and not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
        
        if not os.path.exists(glove_file):
            print("Arquivo GloVe não encontrado. Fazendo download...")
            url = "http://nlp.stanford.edu/data/glove.6B.zip"
            zip_path = "glove.6B.zip"
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(embeddings_dir)
            print("Download e extração completos.")
        else:
            print("Arquivo GloVe encontrado. Pulando download.")

    # Embeddings personalizados para o model (opcional)
    embedding_file_30 = os.path.join(embeddings_dir, 'word_embeddings_30.txt')
    embedding_file_50 = os.path.join(embeddings_dir, 'word_embeddings_50.txt')
    embedding_file_70 = os.path.join(embeddings_dir, 'word_embeddings_70.txt')

    if model_choice == 'model' and (not os.path.exists(embedding_file_30) or not os.path.exists(embedding_file_50) or not os.path.exists(embedding_file_70)):
        print("Arquivos de embeddings a partir de factoryReports.csv. Criando os arquivos...")
        subprocess.run(["python", "create_embedding.py"])
    else:
        print("Arquivos de embeddings personalizados encontrados. Pulando criação.")

        print("Utilizando o Modelo 01 com embeddings pre treinadas")
    
    if embedding_type == 'glove':
        print("Utilizando Glove de 50 dimensoes")
        embedding_matrix = load_glove_embeddings(glove_file, tokenizer)
    elif embedding_type == 'custom_30':
        print("Utilizando embedded de 30 dimensoes")
        embedding_matrix = load_custom_embeddings(embedding_file_30, tokenizer, 30)
    elif embedding_type == 'custom_50':
        print("Utilizando embedded de 50 dimensoes")
        embedding_matrix = load_custom_embeddings(embedding_file_50, tokenizer, 50)
    elif embedding_type == 'custom_70':
        print("Utilizando embedded de 70 dimensoes")
        embedding_matrix = load_custom_embeddings(embedding_file_70, tokenizer, 70)
    else:
        raise ValueError("Tipo de embedding inválido.")
    input_shape = (X_train.shape[1],)
    model = create_lstm_model_pre_processed(input_shape, embedding_matrix, lstm_neurons)
# Model 2 utiliza embeddings treinados com o modelo
elif model_choice == 'model2':
    print("Utilizando Modelo 02")
    max_tokens = 5000
    input_shape = (X_train.shape[1],)
    model = create_lstm_model(input_shape, max_tokens, embedding_dim, lstm_neurons)

# Treinar o modelo
history = train_model(model, X_train, y_train, X_val, y_val)

# Avaliar o modelo
conf_matrix, class_names = evaluate_model(model, X_test, y_test, class_names=["Eletronic", "Leak", "Mechanical", "Software"])

# Salvar as previsões em um arquivo CSV
#save_predictions_to_csv(model, X_test, y_test, class_names, file_name='predictions.csv')
save_predictions_to_csv(model, X_test, y_test, class_names, desc_test)

# Resultados
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plot_confusion_matrix(conf_matrix, class_names)
plot_training_history(history)
