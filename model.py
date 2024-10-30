from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
import numpy as np

def load_glove_embeddings(glove_file, tokenizer):
    # Carregar embeddings do GloVe
    embedding_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    # Criar matriz de embeddings
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, 50))  # Dimensão de 50 do GloVe
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_custom_embeddings(embedding_file, tokenizer, embedding_dim):
    # Carregar embeddings personalizados
    embedding_index = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    # Criar matriz de embeddings
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))  # Dimensão de 30, 50 ou 70
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def create_lstm_model_pre_processed(input_shape, embedding_matrix, lstm_neurons):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], 
                        output_dim=embedding_matrix.shape[1],
                        weights=[embedding_matrix], 
                        input_length=input_shape[0], 
                        trainable=False))  # Embedding com pesos fixos
    #Camada LSTM
    model.add(Bidirectional(LSTM(lstm_neurons)))
    
    #model.add(LSTM(100))
    model.add(Dropout(0.2))
    #Camada densa de saida para as 4 categorias
    model.add(Dense(4, activation='softmax'))

    # Compilar o modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
