from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Lambda, Embedding, Bidirectional
import tensorflow as tf

def create_lstm_model(input_shape, max_tokens, embedding_dim, lstm_neurons):
    model = Sequential()
    
    # 
    #model.add(Lambda(lambda x: tf.one_hot(x, depth=max_tokens), 
    #                 output_shape=(input_shape[0], max_tokens),
    #                 input_shape=(input_shape[0],)))
    model.add(Embedding(input_dim=max_tokens, 
                         output_dim=embedding_dim, 
                         input_length=input_shape[0]))
    # Camadas LSTM
    model.add(Bidirectional(LSTM(lstm_neurons)))  # Primeira camada LSTM
    model.add(Dropout(0.5))
    #model.add(LSTM(100)) 
    
    # Camada densa de saída para as 4 categorias
    model.add(Dense(4, activation='softmax'))
    
    # Compilar o modelo
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
