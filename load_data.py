import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path, max_len):
    # Carregar o CSV
    data = pd.read_csv(file_path)
    descriptions = data['Description'].values
    categories = data['Category'].values


    # Contar o número de amostras em cada categoria
    category_counts = data['Category'].value_counts()
    print("Número de amostras por categoria:")
    print(category_counts)

    # Codificação das categorias
    label_encoder = LabelEncoder()
    categories = label_encoder.fit_transform(categories)

    # Tokenização
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptions)
    sequences = tokenizer.texts_to_sequences(descriptions)

    # Padding
    padded_sequences = pad_sequences(sequences, max_len, padding='post')

    # Dividir os dados (padded_sequences e categories)
    X_train, X_temp, y_train, y_temp = train_test_split(
        padded_sequences, categories, test_size=0.2, random_state=56, stratify=categories
    )

    # Dividir as descrições separadamente para manter a ordem
    desc_train, desc_temp = train_test_split(
        descriptions, test_size=0.2, random_state=56, stratify=categories
    )

    # Dividir X_temp, y_temp, e desc_temp em validação e teste
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=56, stratify=y_temp
    )
    desc_val, desc_test = train_test_split(
        desc_temp, test_size=0.5, random_state=56, stratify=y_temp
    )

   

    return X_train, X_val, y_train, y_val, X_test, y_test, tokenizer, desc_test
