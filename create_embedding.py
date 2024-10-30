import pandas as pd
import os
from gensim.models import Word2Vec

# Criar uma pasta para os embeddings, se não existir
embeddings_dir = 'embeddings'

# Carregar o CSV
data = pd.read_csv('factoryReports.csv')

# Extraindo as descrições e tokenizando
sentences = data['Description'].apply(lambda x: x.split()).tolist()

# Treinar modelos de embeddings
model_30 = Word2Vec(sentences, vector_size=30, window=5, min_count=1, workers=4)
model_30.save(os.path.join(embeddings_dir, "my_word2vec_30.model"))

model_50 = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4)
model_50.save(os.path.join(embeddings_dir, "my_word2vec_50.model"))

model_70 = Word2Vec(sentences, vector_size=70, window=5, min_count=1, workers=4)
model_70.save(os.path.join(embeddings_dir, "my_word2vec_70.model"))

# Função para salvar os embeddings em arquivos de texto
def save_embeddings(model, filename):
    with open(filename, 'w') as f:
        for word in model.wv.key_to_index:
            f.write(f"{word} {' '.join(map(str, model.wv[word]))}\n")

# Salvar os embeddings em arquivos de texto na pasta
save_embeddings(model_30, os.path.join(embeddings_dir, "word_embeddings_30.txt"))
save_embeddings(model_50, os.path.join(embeddings_dir, "word_embeddings_50.txt"))
save_embeddings(model_70, os.path.join(embeddings_dir, "word_embeddings_70.txt"))

print("Embeddings personalizados criados com sucesso.")
