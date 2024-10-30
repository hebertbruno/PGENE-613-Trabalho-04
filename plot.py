import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(history):

    accuracy_path='results/accuracy_plot.jpeg'
    loss_path='results/loss_plot.jpeg'
    # Gráfico de acurácia
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Acurácia ao longo do Treinamento')
    plt.legend()
    plt.savefig(accuracy_path) 
    plt.close() 

    # Gráfico de perda
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda ao longo do Treinamento')
    plt.legend()
    plt.savefig(loss_path)
    plt.close() 
    print("Graficos gerados")

def plot_confusion_matrix(conf_matrix, class_names):
    output_path='results/confusion_matrix.jpeg'
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    print("Matriz de confusao gerada")