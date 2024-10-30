import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_test, y_test, class_names):
    # Avaliar o modelo no conjunto de teste
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')
    print(f'Perda no conjunto de teste: {test_loss * 100:.2f}%')
    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Gerar a tabela de confusão
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    

    # Retorne a matriz de confusão e os nomes das classes
    return conf_matrix, class_names
