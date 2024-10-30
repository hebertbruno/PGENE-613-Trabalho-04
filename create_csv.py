import pandas as pd

def save_predictions_to_csv(model, X_test, y_test, class_names, descriptions, file_name='predictions.csv'):
    predictions = model.predict(X_test)
    predicted_labels = predictions.argmax(axis=1)

    y_true_labels = [class_names[i] for i in y_test]
    y_pred_labels = [class_names[i] for i in predicted_labels]

    # Criar DataFrame com as descrições
    df = pd.DataFrame({'Descrição': descriptions, 'Real': y_true_labels, 'Predito': y_pred_labels})

    # Salvar em CSV
    df.to_csv(file_name, index=False, sep=';')
    print(f"Arquivo '{file_name}' salvo com sucesso.")
