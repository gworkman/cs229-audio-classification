from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['Heiti TC']


for n_mfccs in [20, 30, 40, 50, 60]:
    df = pd.read_pickle(f'../data/preprocessed/mfccs_{n_mfccs}.pkl')
    df = df.explode('mfccs')

    X = np.array(df['mfccs'].apply(lambda m: m.flatten()).to_list())
    y = df['tone'].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    model = LogisticRegression(penalty='l2', C=0.0001, multi_class='ovr')
    model.fit(X_train, y_train)
    y_hat = model.predict(X_train)

    train_accuracy = np.mean(y_train == y_hat)

    y_hat = model.predict(X_test)

    test_accuracy = np.mean(y_test == y_hat)

    print(f'Train accuracy: {train_accuracy:0.2f} Test accuracy: {test_accuracy:0.2f}')

    fig, ax = plt.subplots()
    display_labels = [f'Tone {i}' for i in np.arange(4) + 1]
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_hat, display_labels=display_labels, ax=ax)
    plt.title(f'Confusion for Logistic Regression, N_MFCCs = {n_mfccs} (Test accuracy: {test_accuracy:0.2f})')
    plt.savefig(f'../images/logistic_regression_confusion_nmfccs={n_mfccs}.png')
