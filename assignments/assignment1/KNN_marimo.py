import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Задание 1.1 - Метод К-ближайших соседей (K-neariest neighbor classifier)

    В первом задании вы реализуете один из простейших алгоритмов машинного обучения - классификатор на основе метода K-ближайших соседей.
    Мы применим его к задачам
    - бинарной классификации (то есть, только двум классам)
    - многоклассовой классификации (то есть, нескольким классам)

    Так как методу необходим гиперпараметр (hyperparameter) - количество соседей, мы выберем его на основе кросс-валидации (cross-validation).

    Наша основная задача - научиться пользоваться numpy и представлять вычисления в векторном виде, а также ознакомиться с основными метриками, важными для задачи классификации.

    Перед выполнением задания:
    - запустите файл `download_data.sh`, чтобы скачать данные, которые мы будем использовать для тренировки
    - установите все необходимые библиотеки, запустив `pip install -r requirements.txt` (если раньше не работали с `pip`, вам сюда - https://pip.pypa.io/en/stable/quickstart/)

    Если вы раньше не работали с numpy, вам может помочь tutorial. Например этот:
    http://cs231n.github.io/python-numpy-tutorial/
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # '%matplotlib inline' command supported automatically in marimo

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return np, plt


@app.cell
def _():
    from dataset import load_svhn
    from knn import KNN
    from metrics import binary_classification_metrics, multiclass_accuracy

    return KNN, binary_classification_metrics, load_svhn, multiclass_accuracy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Загрузим и визуализируем данные

    В задании уже дана функция `load_svhn`, загружающая данные с диска. Она возвращает данные для тренировки и для тестирования как numpy arrays.

    Мы будем использовать цифры из датасета Street View House Numbers (SVHN, http://ufldl.stanford.edu/housenumbers/), чтобы решать задачу хоть сколько-нибудь сложнее MNIST.
    """)
    return


@app.cell
def _(load_svhn):
    train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)
    return test_X, test_y, train_X, train_y


@app.cell
def _(np, plt, train_X, train_y):
    samples_per_class = 5  # Number of samples per class to visualize
    plot_index = 1
    for example_index in range(samples_per_class):
        for class_index in range(10):
            plt.subplot(5, 10, plot_index)
            image = train_X[train_y == class_index][example_index]
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
            plot_index = plot_index + 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Сначала реализуем KNN для бинарной классификации

    В качестве задачи бинарной классификации мы натренируем модель, которая будет отличать цифру 0 от цифры 9.
    """)
    return


@app.cell
def _(test_X, test_y, train_X, train_y):
    # First, let's prepare the labels and the source data

    # Only select 0s and 9s
    binary_train_mask = (train_y == 0) | (train_y == 9)
    binary_train_X = train_X[binary_train_mask]
    binary_train_y = train_y[binary_train_mask] == 0

    binary_test_mask = (test_y == 0) | (test_y == 9)
    binary_test_X = test_X[binary_test_mask]
    binary_test_y = test_y[binary_test_mask] == 0

    # Reshape to 1-dimensional array [num_samples, 32*32*3]
    binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
    binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)
    return binary_test_X, binary_test_y, binary_train_X, binary_train_y


@app.cell
def _(KNN, binary_train_X, binary_train_y):
    # Create the classifier and call fit to train the model
    # KNN just remembers all the data
    knn_classifier = KNN(k=1)
    knn_classifier.fit(binary_train_X, binary_train_y)
    return (knn_classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Пришло время написать код!

    Последовательно реализуйте функции `compute_distances_two_loops`, `compute_distances_one_loop` и `compute_distances_no_loops`
    в файле `knn.py`.

    Эти функции строят массив расстояний между всеми векторами в тестовом наборе и в тренировочном наборе.
    В результате они должны построить массив размера `(num_test, num_train)`, где координата `[i][j]` соотвествует расстоянию между i-м вектором в test (`test[i]`) и j-м вектором в train (`train[j]`).

    **Обратите внимание** Для простоты реализации мы будем использовать в качестве расстояния меру L1 (ее еще называют [Manhattan distance](https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5_%D0%B3%D0%BE%D1%80%D0%BE%D0%B4%D1%81%D0%BA%D0%B8%D1%85_%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%BE%D0%B2)).

    ![image.png](attachment:image.png)
    """)
    return


@app.cell
def _(binary_test_X, binary_train_X, knn_classifier, np):
    # TODO: implement compute_distances_two_loops in knn.py
    _dists = knn_classifier.compute_distances_two_loops(binary_test_X)
    assert np.isclose(_dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))
    return


@app.cell
def _(binary_test_X, binary_train_X, knn_classifier, np):
    # TODO: implement compute_distances_one_loop in knn.py
    _dists = knn_classifier.compute_distances_one_loop(binary_test_X)
    assert np.isclose(_dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))
    return


@app.cell
def _(binary_test_X, binary_train_X, knn_classifier, np):
    # TODO: implement compute_distances_no_loops in knn.py
    _dists = knn_classifier.compute_distances_no_loops(binary_test_X)
    assert np.isclose(_dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))
    return


@app.cell
def _():
    # Lets look at the performance difference
    # magic command not supported in marimo; please file an issue to add support
    # %timeit knn_classifier.compute_distances_two_loops(binary_test_X)
    # magic command not supported in marimo; please file an issue to add support
    # %timeit knn_classifier.compute_distances_one_loop(binary_test_X)
    # magic command not supported in marimo; please file an issue to add support
    # %timeit knn_classifier.compute_distances_no_loops(binary_test_X)
    return


@app.cell
def _(binary_test_X, knn_classifier):
    # TODO: implement predict_labels_binary in knn.py
    prediction = knn_classifier.predict(binary_test_X)
    return (prediction,)


@app.cell
def _(
    binary_classification_metrics,
    binary_test_y,
    knn_classifier,
    prediction,
):
    # TODO: implement binary_classification_metrics in metrics.py
    _precision, _recall, _f1, _accuracy = binary_classification_metrics(prediction, binary_test_y)
    print('KNN with k = %s' % knn_classifier.k)
    print('Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f' % (_accuracy, _precision, _recall, _f1))
    return


@app.cell
def _(
    KNN,
    binary_classification_metrics,
    binary_test_X,
    binary_test_y,
    binary_train_X,
    binary_train_y,
):
    knn_classifier_3 = KNN(k=3)
    knn_classifier_3.fit(binary_train_X, binary_train_y)
    prediction_1 = knn_classifier_3.predict(binary_test_X)
    _precision, _recall, _f1, _accuracy = binary_classification_metrics(prediction_1, binary_test_y)
    print('KNN with k = %s' % knn_classifier_3.k)
    print('Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f' % (_accuracy, _precision, _recall, _f1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Кросс-валидация (cross-validation)

    Попробуем найти лучшее значение параметра k для алгоритма KNN!

    Для этого мы воспользуемся k-fold cross-validation (https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). Мы разделим тренировочные данные на 5 фолдов (folds), и по очереди будем использовать каждый из них в качестве проверочных данных (validation data), а остальные -- в качестве тренировочных (training data).

    В качестве финальной оценки эффективности k мы усредним значения F1 score на всех фолдах.
    После этого мы просто выберем значение k с лучшим значением метрики.

    *Бонус*: есть ли другие варианты агрегировать F1 score по всем фолдам? Напишите плюсы и минусы в клетке ниже.
    """)
    return


@app.cell
def _():
    # Find the best k using cross-validation based on F1 score
    _num_folds = 5
    _train_folds_X = []
    _train_folds_y = []
    _k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
    # TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y
    k_to_f1 = {}
    for _k in _k_choices:
        pass  # dict mapping k values to mean F1 scores (int -> float)
    for _k in sorted(k_to_f1):
        print('k = %d, f1 = %f' % (_k, k_to_f1[_k]))  # TODO: perform cross-validation  # Go through every fold and use it for testing and all other folds for training  # Perform training and produce F1 score metric on the validation dataset  # Average F1 from all the folds and write it into k_to_f1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Проверим, как хорошо работает лучшее значение k на тестовых данных (test data)
    """)
    return


@app.cell
def _(
    KNN,
    binary_classification_metrics,
    binary_test_X,
    binary_test_y,
    binary_train_X,
    binary_train_y,
):
    _best_k = 1
    _best_knn_classifier = KNN(k=_best_k)
    _best_knn_classifier.fit(binary_train_X, binary_train_y)
    prediction_2 = _best_knn_classifier.predict(binary_test_X)
    _precision, _recall, _f1, _accuracy = binary_classification_metrics(prediction_2, binary_test_y)
    print('Best KNN with k = %s' % _best_k)
    print('Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f' % (_accuracy, _precision, _recall, _f1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Многоклассовая классификация (multi-class classification)

    Переходим к следующему этапу - классификации на каждую цифру.
    """)
    return


@app.cell
def _(KNN, test_X, train_X, train_y):
    # Now let's use all 10 classes
    train_X_1 = train_X.reshape(train_X.shape[0], -1)
    test_X_1 = test_X.reshape(test_X.shape[0], -1)
    knn_classifier_1 = KNN(k=1)
    knn_classifier_1.fit(train_X_1, train_y)
    return knn_classifier_1, test_X_1, train_X_1


@app.cell
def _(knn_classifier_1, test_X_1):
    # TODO: Implement predict_labels_multiclass
    predict = knn_classifier_1.predict(test_X_1)
    return (predict,)


@app.cell
def _(multiclass_accuracy, predict, test_y):
    # TODO: Implement multiclass_accuracy
    _accuracy = multiclass_accuracy(predict, test_y)
    print('Accuracy: %4.2f' % _accuracy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Снова кросс-валидация. Теперь нашей основной метрикой стала точность (accuracy), и ее мы тоже будем усреднять по всем фолдам.
    """)
    return


@app.cell
def _():
    # Find the best k using cross-validation based on accuracy
    _num_folds = 5
    _train_folds_X = []
    _train_folds_y = []
    _k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
    # TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y
    k_to_accuracy = {}
    for _k in _k_choices:
        pass
    for _k in sorted(k_to_accuracy):
        print('k = %d, accuracy = %f' % (_k, k_to_accuracy[_k]))  # TODO: perform cross-validation  # Go through every fold and use it for testing and all other folds for validation  # Perform training and produce accuracy metric on the validation dataset  # Average accuracy from all the folds and write it into k_to_accuracy
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Финальный тест - классификация на 10 классов на тестовой выборке (test data)

    Если все реализовано правильно, вы должны увидеть точность не менее **0.2**.
    """)
    return


@app.cell
def _(KNN, multiclass_accuracy, test_X_1, test_y, train_X_1, train_y):
    _best_k = 1
    _best_knn_classifier = KNN(k=_best_k)
    _best_knn_classifier.fit(train_X_1, train_y)
    prediction_3 = _best_knn_classifier.predict(test_X_1)
    _accuracy = multiclass_accuracy(prediction_3, test_y)
    print('Accuracy: %4.2f' % _accuracy)
    return


if __name__ == "__main__":
    app.run()
