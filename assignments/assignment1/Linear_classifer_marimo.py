import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Задание 1.2 - Линейный классификатор (Linear classifier)

    В этом задании мы реализуем другую модель машинного обучения - линейный классификатор. Линейный классификатор подбирает для каждого класса веса, на которые нужно умножить значение каждого признака и потом сложить вместе.
    Тот класс, у которого эта сумма больше, и является предсказанием модели.

    В этом задании вы:
    - потренируетесь считать градиенты различных многомерных функций
    - реализуете подсчет градиентов через линейную модель и функцию потерь softmax
    - реализуете процесс тренировки линейного классификатора
    - подберете параметры тренировки на практике

    На всякий случай, еще раз ссылка на туториал по numpy:
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
    from dataset import load_svhn, random_split_train_val
    from gradient_check import check_gradient
    from metrics import multiclass_accuracy 
    import linear_classifer

    return (
        check_gradient,
        linear_classifer,
        load_svhn,
        multiclass_accuracy,
        random_split_train_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Как всегда, первым делом загружаем данные

    Мы будем использовать все тот же SVHN.
    """)
    return


@app.cell
def _(load_svhn, np, random_split_train_val):
    def prepare_for_linear_classifier(train_X, test_X):
        train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
        test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0

        # Subtract mean
        mean_image = np.mean(train_flat, axis = 0)
        train_flat -= mean_image
        test_flat -= mean_image

        # Add another channel with ones as a bias term
        train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0], 1))])
        test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0], 1))])    
        return train_flat_with_ones, test_flat_with_ones

    train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)    
    train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
    # Split train into train and val
    train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val = 1000)
    return test_X, test_y, train_X, train_y, val_X, val_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Играемся с градиентами!

    В этом курсе мы будем писать много функций, которые вычисляют градиенты аналитическим методом.

    Все функции, в которых мы будем вычислять градиенты, будут написаны по одной и той же схеме.
    Они будут получать на вход точку, где нужно вычислить значение и градиент функции, а на выходе будут выдавать кортеж (tuple) из двух значений - собственно значения функции в этой точке (всегда одно число) и аналитического значения градиента в той же точке (той же размерности, что и вход).
    ```
    def f(x):
        \"\"\"
        Computes function and analytic gradient at x

        x: np array of float, input to the function

        Returns:
        value: float, value of the function
        grad: np array of float, same shape as x
        \"\"\"
        ...

        return value, grad
    ```

    Необходимым инструментом во время реализации кода, вычисляющего градиенты, является функция его проверки. Эта функция вычисляет градиент численным методом и сверяет результат с градиентом, вычисленным аналитическим методом.

    Мы начнем с того, чтобы реализовать вычисление численного градиента (numeric gradient) в функции `check_gradient` в `gradient_check.py`. Эта функция будет принимать на вход функции формата, заданного выше, использовать значение `value` для вычисления численного градиента и сравнит его с аналитическим - они должны сходиться.

    Напишите часть функции, которая вычисляет градиент с помощью численной производной для каждой координаты. Для вычисления производной используйте так называемую two-point formula (https://en.wikipedia.org/wiki/Numerical_differentiation):

    ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/22fc2c0a66c63560a349604f8b6b39221566236d)

    Все функции приведенные в следующей клетке должны проходить gradient check.
    """)
    return


@app.cell
def _(check_gradient, np):
    # TODO: Implement check_gradient function in gradient_check.py
    # All the functions below should pass the gradient check

    def square(x):
        return float(x*x), 2*x

    check_gradient(square, np.array([3.0]))

    def array_sum(x):
        assert x.shape == (2,), x.shape
        return np.sum(x), np.ones_like(x)

    check_gradient(array_sum, np.array([3.0, 2.0]))

    def array_2d_sum(x):
        assert x.shape == (2,2)
        return np.sum(x), np.ones_like(x)

    check_gradient(array_2d_sum, np.array([[3.0, 2.0], [1.0, 0.0]]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Начинаем писать свои функции, считающие аналитический градиент

    Теперь реализуем функцию softmax, которая получает на вход оценки для каждого класса и преобразует их в вероятности от 0 до 1:
    ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3)

    **Важно:** Практический аспект вычисления этой функции заключается в том, что в ней учавствует вычисление экспоненты от потенциально очень больших чисел - это может привести к очень большим значениям в числителе и знаменателе за пределами диапазона float.

    К счастью, у этой проблемы есть простое решение -- перед вычислением softmax вычесть из всех оценок максимальное значение среди всех оценок:
    ```
    predictions -= np.max(predictions)
    ```
    (подробнее здесь - http://cs231n.github.io/linear-classify/#softmax, секция `Practical issues: Numeric stability`)
    """)
    return


@app.cell
def _(linear_classifer, np):
    # TODO Implement softmax and cross-entropy for single sample
    _probs = linear_classifer.softmax(np.array([-10, 0, 10]))
    _probs = linear_classifer.softmax(np.array([1000, 0, 0]))
    # Make sure it works for big numbers too!
    assert np.isclose(_probs[0], 1.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Кроме этого, мы реализуем cross-entropy loss, которую мы будем использовать как функцию ошибки (error function).
    В общем виде cross-entropy определена следующим образом:
    ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/0cb6da032ab424eefdca0884cd4113fe578f4293)

    где x - все классы, p(x) - истинная вероятность принадлежности сэмпла классу x, а q(x) - вероятность принадлежности классу x, предсказанная моделью.
    В нашем случае сэмпл принадлежит только одному классу, индекс которого передается функции. Для него p(x) равна 1, а для остальных классов - 0.

    Это позволяет реализовать функцию проще!
    """)
    return


@app.cell
def _(linear_classifer, np):
    _probs = linear_classifer.softmax(np.array([-5, 0, 5]))
    linear_classifer.cross_entropy_loss(_probs, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    После того как мы реализовали сами функции, мы можем реализовать градиент.

    Оказывается, что вычисление градиента становится гораздо проще, если объединить эти функции в одну, которая сначала вычисляет вероятности через softmax, а потом использует их для вычисления функции ошибки через cross-entropy loss.

    Эта функция `softmax_with_cross_entropy` будет возвращает и значение ошибки, и градиент по входным параметрам. Мы проверим корректность реализации с помощью `check_gradient`.
    """)
    return


@app.cell
def _(check_gradient, linear_classifer, np):
    # TODO Implement combined function or softmax and cross entropy and produces gradient
    _loss, grad = linear_classifer.softmax_with_cross_entropy(np.array([1, 0, 0]), 1)
    check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, 1), np.array([1, 0, 0], np.float))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В качестве метода тренировки мы будем использовать стохастический градиентный спуск (stochastic gradient descent или SGD), который работает с батчами сэмплов.

    Поэтому все наши фукнции будут получать не один пример, а батч, то есть входом будет не вектор из `num_classes` оценок, а матрица размерности `batch_size, num_classes`. Индекс примера в батче всегда будет первым измерением.

    Следующий шаг - переписать наши функции так, чтобы они поддерживали батчи.

    Финальное значение функции ошибки должно остаться числом, и оно равно среднему значению ошибки среди всех примеров в батче.
    """)
    return


@app.cell
def _(check_gradient, linear_classifer, np):
    # TODO Extend combined function so it can receive a 2d array with batch of samples
    np.random.seed(42)
    # Test batch_size = 1
    _num_classes = 4
    _batch_size = 1
    predictions = np.random.randint(-1, 3, size=(_batch_size, _num_classes)).astype(np.float)
    _target_index = np.random.randint(0, _num_classes, size=(_batch_size, 1)).astype(np.int)
    check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, _target_index), predictions)
    _num_classes = 4
    # Test batch_size = 3
    _batch_size = 3
    predictions = np.random.randint(-1, 3, size=(_batch_size, _num_classes)).astype(np.float)
    _target_index = np.random.randint(0, _num_classes, size=(_batch_size, 1)).astype(np.int)
    check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, _target_index), predictions)
    _probs = linear_classifer.softmax(np.array([[20, 0, 0], [1000, 0, 0]]))
    # Make sure maximum subtraction for numberic stability is done separately for every sample in the batch
    assert np.all(np.isclose(_probs[:, 0], 1.0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Наконец, реализуем сам линейный классификатор!

    softmax и cross-entropy получают на вход оценки, которые выдает линейный классификатор.

    Он делает это очень просто: для каждого класса есть набор весов, на которые надо умножить пиксели картинки и сложить. Получившееся число и является оценкой класса, идущей на вход softmax.

    Таким образом, линейный классификатор можно представить как умножение вектора с пикселями на матрицу W размера `num_features, num_classes`. Такой подход легко расширяется на случай батча векторов с пикселями X размера `batch_size, num_features`:

    `predictions = X * W`, где `*` - матричное умножение.

    Реализуйте функцию подсчета линейного классификатора и градиентов по весам `linear_softmax` в файле `linear_classifer.py`
    """)
    return


@app.cell
def _(check_gradient, linear_classifer, np):
    # TODO Implement linear_softmax function that uses softmax with cross-entropy for linear classifier
    _batch_size = 2
    _num_classes = 2
    num_features = 3
    np.random.seed(42)
    W = np.random.randint(-1, 3, size=(num_features, _num_classes)).astype(np.float)
    X = np.random.randint(-1, 3, size=(_batch_size, num_features)).astype(np.float)
    _target_index = np.ones(_batch_size, dtype=np.int)
    _loss, dW = linear_classifer.linear_softmax(X, W, _target_index)
    check_gradient(lambda w: linear_classifer.linear_softmax(X, w, _target_index), W)
    return (W,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### И теперь регуляризация

    Мы будем использовать L2 regularization для весов как часть общей функции ошибки.

    Напомним, L2 regularization определяется как

    l2_reg_loss = regularization_strength * sum<sub>ij</sub> W[i, j]<sup>2</sup>

    Реализуйте функцию для его вычисления и вычисления соотвествующих градиентов.
    """)
    return


@app.cell
def _(W, check_gradient, linear_classifer):
    # TODO Implement l2_regularization function that implements loss for L2 regularization
    linear_classifer.l2_regularization(W, 0.01)
    check_gradient(lambda w: linear_classifer.l2_regularization(w, 0.01), W)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Тренировка!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Градиенты в порядке, реализуем процесс тренировки!
    """)
    return


@app.cell
def _(linear_classifer, train_X, train_y):
    # TODO: Implement LinearSoftmaxClassifier.fit function
    classifier = linear_classifer.LinearSoftmaxClassifier()
    loss_history = classifier.fit(train_X, train_y, epochs=10, learning_rate=1e-3, batch_size=300, reg=1e1)
    return classifier, loss_history


@app.cell
def _(loss_history, plt):
    # let's look at the loss history!
    plt.plot(loss_history)
    return


@app.cell
def _(classifier, multiclass_accuracy, train_X, train_y, val_X, val_y):
    # Let's check how it performs on validation set
    pred = classifier.predict(val_X)
    accuracy = multiclass_accuracy(pred, val_y)
    print("Accuracy: ", accuracy)

    # Now, let's train more and see if it performs better
    classifier.fit(train_X, train_y, epochs=100, learning_rate=1e-3, batch_size=300, reg=1e1)
    pred = classifier.predict(val_X)
    accuracy = multiclass_accuracy(pred, val_y)
    print("Accuracy after training for 100 epochs: ", accuracy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Как и раньше, используем кросс-валидацию для подбора гиперпараметтов.

    В этот раз, чтобы тренировка занимала разумное время, мы будем использовать только одно разделение на тренировочные (training) и проверочные (validation) данные.

    Теперь нам нужно подобрать не один, а два гиперпараметра! Не ограничивайте себя изначальными значениями в коде.
    Добейтесь точности более чем **20%** на проверочных данных (validation data).
    """)
    return


@app.cell
def _():
    num_epochs = 200
    _batch_size = 300
    learning_rates = [0.001, 0.0001, 1e-05]
    reg_strengths = [0.0001, 1e-05, 1e-06]
    best_classifier = None
    best_val_accuracy = None
    # TODO use validation set to find the best hyperparameters
    # hint: for best results, you might need to try more values for learning rate and regularization strength 
    # than provided initially
    print('best validation accuracy achieved: %f' % best_val_accuracy)
    return (best_classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Какой же точности мы добились на тестовых данных?
    """)
    return


@app.cell
def _(best_classifier, multiclass_accuracy, test_X, test_y):
    test_pred = best_classifier.predict(test_X)
    test_accuracy = multiclass_accuracy(test_pred, test_y)
    print('Linear softmax classifier test set accuracy: %f' % (test_accuracy, ))
    return


if __name__ == "__main__":
    app.run()
