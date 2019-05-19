import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')

"""
-Hello world of Deep learning
-MNIST ("Modified National Institute of Standards and Technology")，计算机视觉新手村
-（28pi*28pi=）包含784个像素点的28*28的像素矩阵，每个点是0-255的整数，表示灰度，数字越大越暗


-像素矩阵和csv列对应关系
001 002 003
004 *** 006
007 008 009
"""


# 分析数据
def analize():
    # 数字出现总数求和，柱状图
    g = sns.countplot(Y_train)
    plt.show()

    # 相当于group by求和
    print(Y_train.value_counts())

    # 缺失值处理
    print(X_train.isnull().any().describe())


# 输出转化后的图片
def get_image():
    reshape_begin = datetime.now()
    print(str(reshape_begin) + " reshape begin")

    # g = plt.imshow(X_train[0][:, :, 0])
    # print(X_train[0])
    # plt.show()

    for i in range(10):
        g = plt.imshow(X_train[i][:, :, 0])
        plt.show()


# CNN建模
def model_cnn():
    model_begin = datetime.now()
    print(str(model_begin) + " model begin")

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    epochs = 1  # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(X_val, Y_val),
                                  verbose=2, steps_per_epoch=X_train.shape[0] // batch_size
                                  , callbacks=[learning_rate_reduction])

    # -将近6min
    # -340s - loss: 0.4151 - acc: 0.8693 - val_loss: 0.0748 - val_acc: 0.9779
    return model


# 预测值和真实值的矩阵
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    matrix_begin = datetime.now()
    print(str(matrix_begin) + " matrix begin")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('./output/matrix.png')
    plt.show()


# 预测错误的真实图片
def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    display_errors_begin = datetime.now()
    print(str(display_errors_begin) + " display_errors begin")

    n = 0
    nrows = 3
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1

    plt.savefig('./output/errors.png')
    plt.show()


# 误差分析
def error_analyze():
    error_begin = datetime.now()
    print(str(error_begin) + " error begin")

    Y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_val, axis=1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

    # 预测值和真实值的矩阵
    plot_confusion_matrix(confusion_mtx, classes=range(10))

    errors = (Y_pred_classes - Y_true != 0)

    Y_pred_classes_errors = Y_pred_classes[errors]
    Y_pred_errors = Y_pred[errors]
    Y_true_errors = Y_true[errors]
    X_val_errors = X_val[errors]

    Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)
    true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)
    # 预测错误Top 9
    most_important_errors = sorted_dela_errors[-9:]

    display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


def predict():
    predict_begin = datetime.now()
    print(str(predict_begin) + " predict begin")

    results = model.predict(test)
    results = np.argmax(results, axis=1)
    results = pd.Series(results, name="Label")
    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
    submission.to_csv("./output/mnist_cnn.csv", index=False)


if __name__ == "__main__":
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    task_begin = datetime.now()
    print(str(task_begin) + " digit-recongizer begin")

    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)

    # 释放内存
    del train

    # 分析数据
    # analize()

    X_train = X_train / 255.0
    test = test / 255.0

    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)
    Y_train = to_categorical(Y_train, num_classes=10)

    random_seed = 2
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

    # 输出转化后的图片
    # get_image()

    # CNN建模
    model = model_cnn()

    # 误差分析
    error_analyze()

    predict()

    task_end = datetime.now()
    print(str(task_end) + " digit-recongizer end")
