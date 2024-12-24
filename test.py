import function.neural_network as nn
import numpy as np
import importlib
import mnist.open as mnist
import time


if __name__ == "__main__":

    Act = 'LReLU'
    Out = 'softmax'

    TrainI = mnist.readmniIST('mnist/train-images.idx3-ubyte')
    TrainImage = TrainI.getImage()
    TrainImage = mnist.image(TrainImage)
    TrainL = mnist.readmniIST('mnist/train-labels.idx1-ubyte')
    TrainLabel = TrainL.getLabel()
    TrainLabel = mnist.hot_encoding(TrainLabel)

    TestI = mnist.readmniIST('mnist/t10k-images.idx3-ubyte')
    TestImage = TestI.getImage()
    TestImage = mnist.image(TestImage)
    TestL = mnist.readmniIST('mnist/t10k-labels.idx1-ubyte')
    TestLabel = TestL.getLabel()
    TestLabel = mnist.hot_encoding(TestLabel)

    Path = 'model/cnn_double_dropout_0.2_shuffle.npy'
    DeepNet = nn.load(Path)
    DeepNet.method(shuffle=False)

    try:

        Para = input('leanring rate and epochs\n').split()
        Rate = float(Para[0])
        Epoch = int(Para[1])

        while 1:

            TrainError = DeepNet.cnn_train(TrainImage, TrainLabel, Rate, Epoch)
            time1 = time.time()
            Result = DeepNet.predict(TestImage, TestLabel)
            time2 = time.time()
            print(Result[1], Result[2])
            print(Rate, Epoch)
            print(time2 - time1)

            nn.save(DeepNet, Path)

            Argv = input('want to end?\n')
            if Argv == '1':
                break
            else:
                Para = Argv.split()
                Rate = float(Para[0])
                Epoch = int(Para[1])

    except KeyboardInterrupt:
        nn.save(DeepNet, Path)
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])
