import function.neural_network as nn
import mnist.open as mnist


if __name__ == "__main__":

    TrainI = mnist.readMNIST('mnist/train-images.idx3-ubyte')
    TrainImage = TrainI.getImage()
    TrainImage = mnist.image(TrainImage)
    TrainL = mnist.readMNIST('mnist/train-labels.idx1-ubyte')
    TrainLabel = TrainL.getLabel()
    TrainLabel = mnist.hot_encoding(TrainLabel)

    TestI = mnist.readMNIST('mnist/t10k-images.idx3-ubyte')
    TestImage = TestI.getImage()
    TestImage = mnist.image(TestImage)
    TestL = mnist.readMNIST('mnist/t10k-labels.idx1-ubyte')
    TestLabel = TestL.getLabel()
    TestLabel = mnist.hot_encoding(TestLabel)

    '''
    Model = [['input', FeatNum],
             ['convolution', [(5, 5), 8, 2, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['convolution', [(5, 5), 16, 0, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['dropout', 0.2],
             ['hidden', 120],
             ['dropout', 0.2],
             ['hidden', 84],
             ['dropout', 0.2],
             ['output', OutNum]]
    '''

    DeepNet, model = nn.load('model/cnn_triple_dropout_0.2_shuffle.npy')

    Result = DeepNet.predict(TestImage, TestLabel)
    print(Result[1], Result[2])

    '''
    Result = DeepNet.predict(TrainImage, TrainLabel)
    print(Result[1], Result[2])
    '''
