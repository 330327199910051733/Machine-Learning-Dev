import crypten
import torch
from crypten import mpc
import time
from model import Model, Model_squre,AlexNet
from crypten.config import cfg
import logging

cfg.communicator.verbose = True

# 初始化 CrypTen
crypten.init()
torch.set_num_threads(1)

# Prepare MNIST data
data_train = crypten.load("./data/alice_train.pth")
# data_train = crypten.load("./data/alice_train.pth")
labels_train = crypten.load("./data/alice_train_labels.pth")
data_test = crypten.load("./data/bob_test.pth")
labels_test = crypten.load("./data/bob_test_labels.pth")


# print(data_train.shape)
# data_train=data_train.view(1,1,28,28)
# print(data_train.shape)
# transform
labels = labels_train.long()
label_eye = torch.eye(10)
labels_one_hot = label_eye[labels]
labels_one_hot_test = label_eye[labels_test]

batch_size=1
# data_train=data_train.reshape(batch_size, 1, 28, 28)

# Encrypt data
images_enc = crypten.cryptensor(data_train)
labels_enc = crypten.cryptensor(labels_one_hot)

images_test_enc = crypten.cryptensor(data_test)
labels_test_enc = crypten.cryptensor(labels_one_hot_test)
# Define Model Architecture
# model = Model_squre()
# model=Model()
model=AlexNet()
# batch_size = 512
batch_size=10
dummy_input = torch.empty(batch_size, 1, 28, 28)
print(dummy_input.shape)
# dummy_input = torch.empty(batch_size, 1, 28, 28)
# load model from pytorch style yo crypten.
crypten_model = crypten.nn.from_pytorch(model, dummy_input)
crypten_model.encrypt()

# @mpc.run_multiprocess(world_size=2)
def train_model(model, train_loader, test_loader, epochs, learning_rate):
    t0 = time.time()
    model.train()
    loss = crypten.nn.MSELoss()
    num_batches = train_loader.size(0) // batch_size
    for epoch in range(epochs):
        for batch in range(num_batches):
            # define the start and end of the training mini-batch
            start, end = batch * batch_size, (batch + 1) * batch_size

            # construct CrypTensors out of training examples / labels
            x_train = train_loader[start:end, :]
            y_train = test_loader[start:end, :]
            # y_train = crypten.cryptensor(y_batch, requires_grad=True)

            x_train = x_train.reshape(batch_size, 1, 28, 28)
            # perform forward pass:
            output = model(x_train)
            loss_value = loss(output, y_train)

            # set gradients to "zero"
            model.zero_grad()

            # perform backward pass:
            loss_value.backward()

            # update parameters
            model.update_parameters(learning_rate)

            # Print progress every batch:
            batch_loss = loss_value.get_plain_text()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch + 1, batch, num_batches, 100. * batch / num_batches, batch_loss.item()))
        torch.save(model, 'model/model1_.pth')
    t1 = time.time()
    print("the train time is", t1 - t0)
    return model



@mpc.run_multiprocess(world_size=2)
def avg_test_accuracy(model, X, y):
    accuracy = 0.0
    num_batches = X.size(0) // batch_size

    for batch in range(num_batches):
        # define the start and end of the training mini-batch
        start, end = batch * batch_size, (batch + 1) * batch_size

        # construct CrypTensors out of training examples / labels
        x_train = X[start:end]
        y_train = y[start:end]

        output = model(x_train).get_plain_text().softmax(0)
        predicted = output.argmax(1)
        labels = y_train.get_plain_text().argmax(1)
        correct = (predicted == labels).sum()
        accuracy += correct
    level = logging.INFO
    logging.getLogger().setLevel(level)
    crypten.print_communication_stats()
    print(f'acc:{float(accuracy/ y.shape[0]) * 100}')


if __name__ == '__main__':
    # train model
    train_model(crypten_model, images_enc, labels_enc, epochs=3,learning_rate=0.1)

    # test model
    model = torch.load("model/model1_.pth")
    t0 = time.time()
    avg_test_accuracy(model, images_test_enc, labels_test_enc)
    t1 = time.time()
    print(f'total time is {t1-t0}')