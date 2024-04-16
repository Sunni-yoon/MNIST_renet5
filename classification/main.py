import dataset
from model import LeNet5, CustomMLP, LeNet5_regularization
import torch
from torchvision import transforms
import time
from torchsummary import summary


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    model.to(device)
    trn_loss_sum = 0
    acc_sum = 0
    trn_loss_list = []
    trn_acc_list = []

    for i, (data, target) in enumerate(trn_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        # Forward pass
        y_hat = model(data)
        trn_loss = criterion(y_hat, target)
        trn_loss_sum += trn_loss.item()
        trn_loss_list.append(float(trn_loss))

        # Backward pass
        trn_loss.backward()
        optimizer.step()

        # Accuracy
        output_label = torch.argmax(y_hat, dim=1)
        cor_pred = (output_label == target)

        acc = cor_pred.sum() / len(cor_pred)
        acc_sum += acc * len(cor_pred)
        trn_acc_list.append(float(acc))

    trn_loss = trn_loss_sum / len(trn_loader.dataset)
    acc = acc_sum / len(trn_loader.dataset)

    print(f'\nTrain set: Average loss: {trn_loss}, Train Accuracy: {acc_sum}/{len(trn_loader.dataset)} ({acc*100}%)\n')
    print('Finished Training Trainset')

    return trn_loss, trn_loss_list, acc, trn_acc_list


def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()
    tst_loss_sum = 0
    acc_sum = 0
    tst_loss_list = []
    tst_acc_list = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tst_loader):
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            y_hat = model(data)
            tst_loss = criterion(y_hat, target)
            tst_loss_sum += tst_loss
            tst_loss_list.append(tst_loss)
            output_tags = torch.argmax(y_hat, 1)

            # Accuracy
            cor_pred = (output_tags == target)
            acc = cor_pred.sum() / len(cor_pred)
            acc_sum += acc * len(cor_pred)
            tst_acc_list.append(acc_sum)

    tst_loss = tst_loss_sum / len(tst_loader.dataset)
    acc = acc_sum / len(tst_loader.dataset)

    print(f'\nTest set: Average loss: {tst_loss}, Test Accuracy: {acc_sum}/{len(tst_loader.dataset)} ({acc*100}%)\n')
    print('Finished Testing Test set')

    return tst_loss, tst_loss_list, acc, tst_acc_list


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 64
    learning_rate = 0.01
    mommentum = 0.9

    # 1)
    train_data = dataset.MNIST(data_dir='data/train.tar', test=False)
    test_data = dataset.MNIST(data_dir='data/test.tar', test=True)

    # 2)
    trn_loader = dataset.DataLoader(train_data, batch_size=batch_size)
    tst_loader = dataset.DataLoader(test_data, batch_size=batch_size)

    """
    LeNet-5 model
    """
    
    lenet_model = LeNet5().to(device)
    optimizer = torch.optim.SGD(lenet_model.parameters(), lr=learning_rate, momentum=mommentum)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    lr_time = time.time()
    le_trn_acc_avg, le_trn_loss_avg, le_tst_acc_avg, le_tst_loss_avg = [], [], [], []

    for epoch in range(epochs):
        le_trn_loss, le_trn_loss_list, le_trn_acc, le_trn_acc_list = train(model=lenet_model,
                                                                           trn_loader=trn_loader,
                                                                           device=device,
                                                                           criterion=criterion,
                                                                           optimizer=optimizer)
        le_tst_loss, le_tst_loss_list, le_tst_acc, le_tst_acc_list = test(model=lenet_model,
                                                                          tst_loader=tst_loader,
                                                                          device=device,
                                                                          criterion=criterion)
        le_trn_acc_avg.append(le_trn_acc)
        le_trn_loss_avg.append(le_trn_loss)
        le_tst_acc_avg.append(le_tst_acc)
        le_tst_loss_avg.append(le_tst_loss)

        print(
            f'\nLeNet-5 Model\n'
            f'{epoch + 1} epochs\n'
            f'training loss: {le_trn_loss}\n'
            f'training accuracy: {le_trn_acc}\n'
            f'validation loss: {le_tst_loss}\n'
            f'validation accuracy: {le_tst_acc}\n'
        )

        if epoch + 1 == 10:
            print(f'LeNet-5 model execution time : {time.time() - lr_time}')

    # model summary 출력
    print("LeNet-5 Model Summary")
    summary(lenet_model.to(device), (1, 28, 28))

    print(le_trn_acc_avg)
    print(le_trn_loss_avg)
    print(le_tst_acc_avg)
    print(le_tst_loss_avg)

    """
    Custom model
    """
    
    custom_model = CustomMLP().to(device)
    optimizer = torch.optim.SGD(custom_model.parameters(), lr=learning_rate, momentum=mommentum)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    lr_time = time.time()
    cu_trn_acc_avg, cu_trn_loss_avg, cu_tst_acc_avg, cu_tst_loss_avg = [], [], [], []
    for epoch in range(epochs):
        cu_trn_loss, cu_trn_loss_list, cu_trn_acc, cu_trn_acc_list = train(model=custom_model,
                                                                           trn_loader=trn_loader,
                                                                           device=device,
                                                                           criterion=criterion,
                                                                           optimizer=optimizer)
        cu_tst_loss, cu_tst_loss_list, cu_tst_acc, cu_tst_acc_list = test(model=custom_model,
                                                                          tst_loader=tst_loader,
                                                                          device=device,
                                                                          criterion=criterion)
        cu_trn_acc_avg.append(cu_trn_acc)
        cu_trn_loss_avg.append(cu_trn_loss)
        cu_tst_acc_avg.append(cu_tst_acc)
        cu_tst_loss_avg.append(cu_tst_loss)
        print(
            f'\nCustom Model\n'
            f'{epoch + 1} epochs\n'
            f'training loss: {cu_trn_loss}\n'
            f'training accuracy: {cu_trn_acc}\n'
            f'validation loss: {cu_tst_loss}\n'
            f'validation accuracy: {cu_tst_acc}\n'
        )
        if epoch + 1 == 10:
            print(f'Custom model execution time : {time.time() - lr_time}')

    # model summary 출력
    print("Custom Model Summary")
    summary(custom_model.to(device), (1, 28, 28))

    print(cu_trn_acc_avg)
    print(cu_trn_loss_avg)
    print(cu_tst_acc_avg)
    print(cu_tst_loss_avg)

    """
    Regularized LeNet5 model
    """
    
    regularized_lenet_model = LeNet5_regularization().to(device)
    optimizer = torch.optim.SGD(regularized_lenet_model.parameters(), lr=learning_rate, momentum=mommentum, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    lr_time = time.time()
    re_trn_acc_avg, re_trn_loss_avg, re_tst_acc_avg, re_tst_loss_avg = [], [], [], []
    for epoch in range(epochs):
        re_trn_loss, re_trn_loss_list, re_trn_acc, re_trn_acc_list = train(model=regularized_lenet_model,
                                                                           trn_loader=trn_loader,
                                                                           device=device,
                                                                           criterion=criterion,
                                                                           optimizer=optimizer)
        re_tst_loss, re_tst_loss_list, re_tst_acc, re_tst_acc_list = test(model=regularized_lenet_model,
                                                                          tst_loader=tst_loader,
                                                                          device=device,
                                                                          criterion=criterion)
        re_trn_acc_avg.append(re_trn_acc)
        re_trn_loss_avg.append(re_trn_loss)
        re_tst_acc_avg.append(re_tst_acc)
        re_tst_loss_avg.append(re_tst_loss)

        print(
            f'\Regularized Lenet Model\n'
            f'{epoch + 1} epochs\n'
            f'training loss: {re_trn_loss}\n'
            f'training accuracy: {re_trn_acc}\n'
            f'validation loss: {re_tst_loss}\n'
            f'validation accuracy: {re_tst_acc}\n'
        )
        if epoch + 1 == 10:
            print(f'Regularized LeNet-5 model execution time : {time.time() - lr_time}')

    # model summary 출력
    print("Regularized LeNet-5 Model Summary")
    summary(regularized_lenet_model.to(device), (1, 28, 28))

    print(re_trn_acc_avg)
    print(re_trn_loss_avg)
    print(re_tst_acc_avg)
    print(re_tst_loss_avg)

    trn_loss_avg = [le_trn_loss_avg, cu_trn_loss_avg, re_trn_loss_avg]
    trn_acc_avg = [le_trn_acc_avg, cu_trn_acc_avg, re_trn_acc_avg]
    tst_loss_avg = [le_tst_loss_avg, cu_tst_loss_avg, re_tst_loss_avg]
    tst_acc_avg = [le_tst_acc_avg, cu_tst_acc_avg, re_tst_acc_avg]

    print(trn_loss_avg)
    print(trn_acc_avg) 

    print(tst_loss_avg)
    print(tst_acc_avg)
