import matplotlib.pyplot as plt

def visualization(trn_loss_avg, trn_acc_avg, tst_loss_avg, tst_acc_avg):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

    # train loss graph
    axes[0, 0].plot(trn_loss_avg[0], label='LeNet-5', c='b')
    axes[0, 0].plot(trn_loss_avg[1], label='custom', c='g')
    axes[0, 0].plot(trn_loss_avg[2], label='regularized LeNet-5', c='r')
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].legend()

    # train acc graph
    axes[0, 1].plot(trn_acc_avg[0], label='LeNet-5', c='b')
    axes[0, 1].plot(trn_acc_avg[1], label='custom', c='g')
    axes[0, 1].plot(trn_acc_avg[2], label='regularized LeNet-5', c='r')
    axes[0, 1].set_title('Train Accuracy')
    axes[0, 1].legend()

    # validation loss graph
    axes[1, 0].plot(tst_loss_avg[0], label='LeNet-5', c='b')
    axes[1, 0].plot(tst_loss_avg[1], label='custom', c='g')
    axes[1, 0].plot(tst_loss_avg[2], label='regularized LeNet-5', c='r')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].legend()

    # validation acc graph
    axes[1, 1].plot(tst_acc_avg[0], label='LeNet-5', c='b')
    axes[1, 1].plot(tst_acc_avg[1], label='custom', c='g')
    axes[1, 1].plot(tst_acc_avg[2], label='regularized LeNet-5', c='r')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()

    plt.savefig('result.png')

    plt.show()

# trn_loss_avg, trn_acc_avg, tst_loss_avg, tst_acc_avg는 main.py에 있는 데이터를 사용
visualization(trn_loss_avg, trn_acc_avg, tst_loss_avg, tst_acc_avg)
