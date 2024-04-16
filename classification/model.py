import torch.nn as nn

# Difine LeNet5 class
class LeNet5(nn.Module):

    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """
        
    def __init__(self):
        super().__init__()

        # 파라미터 수 = (커널 높이×커널 너비×입력 채널 수+1(bias))×출력 채널 수
        # (5*5*1+1)*6 = 156
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # (5*5*6+1)*16 = 2416
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.act = nn.Tanh()

        # 학습 파라미터 x
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 파라미터 수 = (입력 특성의 수+1(bias))×출력 특성의 수
        # 입력 특성의 수 : 이전 레이어의 출력 size

        # (16×5×5+1)×120 = 48120
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        # (120+1)×84 = 10164
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # (84+1)×10 = 850
        self.fc3 = nn.Linear(in_features=84, out_features=10)

        self.softmax = nn.Softmax(dim=1)

        # Total parameters = 61,706
        # Forward/backward parameters = 123,412

    def forward(self, img):

        x = self.act(self.conv1(img))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        output = self.softmax(self.fc3(x))

        return output
    

class CustomMLP(nn.Module):

    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
        with LeNet-5
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            # (28*28+1)*64 = 50240
            nn.Linear(28 * 28, 64),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            # (64+1)*64 = 4160
            nn.Linear(64, 64),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            # (64+1)*64 = 4160
            nn.Linear(64, 64),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            # (64+1)*32 = 2080
            nn.Linear(64, 32),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            # (32+1)*16 = 528
            nn.Linear(32, 16),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            # (16+1)*16 = 272
            nn.Linear(16, 16),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            # (16+1)*10 = 170
            nn.Linear(16, 10),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer8 = nn.Sequential(
            # (10+1)*10 = 110
            nn.Linear(10, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        output = self.layer8(x)

        return output
    
    # Total parameters = 61,720
    # Forward/backward parameters = 123,440


# regularization을 추가한 LeNet
class LeNet5_regularization(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),  # BatchNorm 첫번째 정규화 기법 추가
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16), 
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)  # dropout 두번째 정규화 기법 추가
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.BatchNorm1d(120), 
            nn.Tanh(),
            nn.Dropout(0.5), 
            nn.Linear(in_features=120, out_features=84),
            nn.BatchNorm1d(84), 
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        x = self.conv_layers(img)
        x = x.view(-1, 16 * 5 * 5)
        output = self.fc_layers(x)
        return output
    

    # Total parameters = 62,158
    # Forward/backward parameters = 124,316