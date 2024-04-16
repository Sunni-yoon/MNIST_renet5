import tarfile
from PIL import Image
import os
import torch
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings


# 데이터가 잘 나오는지/어떻게 생겼는지 이미지로 확인
def extract_and_show_image_from_tar_by_index(tar_path):
    with tarfile.open(tar_path, 'r') as tar:
        image_filename = tar.getnames()[3] # 원하는 인덱스 찍어보기
        img_data = tar.extractfile(image_filename).read()
        img = Image.open(io.BytesIO(img_data))
        img.show()

tar_path = 'data/train.tar'
extract_and_show_image_from_tar_by_index(tar_path)


# ingore
warnings.filterwarnings("ignore")


transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.1307], [0.3081])])

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
                - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png

        
    """

    def __init__(self, data_dir, test=False):

        self.data_dir = data_dir
        self.file_open = tarfile.open(data_dir, 'r')
        self.img_data = self.file_open.getnames()[1:]
        self.labels = [int(os.path.basename(i)[-5]) for i in self.img_data]
        self.numbers = [os.path.basename(i)[:5] for i in self.img_data]
        self.test = test

                # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])

    def __len__(self):

        return len(self.numbers)

    def __getitem__(self, idx):

        str_idx = str(idx)

        if len(str(idx)) != 5:
            str_idx = '0' * (5 - len(str_idx)) + str_idx

        label = torch.tensor(self.labels[self.numbers.index(str_idx)])

        if self.test == False:
            folder = 'train'
        else:
            folder = 'test'

        extract_file = self.file_open.extractfile(f'{folder}/{str_idx}_{label}.png').read()
        open_image = Image.open(io.BytesIO(extract_file))
        img = transform(open_image)

        return img, label


if __name__ == '__main__':

    train = MNIST(data_dir='data/train.tar', test=False)
    test = MNIST(data_dir='data/test.tar', test=True)

    trn_loader = DataLoader(train, batch_size=64)
    tst_loader = DataLoader(test, batch_size=64)

    print(test.__getitem__(0)[1])
    # tensor(6)
    print(train.__getitem__(0)[1])
    # tensor(5)

    # print dataloader size
    print(next(iter(trn_loader))[0].size())
    # torch.Size([64, 1, 28, 28])
    # batch size, channel, height, widths