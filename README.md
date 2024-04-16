# MNIST_classification_renet5 (인공신경망과 딥러닝)

Notice
- train.tar의 용량이 커서 data폴더에는 test.tar과 동일한 파일로 대체하여 업로드 하였습니다.
- train.tar(실제로는 test.tar)은 모델이 돌아가는 것을 확인하는 용으로 올려놓았으며, 실제 성능과 차이가 있을 수 있습니다.
- assignment report : https://ubiquitous-sparrow-801.notion.site/Assignment_Report-102dfe486ffc403e80cd12692cfc98fc?pvs=4
- 혹시 몰라 pdf도 첨부하였지만, 해당 링크로 들어가주시면 감사하겠습니다.
  
--------------------------------------------------------------------------------------
## 파일 설명

```main.py``` : 모델 학습 & 테스트 전체 <br/>
```dataset.py``` : MNIST 데이터셋 전처리 <br/>
```model.py``` : LeNet-5, Custom MLP, Regularized LeNet-5 세가지 모델 구현 <br/>
```plot_visualization.py``` : main 결과값 plot 시각화 <br/>

--------------------------------------------------------------------------------------

## 과제 설명
In this assignment, you build a neural network classifier with MNIST dataset. For a detailed description about MNIST dataset, please refer to [this link](http://yann.lecun.com/exdb/mnist/).

- Requirements
    1. You should write your own pipeline to provide data to your model. Write your code in the template `dataset.py`. Please read the comments carefully and follow those instructions.
    2. (Report) Implement LeNet-5 and your custom MLP models in `model.py`. Some instructions are given in the file as comments. Note that your custom MLP model should have about the same number of model parameters with LeNet-5. Describe the number of model parameters of LeNet-5 and your custom MLP and how to compute them in your report.
    3. Write `main.py` to train your models, LeNet-5 and custom MLP. Here, you should monitor the training process. To do so, you need some statistics such as average loss values and accuracy at the end of each epoch.
    4. (Report) Plot above statistics, average loss value and accuracy, for training and testing. It is fine to use the test dataset as a validation dataset. Therefore, you will have four plots for each model: loss and accuracy curves for training and test datasets, respectively.
    5. (Report) Compare the predictive performances of LeNet-5 and your custom MLP. Also, make sure that the accuracy of LeNet-5 (your implementation) is similar to the known accuracy. 
    6. (Report) Employ at least more than two regularization techniques to improve LeNet-5 model. You can use whatever techniques if you think they may be helpful to improve the performance. Verify that they actually help improve the performance. Keep in mind that when you employ the data augmentation technique, it should be applied only to training data. So, the modification of provided `MNIST` class in `dataset.py` may be needed.
- **Note that the details of training configuration which are not mentioned in this document and the comments can be defined yourself.** For example, decide how many epochs you will train the model.

