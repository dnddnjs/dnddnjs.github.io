---
layout: post
title: "CIFAR-10 정복 시리즈 2: PyramidNet"
subtitle: "Deep Pyramidal Residual Networks"
categories: cifar10
tags: dl
comments: true
---

## CIFAR-10 정복하기 시리즈 소개
CIFAR-10 정복하기 시리즈에서는 딥러닝이 CIFAR-10 데이터셋에서 어떻게 성능을 높여왔는지 그 흐름을 알아본다. 또한 코드를 통해서 동작원리를 자세하게 깨닫고 실습해볼 것이다. 

- CIFAR-10 정복하기 시리즈 목차(클릭해서 바로 이동하기)
  - [CIFAR-10 정복 시리즈 0: 시작하기](https://dnddnjs.github.io/cifar10/2018/10/07/start_cifar10/)
  - [CIFAR-10 정복 시리즈 1: ResNet](https://dnddnjs.github.io/cifar10/2018/10/09/resnet/)
  - [CIFAR-10 정복 시리즈 2: PyramidNet](https://dnddnjs.github.io/cifar10/2018/10/24/pyramidnet/)
  - [CIFAR-10 정복 시리즈 3: Shake-Shake](https://dnddnjs.github.io/cifar10/2018/10/25/shake_shake/)

- 관련 코드 링크
  - [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 

<br>

## CIFAR-10 정복 시리즈 2: PyramidNet
이전 포스트에서는 ResNet에 대해서 알아봤다. ResNet이 등장한 2015년 이후로 ResNet 기반의 새로운 네트워크가 많이 나왔다. 이번 포스트에서는 ResNet 이후 모델 구조에 관한 논문 중에 대표적인 3개 논문을 살펴볼 것이다. **WideResNet**[^0]은 ResNet의 깊이가 깊어지면 학습이 어렵다는 점을 해결하고자 깊게 쌓는 것이 아니라 넓게 쌓는 방법을 제안했다. **DenseNet**[^1]은 ResNet의 shortcut connection을 주의깊게 보고 더 dense한 connection 방법을 제안했다. **PyramidNet**[^2]은 ResNet의 down sampling 부분에 주목했다. ResNet에서 donw sampling을 특정 layer에서 수행하는 것이 아니라 모든 layer에서 나눠서 하는 것이 PyramidNet의 아이디어이다. 자 이제 하나씩 살펴보자.

1. [WideResNet](#wideresnet)
2. [DenseNet](#densenet)
3. [PyramidNet](#pyramidnet)

<br>

---

## WideResNet

ResNet은 1000 layer 이상의 네트워크도 학습이 되도록 했다. 하지만 100 layer 대의 ResNet 보다 1000 layer 이상의 네트워크가 오히려 성능이 안좋아지는 문제가 생겼다. 물론 이전 포스트에서 Kaiming He의 후속 논문인 **Identity Mappings in Deep Residual Networks**[^3]에서 이 문제를 어느정도 해결했다는 것을 봤다. Wide ResNet은 이와 같은 activation 순서에 대한 연구를 통해 네트워크의 **depth**를 늘리는 것이 아닌 다른 방법을 제안하고 있다. 네트워크의 **width**를 키움으로서 네트워크의 성능을 올리는 것이다. 

Wide ResNet 저자가 이러한 생각을 한 이유는 identity mapping에 대한 생각 때문이다. 논문에서는 Identity mapping이 ResNet의 강점인 동시에 약점이라고 주장한다. Identity mapping 즉 shortcut connection은 아무 파라메터가 없다. 따라서 gradient가 shortcut connection을 따라서 흐를 경우 네트워크는 아무것도 배우지 않을 가능성이 있다. 그 결과 깊은 ResNet의 layer 들 중에서 일부만 유용한 representation을 학습하거나 여러 layer들이 정보를 조금씩 나눠서 가지고 있을 수 있다. 

이것은 마치 fully-connected layer에서 생기는 co-adaptation 문제와 유사하다. Fully-connected layer에서는 이 문제를 해결하기 위해 **dropout**[^4]을 사용한다. ResNet에서도 따라서 일부 layer에 유용한 respresentation이 몰려있는 현상이 발생하기 때문에 **ResDrop**[^5]과 같은 논문이 나왔다. Dropout이 랜덤하게 특정 뉴런을 사용안했다면 ResDrop은 특정 residual block을 랜덤하게 사용안한다. 다음 그림은 ResDrop의 작동방식을 보여준다. 네트워크의 뒤로 갈수록 50%에 가까운 확률로 residual learning을 하는 weight를 비활성화한다. **Stochastic depth**를 사용하면 CIFAR-10 데이터에서 1% 이상의 accuracy가 오른다. 따라서 깊은 네트워크의 layer 중에 일부만 유용한 representation을 학습했다는 주장을 하는 것이다. 따라서 단순히 깊이를 늘리는 것은 학습 성능을 올리는데 좋은 방법이 아닐 수 있다. 

<figure>
  <img src="https://www.dropbox.com/s/k3ac39dapahvasa/Screenshot%202018-11-20%2023.02.25.png?dl=1" width="500px">
  <figcaption>
    https://arxiv.org/pdf/1603.09382.pdf
  </figcaption>
</figure>

<br>
ResNet은 네트워크를 깊게 쌓기 위해 residual block의 convolution filter의 개수를 최소화했다. 심지어 bottleneck block을 사용함으로서 더 적은 filter를 사용하였다. Wide ResNet에서 말하는 width는 이 convolution filter 개수를 의미한다. Wide ResNet은 깊이를 늘리는 것에는 관심이 없기 때문에 residual block의 convolution filter 수를 늘렸다. 기존의 residual block이 아래 그림의 제일 왼쪽이라면 논문에서 제안하는 wide residual block은 오른쪽 두 개와 같다. 


<figure>
  <img src="https://www.dropbox.com/s/f5dwsef7crx97f7/Screenshot%202018-10-12%2018.46.05.png?dl=1">
  <figcaption>
    https://arxiv.org/pdf/1605.07146.pdf
  </figcaption>
</figure>

<br>

CIFAR 데이터셋에 적용된 ResNet의 경우 입력 이미지 사이즈가 32 --> 16 --> 8으로 줄면서 width는 16 --> 32 --> 64로 늘어난다. Wide residual block에서는 이 width를 정수배로 늘린다. WideResNet은 16부터 40의 깊이를 가진다. 깊이가 얕을수록 residual block의 width는 커진다. 

<figure>
  <img src="https://www.dropbox.com/s/7h5whxvdthu8y18/Screenshot%202018-10-12%2018.15.28.png?dl=1">
  <figcaption>
    https://arxiv.org/pdf/1605.07146.pdf
  </figcaption>
</figure>

<br>

WideResNet 논문에서 n=40, 28, 22, 16 그리고 k = 1, 2, 4, 8, 10, 12에 대해 실험했다. 그 결과는 다음과 같다. CIFAR10에 대해서는 n=28 / k=10이 제일 좋은 정확도를 가진다. n이 28이면 conv2, conv3, conv4의 block의 수는 (28-4) / 6 = 4이다. conv2의 width는 16x10 = 160이고 conv3의 width는 32x10인 320이다. conv4의 width는 64x10인 640이다. WideResNet-28-10의 경우 CIFAR-10에서 4.17%의 error rate를 가진다. 이 때 파라메터의 수는 36.5M인데 ResNet-1001이 10.2M인 것과 비교하면 3배 이상 parameter가 많다. ResNet-1001과 비슷한 파라메터 수를 가지는 WideResNet-16-8과 비교했을 때 ResNet-1001과 거의 동일한 성능을 가진다. ResNet-1001의 CIFAR-10에서의 error rate는 4.92%이고 WideResNet-16-8은 4.81%이다. 또한 기존 ResNet-1001보다 WideResNet-28-10이 파라메터 수가 3배 이상 많음에도 accuracy가 더 높은 것을 보아 WideResNet이 기존 ResNet보다 학습이 잘 되는 것을 확인할 수 있다. 

<figure>
   <img src="https://www.dropbox.com/s/8kaggunevtssp8r/Screenshot%202018-10-12%2021.02.35.png?dl=1">
  <figcaption>
    https://arxiv.org/pdf/1605.07146.pdf
  </figcaption>
</figure>

<br>

여기서 좀 더 나아가 WideResNet에 dropout을 적용하면 State-of-the-art를 달성할 수 있었다. WideResNet의 경우 width를 넓히면서 residual block의 parameter 수가 많아졌다. 따라서 추가적인 regularization이 필요할 수 있는데 실제로 dropout을 convolutional layer 사이에 넣으면 성능이 향상된다. 위에서 4.17% error를 가지는 WideResNet-28-10에 dropout을 적용하면 3.89%의 error rate를 달성할 수 있다. Dropout rate는 0.3을 사용하였다. 

<figure>
   <img src="https://www.dropbox.com/s/bugro9g0uxbx40t/Screenshot%202018-10-12%2021.13.13.png?dl=1">
  <figcaption>
    https://arxiv.org/pdf/1605.07146.pdf
  </figcaption>
</figure>

<br>
간단히 wide residual block의 forward pass를 보면 다음과 같다. 기억해야할 것은 WideResNet에서는 Pre-activation residual block의 구조를 사용한다. 따라서 batch normalization + relu + conv의 순서로 연산을 진행한다. 첫번째 conv 이후에 dropout을 사용한다. 이제 DenseNet을 살펴보자.

~~~python
def forward(self, x):
    out = self.dropout(self.conv1(F.relu(self.bn1(x))))
    out = self.conv2(F.relu(self.bn2(out)))
    out += self.shortcut(x)

    return out
~~~


<br>

---

## DenseNet

WideResNet이 네트워크의 width에 초점을 맞췄다면 DenseNet은 ResNet의 **Shortcut connection**에 초점을 맞췄다. ResNet과 Highway network, ResDrop 모두 성능을 높이기 위해 앞쪽 layer에서 뒤쪽 layer로 가는 **shortcut connection**을 사용했다. 이 때 바로 전 layer의 출력만을 다음 layer로 보냈지만 DenseNet은 **이전의 많은 layer의 출력을 한꺼번에 받는다**. 상당히 간단한 아이디어로 다음 그림을 통해 이해할 수 있다. 다음 그림은 ResNet에서 conv1, conv2, conv3에 해당하는 DenseNet의 DenseBlock을 그림으로 그린 것이다. 기존의 Residual block이 $$x_l = H_l(x_{l-1}) + x_{l-1}$$이라고 한다면 DenseBlock에서 한 layer는 $$x_l = H_l([x_0, x_1, .... , x_{l-1}])$$ 이라고 표현할 수 있다. Residual block과의 차이점은 DenseBlock 내부에서 이전 layer의 입력을 모두 받는다는 것과 addition이 아닌 concatenation으로 입력을 합친다는 것이다. 

<figure>
  <img src="https://www.dropbox.com/s/qlw9b3vad5osrqa/Screenshot%202018-10-11%2016.16.10.png?dl=1" width="500px">
  <figcaption>
    https://arxiv.org/pdf/1608.06993.pdf
  </figcaption>
</figure>

<br>
DenseNet의 전체 구조는 다음과 같다. 입력이 들어오면 convolution을 하나 거친 이후에 DenseBlock을 통과한다. Dense Block에서는 이전 Residual block과는 달리 down sampling이 일어나지 않는다. 대신 **transition layer**를 두어서 여기서 down sampling을 한다. Down sampling으로는 average pooling을 사용한다. Transition layer는 BN + conv1x1 + average pooling으로 구성되어있다. DenseBlock 내부에서는 pre-activation을 사용한다. 즉 layer 하나의 연산은 bn-relu-conv3x3-bn-relu-conv3x3가 된다. 

<figure>
  <img src="https://www.dropbox.com/s/3y5idt67bea7jid/Screenshot%202018-10-11%2017.00.25.png?dl=1">
  <figcaption>
    https://arxiv.org/pdf/1608.06993.pdf
  </figcaption>
</figure>

<br>
DenseNet에서는 한 가지 유의해야할 점이 있다. ResNet에서는 이전 입력과 합칠 때 addition을 사용하기 때문에 feature map의 channel 수는 변하지 않는다. 하지만 DenseNet에서는 이전 입력과 모두 **concatenation** 을 하기 때문에 feature map의 channel 수가 점점 늘어난다. 만일 DenseBlock 안에서 l번째 layer라고 한다면 이 layer의 입력은 $$k_0 + k \times (l-1)$$의 channel 수를 가진다. $$k_0$$는 DenseBlock의 입력 channel을 의미하고 k는 각 convolution의 channel을 의미한다. Layer의 입력으로는 이전 입력이 모두 concatenated되서 들어오기 때문에 channel 수가 계속 달라지지만 layer 내부에서는 k의 channel로 유지되는 것이다. DenseNet의 특성 상 layer의 width는 상당히 작게 유지될 수 있다. 논문에서는 12, 24, 40 정도를 k 값으로 사용한다. 

CIFAR에서 DenseNet은 3개의 Dense block을 사용한다. 그 이외에는 기존 ResNet 구조와 거의 동일하다. DenseNet의 성능은 다음과 같다. 표에서 C10+는 augmentation 한 것을 의미한다. DenseNet (k=12)를 보면 네트워크의 깊이는 40이며 파라메터 수는 1.0M이다. Error rate는 5.24 %인데 동일한 성능의 WideResNet과 비교하면 2배정도 파라메터가 적다. DenseNet 저자는 이렇게 성능이 향상된 이유로 **Deep supervision**의 가능성을 제시했다. 이전 layer의 출력이 뒤로 모두 전달이 되는데 결국 loss function이 모든 layer에 적용될 수 있다. 이는 각 layer마다 supervision의 영향력을 직접적으로 걸어주는 역할을 한다고 해석할 수 있다. 

<figure>
  <img src="https://www.dropbox.com/s/05baxltjfhiqqux/Screenshot%202018-10-11%2017.16.51.png?dl=1" width="500px">
  <figcaption>
    https://arxiv.org/pdf/1608.06993.pdf
  </figcaption>
</figure>


<br>

---

## PyramidNet

PyramidNet은 ResNet의 down sampling에서 일어나는 급격한 width의 변화에 초점을 맞췄다. 일반적인 ResNet 구조의 네트워크들은 feature map size를 반으로 줄이면서 feature map channels는 2배로 늘린다. PyramidNet은 모든 layer에서 channel 수가 변하도록 해서 특정 layer에 집중되어있던 width의 변화를 전체 네트워크로 분산시켰다. 이러한 생각을 하게 된 것은 ResDrop의 연구결과 때문이다. 

ResNet은 **"Residual networks behave like ensembles of relatively shallow networks"**[^6] 논문에서 언급한 것처럼 일종의 얕은 네트워크들의 앙상블처럼 행동한다. 따라서 ResDrop에서 특정 layer들을 없애도 전체 성능에 영향이 별로 없던 것이다. 하지만 down sample이 일어나는 layer를 없앴을 경우 다른 layer에 비해 큰 폭으로 성능 저하가 일어났다. 다음 그림이 ResNet의 특정 layer를 없앨 경우 성능이 어떻게 변하는지를 보여준다. 파란색 수직선이 down sample이 일어나는 layer이다. 왼쪽이 Pre-activation ResNet인데 down sample이 일어날 때 2% 정도 성능이 저하되는 것을 볼 수 있다. 오른쪽이 PyramidNet에서 같은 실험을 한 것인데 모든 layer에서 성능 저하가 거의 동일한 것을 볼 수 있다. 

<figure>
  <img src="https://www.dropbox.com/s/mjlw97g7e1cte5i/Screenshot%202018-11-21%2023.33.55.png?dl=1">
  <figcaption>
      https://arxiv.org/pdf/1610.02915.pdf
  </figcaption>
</figure>

<br>

PyramidNet에서 residual block은 다음 그림에서 (d)나 (e)와 같다. 기존에는 convolution filter의 수가 down sample이 아닌 곳에서는 모두 같았다면 PyramidNet에서는 모든 residual block에서 convolution filter 수가 증가한다.

<figure>
  <img src="https://www.dropbox.com/s/fm7yui43ojdt5rt/Screenshot%202018-10-24%2015.36.15.png?dl=1"> 
  <figcaption>
      https://arxiv.org/pdf/1610.02915.pdf
  </figcaption>
</figure>

<br>

PyramidNet에서 layer의 width를 늘리는 방법에는 두 가지가 있다. (1) additive 방식으로 전체 layer 동안 얼마나 width를 늘릴지 $$\alpha$$를 정한 다음에 이전 width에 비해서 $$\alpha / N$$만큼 늘리는 것이다. (2) multiplicative 방식으로 지수배로 늘리는 방식이다. 다음 그림에서 가운데가 multiplicative 방식이며 오른쪽이 additive 방식이다. 성능은 additive 방식이 좋은 편인데 초기 layer 들의 width가 multiplicative 방식보다 더 큰 경향이 있기 때문이다. 

<figure>
  <img src="https://www.dropbox.com/s/ehjxn6g7lyjhxd9/Screenshot%202018-11-21%2023.45.57.png?dl=1">
  <figcaption>
      https://arxiv.org/pdf/1610.02915.pdf
  </figcaption>
</figure>

<br>

PyramidNet에서는 residual block 내부 구조가 Pre-activation ResNet과 좀 다르다. Pre-activation ResNet의 block 내부 구조 자체도 개선의 여지가 있다고 생각했기 때문에 논문에서 여러 구조로 실험을 해봤다. Residual block 안에 ReLU가 너무 많으면 성능이 안좋아지는 경향이 있다. 따라서 residual block 안에서 첫번째 ReLU를 생략한다. 다음 그림의 (b)와 (d)에 해당에 해당한다. 하지만 첫번째 ReLU를 생략하면 두 개의 convolution 사이에 non-linearity가 없어서 representation power가 약해진다. 따라서 Batch Normalization layer를 하나 더 추가해주는 것이 좋다. 따라서 PyramidNet에서는 (d)를 Residual block의 구조로 사용한다. 

<figure>
  <img src="https://www.dropbox.com/s/lv6lvozm1uzgm4h/Screenshot%202018-10-24%2021.15.39.png?dl=1">
  <figcaption>
      https://arxiv.org/pdf/1610.02915.pdf
  </figcaption>
</figure>

<br>

모든 layer에서 width의 변화가 있기 때문에 shortcut connection도 단순히 identity mapping이 될 수 없다. Width가 변하는 경우는 1x1 convolution을 쓰거나 zero-padding을 써야한다. 모든 layer에 1x1 convolution을 shortcut connection으로 사용하면 parameter의 수 증가의 문제도 있고 적용했을 때 결과가 별로 좋지 않다. 따라서 PyramidNet에서는 zero-padding 방법을 사용한다. CIFAR 데이터셋에 적용했던 ResNet 또한 down sampling이 일어날 때 zero-padding을 사용했던 것을 생각하면 동일한 방법임을 알 수 있다. 다음 표는 여러가지 shortcut connection 방식을 비교한 것이다. Projection이 1x1 convolution을 의미하는데 모두 projection을 한 것보다 zero-padding만 한 것이 2% 이상 성능이 좋다. 

<figure>
  <img src="https://www.dropbox.com/s/j5evvkrjmonyfcl/Screenshot%202018-11-21%2023.58.57.png?dl=1">
  <figcaption>
      https://arxiv.org/pdf/1610.02915.pdf
  </figcaption>
</figure>

<br/>

다음은 PyramidNet의 학습 결과표이다. PyramidNet은 CIFAR-10에서 최대 3.31 %의 error rate를 달성한 것을 볼 수 있다. PyramidNet의 학습은 ResNet 학습과 거의 동일하다. Augmentation은 동일하고 SGD with momentum으로 네트워크를 업데이트했다. SGD의 처음 learning rate는 0.1로 시작해서 150 epoch에서 0.01로 225 epoch에서 0.001로 감소시켰다. 

<figure>
  <img src="https://www.dropbox.com/s/3y88bc0n16mmisf/Screenshot%202018-10-24%2021.24.21.png?dl=1">
  <figcaption>
      https://arxiv.org/pdf/1610.02915.pdf
  </figcaption>
</figure>

<br>

간단히 PyramidNet의 모델 구조를 코드로 살펴보자. 코드는 https://github.com/dnddnjs/pytorch-cifar10/tree/pyramid/pyramidnet 에 있다. PyramidNet의 residual block은 다음과 같다. 위에서 살펴봤듯이 기존 pre-actiovation residual block과는 다르게 relu가 하나밖에 없으며 대신 batch normalization이 3개가 있다. Shortcut connection으로는 이전 post에서 소개했던 IdentityPadding을 사용한다. 

~~~python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)      
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                stride=1, padding=1, bias=False)    
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = IdentityPadding(in_channels, out_channels, stride)
            
        self.stride = stride

    def forward(self, x):
        shortcut = self.down_sample(x)
        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        out += shortcut
        return out
~~~

<br>

IdentityPadding은 다음과 같다. 한가지 특이한 점은 IdentityPadding이 항상 적용되고 있기 때문에 down sampling이 일어날 때(stride가 2일 때) average pooling을 통해 shortcut을 지나는 feature map의 사이즈를 줄이는 것이다. 

~~~python
class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityPadding, self).__init__()

        if stride == 2:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.pooling = None
            
        self.add_channels = out_channels - in_channels
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        if self.pooling is not None:
            out = self.pooling(out)
        return out
~~~

<br>

PyramidNet 클래스의 __init__ 부분은 다음과 같다. 전체 네트워크의 깊이가 110이라면 num_layer는 18층이다. 각 block마다 width를 얼마나 늘려야하는지는 self.add_rate로 정의해놓았다. Additive 방식이며 self.get_layers 함수 내에서 사용된다. 

~~~python
# num_layers = (110 - 2)/6 = 18
self.num_layers = num_layers
self.addrate = alpha / (3*self.num_layers*1.0)

self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                       stride=1, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(16)

# feature map size = 32x32
self.layer1 = self.get_layers(block, stride=1)
# feature map size = 16x16
self.layer2 = self.get_layers(block, stride=2)
# feature map size = 8x8
self.layer3 = self.get_layers(block, stride=2)

self.out_channels = int(round(self.out_channels))
self.bn_out= nn.BatchNorm2d(self.out_channels)
self.relu_out = nn.ReLU(inplace=True)
self.avgpool = nn.AvgPool2d(8, stride=1)
self.fc_out = nn.Linear(self.out_channels, num_classes)
~~~

<br>

get_layers는 다음과 같다. 계속 out_channels를 in_channels보다 addrate만큼 증가시킨다. 이 때, channel 수로는 정수만 사용해야하므로 round와 int를 사용한다. 
~~~python
def get_layers(self, block, stride):
    layers_list = []
    for _ in range(self.num_layers - 1):
        self.out_channels = self.in_channels + self.addrate
        layers_list.append(block(int(round(self.in_channels)), 
                                 int(round(self.out_channels)), 
                                 stride))
        self.in_channels = self.out_channels
        stride=1

    return nn.Sequential(*layers_list)
~~~

<br>

PyramidNet을 학습한 그래프는 다음과 같다. 최고 error rate는 4.81%를 기록했다. 

<img src="https://www.dropbox.com/s/lwdujgn4uuunuwj/Screenshot%202018-11-23%2000.12.55.png?dl=1">

<br>

---

### 참고문헌
[^0]: https://arxiv.org/pdf/1605.07146.pdf
[^1]: https://arxiv.org/pdf/1608.06993.pdf
[^2]: https://arxiv.org/pdf/1610.02915.pdf
[^3]: https://arxiv.org/abs/1603.05027
[^4]: https://arxiv.org/pdf/1207.0580.pdf
[^5]: https://arxiv.org/pdf/1603.09382.pdf
[^6]: https://arxiv.org/pdf/1605.06431.pdf