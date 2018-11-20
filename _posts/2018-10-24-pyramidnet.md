---
layout: post
title: "CIFAR-10 정복 시리즈 6: PyramidNet"
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
  - [CIFAR-10 정복 시리즈 3: Shake-shake](https://dnddnjs.github.io/cifar10/2018/10/13/shake_shake/)
  - [CIFAR-10 정복 시리즈 4: Shake-Drop](https://dnddnjs.github.io/cifar10/2018/10/19/shake_drop/)
  - [CIFAR-10 정복 시리즈 5: ENAS](https://dnddnjs.github.io/cifar10/2018/11/03/enas/)

- 관련 코드 링크
  - [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 

<br>

## CIFAR-10 정복 시리즈 2: PyramidNet
이전 포스트에서는 ResNet에 대해서 알아봤다. ResNet이 등장한 2015년 이후로 ResNet 기반의 새로운 네트워크가 많이 나왔다. 이번 포스트에서는 ResNet 이후 모델 구조에 관한 논문 중에 대표적인 3개 논문을 살펴볼 것이다. **WideResNet**[^0]은 ResNet의 깊이가 깊어지면 학습이 어렵다는 점을 해결하고자 깊게 쌓는 것이 아니라 넓게 쌓는 방법을 제안했다. **DenseNet**[^1]은 ResNet의 shortcut connection을 주의깊게 보고 더 dense한 connection 방법을 제안했다. **PyramidNet**[^2]은 ResNet의 down sampling 부분에 주목했다. ResNet에서 donw sampling을 특정 layer에서 수행하는 것이 아니라 모든 layer에서 나눠서 하는 것이 PyramidNet의 아이디어이다. 자 이제 하나씩 살펴보자.

1. [WideResNet](#WideResNet)
2. [DenseNet](#DenseNet)
3. [PyramidNet](#PyramidNet)

<br>

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

## DenseNet

### Abstract
- 단순히 말하자면 densenet은 resnet을 보다 깊게 쌓는 효과를 skip connection을 더 많이 쓰는 방법으로 얻는 것이다. 조금 더 나은 성능에 더 적은 parameter, 더 적은 computation이 필요하다.
- 다음 그림을 이해하면 densenet은 절반은 이해한 것이다. 각 "block"은 모든 앞의 block의 출력을 합해서 입력으로 받는다(더하기 아니고 concat). 그리고 그 block의 출력은 역시 그 이후의 모든 block의 입력이 된다.
- 보면 간단하지만 실제로 학습시켜서 resnet보다 더 좋은 성능을 내기는 어려웠을 것이다. 어떻게 했을까. 한 번 살펴보자.

<img src="https://www.dropbox.com/s/qlw9b3vad5osrqa/Screenshot%202018-10-11%2016.16.10.png?dl=1">

<br/>

### Introduction
- Resnet, Highway Networks, Stochastic depth(Resnet), FractalNets 모두 동일한 이야기를 하고 있다. 성능을 높이기 위해 앞쪽 layer에서 뒤 쪽 layer로 가는 short path를 써라. (그런거 보면 resnet의 아이디어는 간단하고 강력하다. 심지어 재현도 잘된다. 이런 아이디어는 사랑해야한다)
- l 번째 layer는 l개의 입력이 있고 그 layer의 출력도 L-l 개의 layer를 통과한다. 이렇게 하면 L(L+1) / 2의 connection이 생긴다. connection이 dense하게 많기 때문에 densenet이라 하겠다. 
- dense한 connection이 왜 좋을까? 반복되는 feature map 학습안하고 더 적은 파라메터로 네트워크를 학습할 수 있기 때문이라고 논문은 말한다.
- resnet의 변형체들을 보면 resnet layer 중에 많은 layer가 성능에 별 영향을 안주고 심지어 dropout 될 수도 있다고 한다. 이런 성향은 RNN과도 비슷하다. 대신 하나 하나의 layer가 독립적으로 parameter를 가지기 때문에 전체 크기가 큰 것이다.
- densenet은 특정 layer로 들어온 여러 information을 똭 분리해서 미분할 수 있다. 
- densenet의 최종 classifier는 모든 feature map을 보고 판단하게 된다.
- densenet은 기존 접근과는 다르게 feature의 재사용에 초점을 뒀다.
- 그러니까 이 모든 설명은 위 그림을 설명하는 것이다.

<br/>

### DenseNets
- H()는 convolution + BN + activation + pooling을 포함한 함수. x는 해당 layer의 output
- ResNet은 다음과 같이 표현 가능함.
<img src="https://www.dropbox.com/s/kamnx4362ntgrsn/Screenshot%202018-10-11%2016.41.56.png?dl=1">

- identity function과 H output이 summation으로 합쳐졌기 때문에 gradient의 flow가 원활하지 못하다.
- DenseNet은 다음과 같다. [x0, x1, ...]은 concatenation 이다.
<img src="https://www.dropbox.com/s/nojzgv0hg5kg61u/Screenshot%202018-10-11%2016.44.59.png?dl=1">

- Densenet 구조는 dense block을 여러개 합친 것이다. 이는 pooling을 통해 feature map 사이즈를 줄이기 위한 것이다. block 사이는 transition layer라고 부르며 BN + 1x1 conv + 2x2 average pooling 이다. 

<img src="https://www.dropbox.com/s/3y5idt67bea7jid/Screenshot%202018-10-11%2017.00.25.png?dl=1">

- feature map의 depth는 k=12 정도로 사용한다. l번째 layer의 input은 k0 + kx(l-1)의 depth를 가진다. k는 growth rate라고 부른다. 
- 이렇게 layer의 depth가 적어도 학습이 잘되는데 그건 collective knowledge 때문이라고 할 수 있다.
- 각 feature map은 global state라고 볼 수도 있는데 각 layer에서는 이 정보에 직접적으로 접근할 수 있다.
- bottleneck layer: BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
- 1x1 layer는 4k의 feature map을 만든다. 이거 사용하면 DenseNet-B
- transition layer에서 feature map 사이즈를 줄인다. 이거 사용하면 DenseNet-C
- bottleneck + compression 둘 다 사용하면 DenseNet-BC
- 다음은 구체적인 DenseNet 구조

<img src="https://www.dropbox.com/s/kvq5eypxxzw0l71/Screenshot%202018-10-11%2017.09.19.png?dl=1">

- ImageNet을 제외하고 모두 3개의 dense block을 사용
- input을 16 channel로 만듬
- transition layer에서는 2x2 average pooling으로 압축
- 각 dense block에서 feature map size: [32x32, 16x16, 8x8]

<br/>

### Experiment
- cifar10 에서의 성능 비교. 학습은 300 epoch. 150에서 lr을 0.1, 225에서 한 번 더.
<img src="https://www.dropbox.com/s/05baxltjfhiqqux/Screenshot%202018-10-11%2017.16.51.png?dl=1">

- DenseNet-BC(L=190, k=40)이 최고 성능 (3.46 %)
- C10+는 augmentation 한 것을 의미
- resnet과 비교해보면 비슷한 성능에서 parameter와 계산량이 확실히 적음을 알 수 있음.

<img src="https://www.dropbox.com/s/ss0fkd96l48b9jw/Screenshot%202018-10-11%2017.23.08.png?dl=1">

- parameter가 더 적으므로 overfitting이 되기 더 어려움
- 왜 더 잘 학습되는지에 대한 논문의 답: One explanation for the improved
accuracy of dense convolutional networks may be
that individual layers receive additional supervision from
the loss function through the shorter connections. One can
interpret DenseNets to perform a kind of “deep supervision”.
The benefits of deep supervision have previously
been shown in deeply-supervised nets (DSN; [20]), which
have classifiers attached to every hidden layer, enforcing the
intermediate layers to learn discriminative features.


<br>

## PyramidNet

<br>


### Abstract
- ResNet 같은 경우 feature map size를 줄이면서 feature dimension은 급격하게 늘린다
- 이 논문에서는 feature map dimension을 점자 늘리는 방식을 소개한다. 
- 이 방법은 generalization ability를 늘리는 효과가 있다.
- CIFAR-10, CIFAR-100, ImageNet에서 테스트함

<br/>

### Introduction
- "Residual networks behave
like ensembles of relatively shallow networks" 논문에 따르면 ResNet은 여러 Shallow Network의 ensemble 효과가 있다고 함
- 실제로 ResNet에서 특정 residual unit을 지우면 성능차이가 별로 없는데 이건 앙상블에서 모델 하나를 빼는 것과 같다.
- 하지만 VGG에서는 layer 하나를 빼면 성능 저하가 심하게 일어난다. 
- ResNet에서도 downsampling이 일어나는 block을 지우면 성능저하가 일어나는데 stochastic depth 방법을 사용하면 성능저하가 일어나지 않는다. 
- 따라서 이 논문에서는 bottleneck에 걸려있는 짐을 여러 layer로 분산시키려한다. 즉 feature map depth의 increase가 bottleneck에서만 일어나는 것이 아니고 모든 layer에서 분산되어 일어나도록 하는 것이다.
- 이렇게 점차 depth가 늘어가는 모양이 pyramid와 같아서 pyramidal residual network라고 부름
- 다음 그림을 보면 바로 이해할 수 있음. 그림에서 넓어져가는 모양은 depth를 뜻한다.

<img src="https://www.dropbox.com/s/fm7yui43ojdt5rt/Screenshot%202018-10-24%2015.36.15.png?dl=1"> 

<br/>

### 2. Network Architecture
- 보통 딥러닝 모델 구조에서는 feature map size가 줄어들 때 feature map depth를 큰 폭으로 늘리는 방법을 사용
- 기존 CIFAR 데이터셋에서의 ResNet 모델의 depth는 다음과 같다. k는 residual unit이 있는 group을 뜻한다.
<img src="https://www.dropbox.com/s/vnniq1ukyeqms2n/Screenshot%202018-10-24%2015.42.25.png?dl=1">

- 우리는 두 가지 방법으로 feature map dimension을 늘릴 것이다.
- 우선 더하는 방식은 다음과 같다.
<img src="https://www.dropbox.com/s/kz410nnpp8qma2b/Screenshot%202018-10-24%2016.08.30.png?dl=1">
- 곱하는 방식은 다음과 같다.
<img src="https://www.dropbox.com/s/cpghqys45lw2lax/Screenshot%202018-10-24%2016.13.27.png?dl=1">
- 그림으로 보면 다음과 같다. 
<img src="https://www.dropbox.com/s/ohfl1n8p3icjli3/Screenshot%202018-10-24%2016.13.57.png?dl=1">
- residual unit안에서의 building block 구조가 여러개 있는데 다음 그림과 같다. 이제 이 그림은 익숙할텐데 이 논문에서는 그 중에 (d)를 사용한다. 자세한 건 다음 파트에서.
<img src="https://www.dropbox.com/s/5qv21enadkxwefq/Screenshot%202018-10-24%2016.38.13.png?dl=1">
- shortcut connection도 고민이 필요하다. Pyramid Net의 경우 모든 unit에서 depth가 늘어나기 때문에 identity mapping은 할 수 없다. 따라서 1x1 conv를 쓰거나 zero-padding을 써야한다. 1x1 conv의 경우 너무 많이 쓰면 결과가 안좋아지기 때문에 이 논문에서는 zero-padding 방법을 사용하기로 했다. 자세한 건 다음 파트에서.

<br/>

### 3. Discussion
- 이 파트에서는 architecture에 대한 심화연구를 소개한다. 
- Effect of PyramidNet
  - pre-activation resnet이랑 pyramidnet 비교
  - training error, test error를 비교. 110 layer resnet 사용. pyramidnet은 alpha=48을 사용. 두 네트워크는 parameter 수가 동일.
  <img src="https://www.dropbox.com/s/en3sz3dprla5hjy/Screenshot%202018-10-24%2017.11.07.png?dl=1">
  - 앙상블 효과를 비교하기 위해 residual unit을 지워보는 실험을 함. 실험 결과는 다음과 같음. resnet 같은 경우 파란 수직선이 있는 downsampling 부분에서 error가 갑자기 튀는 것을 볼 수 있다. 하지만 pyramidnet의 경우 downsampling 부분에서도 error의 변화는 미비하다. 이것을 통해 기존 resnet 보다 pyramidnet의 앙상블 효과가 더 뛰어남을 볼 수 있다. 
  <img src="https://www.dropbox.com/s/68osemsp4orh1y6/Screenshot%202018-10-24%2017.15.05.png?dl=1">

- Zero-padded shorcut connection
  - resnet과 pre-activation resnet에서는 다양한 타입의 shortcut을 연구함. 이전 연구에 따르면 identity mapping이 다른 형태보다 파라메터를 가지지 않는다는 장점이 있어서 더 낫다. 파라메터가 더 적으면 오버피팅될 수 있는 가능성이 더 적기 때문이다. 또한 gradient를 그대로 흘려보낼 수 있다.
  - pyramidnet에서는 identity mapping을 사용할 수 없다. pyramidnet에서 사용하는 zero-padded identity mapping shortcut은 다음과 같다. k는 residual unit의 index, n는 group의 index, l은 feature map의 index이다. 매 unit 마다 feature map dimension이 늘어난다. 이전 unit의 feature map dimension은 Dk-1이고 현재 unit의 feature map dimension이 Dk라고 하면 (Dk)-(Dk-1) 만큼의 dimension을 zero-padding으로 채우는 것이다. 
  <img src="https://www.dropbox.com/s/ebj0hlb2n1s9lro/Screenshot%202018-10-24%2017.20.46.png?dl=1">
  - 그림으로 보자면 다음과 같다. 결국 zero-padding plane이 있고 그 plane을 이전 unit의 output이랑 concat 하는 것이라 보면 된다. (a)와 (b)는 동일하다 볼 수 있다. 위 식에서 Dk-1이상 Dk이하는 plain unit처럼 식이 써져있는것도 이 이유다. 즉 mixture of residual net and plain net이 되는 것이다.
  <img src="https://www.dropbox.com/s/gojtryzv82pvexb/Screenshot%202018-10-24%2017.27.00.png?dl=1">
  - 여러가지 shortcut connection 방법에 따른 성능은 다음과 같다. zero padding이 제일 좋다. 생각보다 방법마다 차이가 많이 난다.
  <img src="https://www.dropbox.com/s/cletnha9n1tkemy/Screenshot%202018-10-24%2017.44.03.png?dl=1">
  

- A New Building block
  - Building Block을 만드는 방법에서도 성능 개선의 여지가 충분히 있다. 
  - ReLU를 building block에 포함시키는 것은 non-linearity 때문에 꼭 필요하다.
  - addition 이후에 ReLU를 사용하는 것은 성능 저하를 가져온다. (이후에 shortcut connection이 모두 non-negative가 된다.)
  - 따라서 Pre-activation을 사용한다. 1000 layer 이상이 되어도 overfitting이 안 일어난다.
  - ReLU가 너무 많으면 성능이 안좋아지는 경향이 있다. 따라서 Residual unit 안에서 첫번째 ReLU를 생략한다. 다음 그림의 (b)와 (d)에 해당
  - 두 번째 ReLU를 생략하면 두 개의 convolution 사이에 non-linearity가 없어서 representation power가 약해진다.
  - BN의 역할은 activation을 normalize해서 수렴을 빠르게 만들어주는 것
  - 마지막에 BN을 붙이면 각 residual unit이 유용한지 아닌지를 판단해줄 수 있음
  - 실험결과는 그림 밑의 표와 같음. (d) = (b) + (c)이므로 (d)가 가장 좋은 성능을 보여줌
  <img src="https://www.dropbox.com/s/lv6lvozm1uzgm4h/Screenshot%202018-10-24%2021.15.39.png?dl=1">
  <img src="https://www.dropbox.com/s/44jj5jllnuafs4c/Screenshot%202018-10-24%2021.22.00.png?dl=1">

<br/>

### 4. Experiment Results
- CIFAR 데이터에 대해서는 standard data augmentation 적용 (horizontal flipping, translation by 4 pixels)
- SGD 사용. momentum = 0.9, 처음에는 lr=0.1, 150와 225 epochs 에서 0.1배씩, weight decay=0.0001
- batch size = 128
- CIFAR-10에서 3.31 % test error를 기록
<img src="https://www.dropbox.com/s/3y88bc0n16mmisf/Screenshot%202018-10-24%2021.24.21.png?dl=1">

- additive vs multiplicative pyramidnet
  - additive가 multiplicative에 비해 input side의 layer가 큰 경향이 있는데 그게 더 성능이 좋음.
  <img src="https://www.dropbox.com/s/vsvjwn6fw8amhf7/Screenshot%202018-10-24%2021.34.18.png?dl=1">


<br>

### 참고문헌
[^0]: https://arxiv.org/pdf/1605.07146.pdf
[^1]: https://arxiv.org/pdf/1608.06993.pdf
[^2]: https://arxiv.org/pdf/1610.02915.pdf
[^3]: https://arxiv.org/abs/1603.05027
[^4]: https://arxiv.org/pdf/1207.0580.pdf
[^5]: https://arxiv.org/pdf/1603.09382.pdf