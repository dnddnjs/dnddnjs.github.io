---
layout: post
title: "CIFAR-10 정복 시리즈 3: DenseNet"
subtitle: "Densely Connected Convolutional Networks"
categories: 
tags: 
comments: true
---

## CIFAR-10 정복하기 시리즈 소개
CIFAR-10 정복하기 시리즈에서는 딥러닝이 CIFAR-10 데이터셋에서 어떻게 성능을 높여왔는지 그 흐름을 알아본다. 또한 코드를 통해서 동작원리를 자세하게 깨닫고 실습해볼 것이다. 

- CIFAR-10 정복하기 시리즈 목차(클릭해서 바로 이동하기)
  - [CIFAR-10 정복 시리즈 0: 시작하기](https://dnddnjs.github.io/cifar10/2018/10/07/start_cifar10/)
  - [CIFAR-10 정복 시리즈 1: Batch-Norm](https://dnddnjs.github.io/cifar10/2018/10/08/batchnorm/)
  - [CIFAR-10 정복 시리즈 2: ResNet](https://dnddnjs.github.io/cifar10/2018/10/09/resnet/)
  - [CIFAR-10 정복 시리즈 3: DenseNet](https://dnddnjs.github.io/cifar10/2018/10/11/densenet/)
  - [CIFAR-10 정복 시리즈 4: Wide ResNet](https://dnddnjs.github.io/cifar10/2018/10/12/wide_resnet/)
  - [CIFAR-10 정복 시리즈 5: Shake-shake](https://dnddnjs.github.io/cifar10/2018/10/13/shake_shake/)
  - [CIFAR-10 정복 시리즈 6: PyramidNet](https://dnddnjs.github.io/cifar10/2018/10/24/pyramidnet/)
  - [CIFAR-10 정복 시리즈 7: Shake-Drop](https://dnddnjs.github.io/cifar10/2018/10/19/shake_drop/)
  - [CIFAR-10 정복 시리즈 8: NASNet](https://dnddnjs.github.io/cifar10/2018/11/03/nasnet/)
  - [CIFAR-10 정복 시리즈 9: ENAS](https://dnddnjs.github.io/cifar10/2018/11/03/enas/)
  - [CIFAR-10 정복 시리즈 10: Auto-Augment](https://dnddnjs.github.io/cifar10/2018/11/05/autoaugment/)

- 관련 코드 링크
  - [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 


## 논문 제목: Densely Connected Convolutional Networks [2016 August]

<img src="https://www.dropbox.com/s/n7peav3s50ontg9/Screenshot%202018-10-11%2016.11.03.png?dl=1">
- 논문 저자: Gao Huang, Zhuang Liu, Laurens van der Maaten, JKilian Q. Weinberger
- 논문 링크: [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)

<br/>

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




