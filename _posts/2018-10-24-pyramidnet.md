---
layout: post
title: "CIFAR-10 정복 시리즈 5-1: PyramidNet"
subtitle: "Deep Pyramidal Residual Networks"
categories: cifar10
tags: dl
comments: true
---

현재 딥러닝은 vision 분야에서 탁월한 성능을 보이며 NLP, Robotics, Sound 분야로 그 영향력을 확장해나가고 있다. 각 분야에서 딥러닝은 다른 모습으로 활용이 되고 있지만 그 기본이 되는 것은 거의 vision 분야에서 나온다. 따라서 vision 분야에서 기본이 되는 논문을 읽어보고 직접 구현해보는 것은 앞으로의 실력 향상에 상당한 도움이 된다. 이 post에서는 vision 분야의 특정 task들의 baseline이 되는 논문을 살펴보고 코드를 리뷰하고자 한다. 

image recognition에서는 다양한 benchmark가 존재한다. 그 중에서 CIFAR-10는 비교적 양이 적은 dataset이다. 그러면서도 도전적인 측면이 있기 때문에 vision 관련 모델을 테스트하기 좋다. 따라서 CIFAR-10에서 State-of-the art를 했던 모델들을 쭉 살펴볼 것이다. 이 series의 이름은 CIFAR-10 정복 시리즈이다. 이 series와 관련된 코드는 다음 github repository에서 볼 수 있다. 

- [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 


## 논문 제목: Deep Pyramidal Residual Networks [2017 Sep]

<img src="https://www.dropbox.com/s/ieukhhznpdtqqoc/Screenshot%202018-10-24%2016.14.35.png?dl=1">
- 논문 저자: YDongyoon Han, Jiwhan Kim, Junmo Kim
- 논문 링크: [https://arxiv.org/pdf/1610.02915.pdf](https://arxiv.org/pdf/1610.02915.pdf)
- 저자 코드: [https://github.com/jhkim89/PyramidNet](https://github.com/jhkim89/PyramidNet)

<br/>

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