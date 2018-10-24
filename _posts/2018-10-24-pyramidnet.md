---
layout: post
title: "CIFAR-10 정복 시리즈 5-1: PyramidNet"
subtitle: "Deep Pyramidal Residual Networks"
categories: paper
tags: dl
comments: true
---

현재 딥러닝은 vision 분야에서 탁월한 성능을 보이며 NLP, Robotics, Sound 분야로 그 영향력을 확장해나가고 있다. 각 분야에서 딥러닝은 다른 모습으로 활용이 되고 있지만 그 기본이 되는 것은 거의 vision 분야에서 나온다. 따라서 vision 분야에서 기본이 되는 논문을 읽어보고 직접 구현해보는 것은 앞으로의 실력 향상에 상당한 도움이 된다. 이 post에서는 vision 분야의 특정 task들의 baseline이 되는 논문을 살펴보고 코드를 리뷰하고자 한다. 

image recognition에서는 다양한 benchmark가 존재한다. 그 중에서 CIFAR-10는 비교적 양이 적은 dataset이다. 그러면서도 도전적인 측면이 있기 때문에 vision 관련 모델을 테스트하기 좋다. 따라서 CIFAR-10에서 State-of-the art를 했던 모델들을 쭉 살펴볼 것이다. 이 series의 이름은 CIFAR-10 정복 시리즈이다. 이 series와 관련된 코드는 다음 github repository에서 볼 수 있다. 

- [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 


## 논문 제목: Deep Pyramidal Residual Networks [2017 Sep]

<img src="https://www.dropbox.com/s/zd09fkmeu47wq3q/Screenshot%202018-10-19%2015.13.31.png?dl=1">
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

- 우리는 다음과 같이 feature map dimension을 늘릴 것이다.
<img src="https://www.dropbox.com/s/kz410nnpp8qma2b/Screenshot%202018-10-24%2016.08.30.png?dl=1">
