---
layout: post
title: "CIFAR-10 정복 시리즈 1: Batch-Norm"
subtitle: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
categories: cifar10
tags: dl
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


## 논문 제목: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [2015]

<img src="https://www.dropbox.com/s/e8qc4v8j88vm3ps/Screenshot%202018-11-04%2000.09.57.png?dl=1">
- 논문 저자: Sergey Ioffe, Christian Szegedy
- 논문 링크: [http://proceedings.mlr.press/v37/ioffe15.pdf](http://proceedings.mlr.press/v37/ioffe15.pdf)

<br/>

### Astract
- Deep Neural Network는 각 layer의 input distribution이 학습하는 동안 계속 변화한다. 이전 layer의 parameter가 계속 업데이트 되기 때문이다. 따라서 학습시키기 어렵다
- 이런 현상은 학습의 과정을 느리게 만든다. input distribution이 계속 변화하므로 optimizer의 learning rate를 작게 잡아야한다. neural network의 parameter를 초기화도 상당히 신경써서 해야한다.
- 이런 현상을 "internal covariance shift"라고 부르겠다. 이 문제를 해결하기 위해 input을 normalize하는 layer를 소개한다.
- 우리가 소개하는 layer는 전체 neural net의 일부이다. 학습할 때 mini-batch마다 normalization을 수행한다.
- "batch-normalization"으로 인해 더 높은 learning rate를 사용할 수 있다. initialization도 좀 더 자유롭게 할 수 있다.
- 어떤 경우에는 dropout의 기능도 대체하기 때문에 dropout이 없어도 된다.
- image classification에서의 SoTA 모델에 적용했봤다. 같은 accuracy를 도달하는데 14배 적은 계산량이 들었다. batch-norm 사용 모델을 ensemble로 사용했을 때 SoTA 달성.

<br/>

### Introduction
- SGD는 다음 수식을 통해 네트워크의 파라메터를 최적화한다.
<img src="https://www.dropbox.com/s/3i15tpiobbdjedy/Screenshot%202018-11-13%2015.55.16.png?dl=1">
