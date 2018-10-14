---
layout: post
title: "CIFAR-10 정복 시리즈 4: Shake-shake"
subtitle: "Shake-Shake regularization"
categories: paper
tags: dl
comments: true
---

현재 딥러닝은 vision 분야에서 탁월한 성능을 보이며 NLP, Robotics, Sound 분야로 그 영향력을 확장해나가고 있다. 각 분야에서 딥러닝은 다른 모습으로 활용이 되고 있지만 그 기본이 되는 것은 거의 vision 분야에서 나온다. 따라서 vision 분야에서 기본이 되는 논문을 읽어보고 직접 구현해보는 것은 앞으로의 실력 향상에 상당한 도움이 된다. 이 post에서는 vision 분야의 특정 task들의 baseline이 되는 논문을 살펴보고 코드를 리뷰하고자 한다. 

image recognition에서는 다양한 benchmark가 존재한다. 그 중에서 CIFAR-10는 비교적 양이 적은 dataset이다. 그러면서도 도전적인 측면이 있기 때문에 vision 관련 모델을 테스트하기 좋다. 따라서 CIFAR-10에서 State-of-the art를 했던 모델들을 쭉 살펴볼 것이다. 이 series의 이름은 CIFAR-10 정복 시리즈이다. 이 series와 관련된 코드는 다음 github repository에서 볼 수 있다. 

- [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 


## 논문 제목: Shake-Shake Regularization [2017 May]

<img src="https://www.dropbox.com/s/9n36ifz9ctxg90q/Screenshot%202018-10-13%2015.30.42.png?dl=1">
- 논문 저자: Xavier Gastaldi
- 논문 링크: [https://arxiv.org/pdf/1705.07485.pdf](https://arxiv.org/pdf/1705.07485.pdf)
- 논문 저자 코드: [https://github.com/xgastaldi/shake-shake](https://github.com/xgastaldi/shake-shake)

<br/>

### Abstract
- 우리는 overfit 문제를 해결하고 싶다
- 아이디어는 multi-branch network를 stochastic affine combination으로 대체하는 것
- CIFAR-10에서 2.86 % 정확도

<br/>

### Introduction

- ResNet은 훌륭한 모델이지만 여전히 적은 데이터셋에 대해서는 overfit
- 이를 해결하기 위한 기존의 방법들
  - weight decay
  - early stopping
  - dropout
  - Batch-norm
  - SGD
  - batch-norm 이후에는 multi-branch network 방법이 나옴

- 이 논문에서는 multi-branch network 연구의 연장선
  - 기존에는 standard summation
  - 우리는 stochastic affine transform

- motivation
  - 보통 augmentation은 input data에 대해서만 했는데 컴퓨터 입장에서는 입력이나 intermediate representation이나 차이없음
  - 그래서 우리는 그 중간에다가 augmentation을 적용해보기로 함
  - 이름은 Shake-Shake regularization임. 2개의 서로 다른 tensor를 blending 한다는 의미에서 이름을 붙임

- model description
  - 기본 2 branch residual branch는 식(1) 과 같음
  - 우리는 alpha라는 random variable을 사용해서 더함. 식(2)
  - 다른 방법론과는 다르게 shake-shake regularization은 whole image tensor를 하나의 스칼라 값으로만 곱해버림

<img src="https://www.dropbox.com/s/47rlhz4hkvt6twh/Screenshot%202018-10-13%2019.05.00.png?dl=1">

- Training Procedure
  - 이 논문의 핵심. 다음 그림을 이해하면 된다.
  - forward pass 할 때마다 새로운 alpha 사용
  - backward pass 할 때마다 새로운 beta 사용
  - test 할 때는 0.5 사용
  - 이전 연구들에서 gradient에 노이즈를 더하면 generalization에 도움이 된다는 것이 확인됌
  - shake-shake regularization은 gradient noise 대신 gradient augmentation 이라고 볼 수 있음.

<img src="https://www.dropbox.com/s/t2ijf2ahf5dkxa1/Screenshot%202018-10-13%2019.32.12.png?dl=1">

<br/>

### Improving on the best single shot published results on CIFAR

- CIFAR-10 implementation detail
  - first layer: 3x3 conv 16 filters
  - 각 stage 마다 feature map size: 32, 16, 8
  - width는 downsampling 할때마다 2배
  - 너무나 당연하게도 residual path는 preactivation: ReLU-Conv3x3-BN-ReLU-Conv3x3-BN-Mul
  - standard translation & flipping data augmentation
  - 1800 epoch 학습 (정말...???). stochasticity
  - learning rate 0.2로 시작. cosine function을 이용한 annealing
  - scaling coefficient가 mini-batch 안에서 image마다 다르면 "Image", 같으면 "Batch"
  - base network: 26 2x32d ResNet --> 26 depth를 가지며 2개의 residual branch를 가지고 있고 첫번째 residual block의 width는 32이다.
  - 논문 이름이 shake-shake인 이유: forward, backward에서 모두다 "shake"(pass 할 때마다 new random variable)

- 다음이 CIFAR-10에서의 실험 결과. Shake-Shake-Image + 26 2x96d 모델이 "2.86%"

<img src="https://www.dropbox.com/s/2t3twjh4sah9gwv/Screenshot%202018-10-13%2020.08.06.png?dl=1">

- 다른 state of the art 알고리즘들의 비교. Wide ResNet에 비해 1% 정도의 개선이 있었다.
<img src="https://www.dropbox.com/s/0pz8j5u8a24pr5z/Screenshot%202018-10-13%2020.09.49.png?dl=1">

<br/>

### Correlation between residual branches

- regularization을 통해 correlation이 늘었는지 줄었는지를 확인하기 위해 다음 실험을 했다.

<img src="https://www.dropbox.com/s/5ou3gczb6dgj17z/Screenshot%202018-10-13%2020.24.27.png?dl=1">