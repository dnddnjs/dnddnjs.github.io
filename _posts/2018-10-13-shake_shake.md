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

<img src="https://www.dropbox.com/s/47rlhz4hkvt6twh/Screenshot%202018-10-13%2019.05.00.png?dl=1">