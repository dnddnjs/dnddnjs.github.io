---
layout: post
title: "CIFAR-10 정복 시리즈 6: Auto-Augment"
subtitle: "Deep Pyramidal Residual Networks"
categories: paper
tags: dl
comments: true
---

현재 딥러닝은 vision 분야에서 탁월한 성능을 보이며 NLP, Robotics, Sound 분야로 그 영향력을 확장해나가고 있다. 각 분야에서 딥러닝은 다른 모습으로 활용이 되고 있지만 그 기본이 되는 것은 거의 vision 분야에서 나온다. 따라서 vision 분야에서 기본이 되는 논문을 읽어보고 직접 구현해보는 것은 앞으로의 실력 향상에 상당한 도움이 된다. 이 post에서는 vision 분야의 특정 task들의 baseline이 되는 논문을 살펴보고 코드를 리뷰하고자 한다. 

image recognition에서는 다양한 benchmark가 존재한다. 그 중에서 CIFAR-10는 비교적 양이 적은 dataset이다. 그러면서도 도전적인 측면이 있기 때문에 vision 관련 모델을 테스트하기 좋다. 따라서 CIFAR-10에서 State-of-the art를 했던 모델들을 쭉 살펴볼 것이다. 이 series의 이름은 CIFAR-10 정복 시리즈이다. 이 series와 관련된 코드는 다음 github repository에서 볼 수 있다. 

- [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 


## 논문 제목: AutoAugment: Learning Augmentation Policies from Data [2018 Oct]

<img src="https://www.dropbox.com/s/akwb88ckyew9vqp/Screenshot%202018-10-31%2015.35.14.png?dl=1">
- 논문 저자: Ekin D. Cubuk, Barret Zoph†, Dandelion Mané, Vijay Vasudevan, Quoc V. Le
- 논문 링크: [https://arxiv.org/pdf/1805.09501.pdf](https://arxiv.org/pdf/1805.09501.pdf)
- 저자 코드: [https://github.com/tensorflow/models/tree/master/
research/autoaugment](https://github.com/tensorflow/models/tree/master/
research/autoaugment)

<br/>

### Abstract
- 이 논문에서는 image를 위한 data augmentation을 볼 것
- AutoAugment를 제안. Data augmentation policy를 찾는 방법
- data augmentation policy를 위한 search space를 정의하고 각 policy를 바로 데이터에 테스트
- 각 미니배치마다 policy 내부의 sub-policy 중 하나를 랜덤으로 선택해서 적용한다
- sub policy는 두 개의 operation으로 구성됌. 각 policy를 augmentation function을 적용할 확률과 얼만큼의 크기로 적용할 것인지를 정한다
- 데이터셋에서 가장 높은 validation accuracy를 가지는 best policy를 찾는다
- CIFAR-10 데이터에서 accuracy 1.48 % 달성
- 한 데이터셋에서 학습된 data augmentation policy는 다른 dataset에서도 잘 작동함

<br/>

### Introduction
- data augmentation은 데이터의 양과 다양성을 늘리는 방법임
- data augmentation은 모델이 데이터 도메인에 대해 invariant하도록 가르침
- 그동안 머신러닝과 컴퓨터 비전에서의 주된 관심은 더 좋은 네트워크 구조를 찾는 것이었다.
- 그에 비해 data augmentation은 별로 신경안씀
- 이 논문은 target dataset에 대해 효과적인 data augmentation policy를 자동으로 찾는 것을 목표로 한다.
- policy는 가능한 data augmentation 방법들의 몇가지 조합과 순서를 이야기한다
- best validation accuracy를 가지는 policy는 reinforcement learning으로 찾는다
- CIFAR-10, CIFAR-100, SVHN, ImageNet에서 SoTA이다
- ImageNet에서 학습한 data augmentation 방법이 FGVC 데이터셋에서 잘 동작함.

<br/>

### AutoAugment
- discrete search 문제에서 best augmentation policy를 찾는 것으로 환경을 설정함
- policy는 5개의 sub-policy로 구성됌. 각 sub-policy는 2개의 image operation을 포함. 이 2개의 operation은 차례대로 적용. 각 operation은 그 operation을 적용할 확률과 어떤 magnitude로 적용할지 두 개의 파라메터가 있음.
- 다음 그림이 Policy의 예시임. 첫번째 batch의 첫번째 sub-policy는 ShearX를 0.9의 확률과 10중 7의 크기로 적용한 후 0.8의 확률로 Invert를 적용. 확률적이기 때문에 Batch1, 2, 3가 다른 것을 볼 수 있음.
- 각 mini-batch마다 5개의 sub-policy 중 하나를 uniform random하게 선택해서 적용함 
- search space of operations
  - PIL에 있는 data augmentation 방법을 사용. 총 16개의 operations. Cutout, SamplePairing
<img src="https://www.dropbox.com/s/t7syz6oqnccf092/Screenshot%202018-10-31%2016.05.37.png?dl=1">
