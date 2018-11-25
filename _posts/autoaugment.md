---
layout: post
title: "CIFAR-10 정복 시리즈 10: Auto-Augment"
subtitle: "AutoAugment: Learning Augmentation Policies from Data"
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

  
## 논문 제목: AutoAugment: Learning Augmentation Policies from Data [2018 Oct]

<img src="https://www.dropbox.com/s/akwb88ckyew9vqp/Screenshot%202018-10-31%2015.35.14.png?dl=1">
- 논문 저자: Ekin D. Cubuk, Barret Zoph, Dandelion Mané, Vijay Vasudevan, Quoc V. Le
- 논문 링크: [https://arxiv.org/pdf/1805.09501.pdf](https://arxiv.org/pdf/1805.09501.pdf)
- 저자 코드: [https://github.com/tensorflow/models/tree/master/research/autoaugment](https://github.com/tensorflow/models/tree/master/
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
<img src="https://www.dropbox.com/s/t7syz6oqnccf092/Screenshot%202018-10-31%2016.05.37.png?dl=1">

- search space of operations
  - PIL에 있는 data augmentation 방법을 사용. 총 16개의 operations. ShearX, ShearY, TranslateX, TranslateY, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Color, Brightness, Sharpness, Cutout, SamplePairing
  - 각 operation은 magnitude가 있다. magnitude 자체는 continuous지만 논문에서는 discrete하게 space를 쪼갰다. 10개로 uniform spacing 처리. 
  - 확률도 continuous 하지만 discrete하게 쪼갬. 11개로. (왜 11개라는 애매한 숫자를 썼을까..)
  - 그래서 search space는 (16x10x11)^2의 크기를 가진다.
  - 하지만 policy는 5개의 sub-policy를 고르는 것이므로 (16x10x11)^10 ~ 2.9x10^32의 경우의 수가 존재. tabular 형식의 강화학습은 안된다.
  - 각 operation의 default range는 다음과 같다.
  <img src="https://www.dropbox.com/s/wrjqn4tco35865x/Screenshot%202018-11-01%2000.23.26.png?dl=1">

- search algorithm detail
  - search algorithm으로 강화학습 사용
  - 강화학습 에이전트는 rnn의 네트워크 구조를 가지며 PPO로 학습시킨다
  - controller는 매 스텝마다 softmax output에서부터 action을 선택한다. controller는 30개의 softmax output을 가진다. 5개의 subpolicy가 있으며 각 subpolicy는 2개의 operation이 있고 각 operation은 2개의 operation type을 결정해야하기 때문이다.
  - 강화학습 에이전트는 reward를 통해서 학습한다. reward는 쉽게 말하면 해당 policy를 사용해 data augmentation을 한 뉴럴넷 모델의 validation accuracy이다. 
  - 각 데이터셋마다 15000개의 policy를 sampling 했다. search가 끝나면 가장 좋은 5개의 policy의 sub-policy를 하나로 concat한다. 이게 하나의 policy가 되고 이걸로 model을 최종적으로 학습한다.
  - 이 부분은 상당히 불친절한 것 같다.

<br/>

### 4. Experiment and Results
- CIFAR-10 데이터를 그대로 search에 사용하기보다는 줄여서 사용. 이거를 reduced CIFAR-10이라 부르겠음. 50000개 중에서 랜덤하게 40000개를 선택.
- 적은 데이터셋에 많은 epoch을 학습하는게 좀 더 큰 데이터셋에 적은 epoch을 학습하는 것보다 더 나음
- child model(뉴럴넷 모델)로는 Wide-ResNet-40-2를 사용. 학습은 120 epoch.
- weight decay는 0.00001, learning rate는 0.01, cosine learning decay 사용
- 하나 알아둬야할 것은 policy에 포함된 augmentation만 적용하는게 아니다. 일단 baseline preprocessing을 한 다음에 policy를 적용한다. baseline pre-processing은 standardization, horizontal flip, zero-padding, random crop. policy를 적용한 다음 cutout을 적용한다. policy안에서 cutout이 있고 그 바로 뒤에 cutout이 또 있는데 따라서 cutout이 두 번 일어날 수 있다. 하지만 테스트해본결과 policy에서는 cutout이 잘 선택되지 않았다. 따라서 보통 하나의 cutout만 일어난다.
- CIFAR-10 데이터셋에서는 보통 color-based transformations을 골랐다. 25개의 sub-policy는 다음과 같다.
<img src="https://www.dropbox.com/s/ye7kgjaguymjboe/Screenshot%202018-11-01%2002.06.13.png?dl=1">

- 성능은 어떨까? PyramidNet + ShakeDrop에 적용했을 때 1.48 % error가 나왔다.
<img src="https://www.dropbox.com/s/o2ayzw67sak5yu1/Screenshot%202018-11-01%2002.09.02.png?dl=1">

- 설명이 정말 빈약하다. 코드를 봐야 이해할 수 있을 것 같다.

<br/>

### Discussion

