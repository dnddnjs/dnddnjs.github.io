---
layout: post
title: "CIFAR-10 정복 시리즈 3: Wide ResNet"
subtitle: "Wide Residual Networks"
categories: paper
tags: dl
comments: true
---

현재 딥러닝은 vision 분야에서 탁월한 성능을 보이며 NLP, Robotics, Sound 분야로 그 영향력을 확장해나가고 있다. 각 분야에서 딥러닝은 다른 모습으로 활용이 되고 있지만 그 기본이 되는 것은 거의 vision 분야에서 나온다. 따라서 vision 분야에서 기본이 되는 논문을 읽어보고 직접 구현해보는 것은 앞으로의 실력 향상에 상당한 도움이 된다. 이 post에서는 vision 분야의 특정 task들의 baseline이 되는 논문을 살펴보고 코드를 리뷰하고자 한다. 

image recognition에서는 다양한 benchmark가 존재한다. 그 중에서 CIFAR-10는 비교적 양이 적은 dataset이다. 그러면서도 도전적인 측면이 있기 때문에 vision 관련 모델을 테스트하기 좋다. 따라서 CIFAR-10에서 State-of-the art를 했던 모델들을 쭉 살펴볼 것이다. 이 series의 이름은 CIFAR-10 정복 시리즈이다. 이 series와 관련된 코드는 다음 github repository에서 볼 수 있다. 

- [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 


## 논문 제목: Wide Residual Networks [2016 March]

<img src="https://www.dropbox.com/s/aqjty3k9tly0yei/Screenshot%202018-10-12%2017.39.25.png?dl=1">
- 논문 저자: Sergey Zagoruyko, Nikos Komodakis
- 논문 링크: [https://arxiv.org/pdf/1605.07146.pdf](https://arxiv.org/pdf/1605.07146.pdf)

<br/>

### Abstract
- Resnet이 1000개나 되는 layer도 학습이 되도록 했지만 깊어지면 깊어질수록 학습이 어려웠음.
- 그래서 Residual block에 대한 실험을 좀 해봄
- 우리는 depth는 줄이고 width를 넓혀봤음 (width는 각 block의 depth를 의미하는듯)
- 1000개 layer의 resnet보다 16-layer wide resnet이 CIFAR, SVHN, COCO에서 더 좋은 성능을 냈음(accuracy 말하는건지?)

<br/>

### Introduction
- 그동안 ResNet에서의 관점은 residual block 내의 activation 순서나 depth 였음
- identity mapping이 resnet의 강점인 동시에 약점이라는 것을 주장함
  - network에서의 gradient flow가 학습하는 동안 특정 block이 아무것도 안 배우도록 할 수도 있음
  - 따라서 적은 block이 유용한 정보를 가지고 있거나 여러 block이 별로 유용하지 않은 정보를 나눠가지는 것이 가능하다
  - 이 문제는 highway network 논문에서 다뤄지고 있다. 
  - stochastic depth 논문에서도 이 문제를 다룬다. dropout처럼 residual block을 랜덤하게 안쓴다. 이러한 방법이 효과를 보는 것 자체가 우리의 주장이 맞다는 것을 보여준다. (개인적으로 논리가 약하다고 생각함. 명쾌한 전개는 아닌듯)

- wider deep residual network가 다른 resnet보다 layer가 50배는 적으면서 2배 빠름
- dropout을 block 사이에서 쓰려고 함. layer의 width를 넓힘으로서 parameter수가 많아졌는데 그로 인한 drawback를 보완하기 위함임

<br/>

### Wide Residual Networks
- 이전 resnet 들에 대해서 이야기는 안하겠다. 하지만 하나 기억해야할 것은 resnet 저자의 후속 논문으로 인해 conv-BN-ReLU 이 아니라 BN-ReLU-conv 을 사용한다는 점이다. [관련 논문: Identity mappings in deep
residual networks](https://arxiv.org/abs/1603.05027) 

- wide resnet에서는 bottleneck layer는 고려하지 않는다. bottleneck은 layer를 더 깊게 쌓고자 하는 것인데 이 논문은 그게 초점이 아니니까.
- layer를 더 wider하게 만드는 걸 역시 다양하게 테스트해봤다. 간단히 다음과 같이
  - block 마다 conv를 더 넣어봤음
  - 각 conv의 feature plane을 더 넣어봤음
  - conv의 filter 사이즈를 늘려봤음
<img src="https://www.dropbox.com/s/7h5whxvdthu8y18/Screenshot%202018-10-12%2018.15.28.png?dl=1">

- Type of convolutions in residual block
  - block 내에 feature plane의 개수는 다 똑같음
  - original basic block은 B(3, 3)임. 3x3 conv + 3x3 conv를 의미함.
  - 이 3x3을 1x1로 대체할 수 있을까해서 여러가지로 실험함.

<img src="https://www.dropbox.com/s/h112dtgbhh7qw0p/Screenshot%202018-10-12%2018.36.18.png?dl=1">




