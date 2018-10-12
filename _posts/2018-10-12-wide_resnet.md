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




