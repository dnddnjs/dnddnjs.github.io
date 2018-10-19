---
layout: post
title: "CIFAR-10 정복 시리즈 5: Shake-Drop"
subtitle: "SHAKEDROP REGULARIZATION"
categories: paper
tags: dl
comments: true
---

현재 딥러닝은 vision 분야에서 탁월한 성능을 보이며 NLP, Robotics, Sound 분야로 그 영향력을 확장해나가고 있다. 각 분야에서 딥러닝은 다른 모습으로 활용이 되고 있지만 그 기본이 되는 것은 거의 vision 분야에서 나온다. 따라서 vision 분야에서 기본이 되는 논문을 읽어보고 직접 구현해보는 것은 앞으로의 실력 향상에 상당한 도움이 된다. 이 post에서는 vision 분야의 특정 task들의 baseline이 되는 논문을 살펴보고 코드를 리뷰하고자 한다. 

image recognition에서는 다양한 benchmark가 존재한다. 그 중에서 CIFAR-10는 비교적 양이 적은 dataset이다. 그러면서도 도전적인 측면이 있기 때문에 vision 관련 모델을 테스트하기 좋다. 따라서 CIFAR-10에서 State-of-the art를 했던 모델들을 쭉 살펴볼 것이다. 이 series의 이름은 CIFAR-10 정복 시리즈이다. 이 series와 관련된 코드는 다음 github repository에서 볼 수 있다. 

- [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 


## 논문 제목: SHAKEDROP REGULARIZATION [2017 Feb]

<img src="https://www.dropbox.com/s/zd09fkmeu47wq3q/Screenshot%202018-10-19%2015.13.31.png?dl=1">
- 논문 저자: Yoshihiro Yamada, Masakazu Iwamura, Koichi Kise
- 논문 링크: [https://arxiv.org/pdf/1802.02375.pdf](https://arxiv.org/pdf/1802.02375.pdf)
- 논문 저자 코드: [https://github.com/xgastaldi/shake-shake](https://github.com/xgastaldi/shake-shake)

<br/>

### Abstract
- 이 논문은 shake-shake가 ResNeXt에만 적용가능하다는 단점을 개선하기 위함
- ShakeDrop은 