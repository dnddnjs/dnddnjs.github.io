---
layout: post
title: "CIFAR-10 정복 시리즈 7: Shake-Drop"
subtitle: "SHAKEDROP REGULARIZATION"
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

## 논문 제목: SHAKEDROP REGULARIZATION [2018 Feb]

<img src="https://www.dropbox.com/s/zd09fkmeu47wq3q/Screenshot%202018-10-19%2015.13.31.png?dl=1">
- 논문 저자: Yoshihiro Yamada, Masakazu Iwamura, Koichi Kise
- 논문 링크: [https://arxiv.org/pdf/1802.02375.pdf](https://arxiv.org/pdf/1802.02375.pdf)

<br/>

### Abstract
- 이 논문은 shake-shake가 ResNeXt에만 적용가능하다는 단점을 개선하기 위함
- ShakeDrop은 ResNeXt 이외에도 ResNet, Wide ResNet, PyramidNet에 다 적용가능
- ShakeDrop만의 특징은 conv layer의 output에 - 도 곱할 수 있다는 점. strong disturb learning
- 그래서 CIFAR10에서 2.31 % error rate 달성

<br/>

### Introduction
- shake-shake의 단점
  - multi branch 구조에만 적용가능
  - memory efficient 하지 않음
- 이 부분을 개선한 ShakeDrop regularization 을 제안함
- shake-shake에 영감을 받아 만들었지만 disturbing하는 메카니즘은 완전 다름
- forwarding pass에서 -도 곱한다
- forward pass와 backward pass에서 conv layer output에 곱하는 값에 다른 값을 사용
- 그러면 학습이 불안정해지는데 따라서 ResDrop을 차용함

<br/>

### Existing Methods Required to introduce the proposed method
- Deep Network Architecture
  - ResNet : open the door to very deep CNNs
  - PyramidNet : vanilla resnet 중에 가장 높은 accuracy (cifar 데이터에서)
  - Wide ResNet : channel을 늘려서 성능 개선
  - ResNeXt : g(x) = x + f1(x) + f2(x)

- regularization
  - stochastic depth
  - shake-shake

<img src="https://www.dropbox.com/s/r4fsxd1z9oioeel/Screenshot%202018-10-20%2013.58.32.png?dl=1">

<br/>

### Proposed Method
- shake-shake는 forward pass에서 두 branch 사이를 interpolation
- feature space 상에서의 interpolation은 synthesizing data라고 해석할 수 있음 (이런 해석도 가능하구나)
- backward pass에서의 random variable은 모델이 오랫동안 학습할 수 있도록 해준다. 즉 너무 일찍 local에 빠지지 않도록 regularize한다는 이야기
- shake-shake는 이것을 하기 위해 2개 이상의 branch가 필요
  - 그러한 구조 때문에 memory를 많이 차지
  - 측정 결과 비슷한 파라메터 수를 가진 ResNeXt 모델에 비해 shake-shake는 11% 메모리를 더 사용
- shake-shake와 같은 regularization이 1 branch에서도 가능하게 하기 위해서는 단순한 interpolation 방법말고 다른 무언가가 필요
- 그 방법은 feature space 상에서 data를 synthesize 할 수 있어야함
- 일단 다음을 1-branch shake라고 부르겠음
  - pyramidnet 에 적용해봤는데 결과는 상당히 나빴음
<img src="https://www.dropbox.com/s/x4q3gw6y8m9x3sd/Screenshot%202018-10-20%2014.14.08.png?dl=1">

- 왜 1-branch shake가 실패했을까? 우리가 생각하기에는 너무 강한 perturbation이 가해졌기 때문이다. 하지만 perturbation을 약하게 하면 regularization의 효과가 줄어든다. 딜레마 (논문이 참 찰지다 빠져든다)
- 그래서 우리는 ResDrop의 아이디어를 쓰기로 했다. 대신 ResDrop을 그대로 사용한다면 너무 강한 perturbation이 일어나기 때문에 서로 다른 두 개의 network 사이를 switch하는 방식을 택하기로 했다.
- 이제 하이라이트!. PyramidNet과 PyramidNet+1-branch Shake 사이를 랜덤하게 오갈 것이다. 다음 식으로 그걸 할 수 있다. Bl은 linear decay rule이 적용되는 bernoulli random variable이다.

<img src="https://www.dropbox.com/s/bv20uznlczyob4q/Screenshot%202018-10-20%2014.31.28.png?dl=1">

- backward pass에서는 alpha 자리에 beta를 사용한다.

<img src="https://www.dropbox.com/s/k9rwk5ekgvev3q5/Screenshot%202018-10-20%2014.39.01.png?dl=1">

<br/>

### Experiments
- CIFAR 100 데이터에 대해서 alpha beta의 range를 바꿔가며 테스트. alpha는 [-1, 1] 사이의 값을 사용하고 beta는 [0, 1] 사이의 값을 사용하는 것이 제일 성능이 좋음.
<img src="https://www.dropbox.com/s/rybln8m5fmip910/Screenshot%202018-10-20%2014.42.09.png?dl=1">

- scaling factor 적용하는 방법에 대해서도 테스트함. Pixel은 scaling factor가 each residual block의 each element에 적용된다는 것. Pixel이 가장 성능은 좋지만 메모리를 많이 먹기 때문에 Image 방법을 사용함. 

<img src="https://www.dropbox.com/s/ppjgqxkxckqhlag/Screenshot%202018-10-20%2014.49.36.png?dl=1">

- regularization 방법을 비교함. 이 때 ResNet, PyramidNet, Wide ResNet, ResNeXt에서 각각 비교. 하나 중요한 점은 residual block이 BN으로 끝나야 한다는 것. 그러지 않으면 alpha beta의 값이 커질 때 학습이 발산할 수 있다. 따라서 EraseReLU가 우리 방법과 상당히 잘 맞음(Resnet과 ResNeXt 에서만). 결론은 Pyramidnet + shakedrop이 가장 성능 좋음

<img src="https://www.dropbox.com/s/nzp5z87drritxiv/Screenshot%202018-10-20%2014.56.07.png?dl=1">

- 가장 중요한 CIFAR10에서의 성능! 두 가지가 필요하다. longer learning은 cosine annealing 을 learning rate에 적용해서 1800 epoch 정도 학습한다. image preprocesiing은 learning image의 부분을 랜덤하게 채운다. (음.. 이건 잘 모르겠다). 결론적으로 CIFAR10 데이터에서 2.31 % error rate를 달성!

<img src="https://www.dropbox.com/s/0c8ahsplod8asry/Screenshot%202018-10-20%2015.01.05.png?dl=1">

- 다음은 네트워크 architecture
<img src="https://www.dropbox.com/s/n1ls9dsr5cqn5qf/Screenshot%202018-10-20%2015.02.08.png?dl=1">