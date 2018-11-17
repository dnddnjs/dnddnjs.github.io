---
layout: post
title: "CIFAR-10 정복 시리즈 4: Wide ResNet"
subtitle: "Wide Residual Networks"
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
- dropout을 conv layer 사이에서 쓰려고 함. layer의 width를 넓힘으로서 parameter수가 많아졌는데 그로 인한 drawback를 보완하기 위함임

<br/>

### Wide Residual Networks
- 이전 resnet 들에 대해서 이야기는 안하겠다. 하지만 하나 기억해야할 것은 resnet 저자의 후속 논문으로 인해 conv-BN-ReLU 이 아니라 BN-ReLU-conv 을 사용한다는 점이다. [관련 논문: Identity mappings in deep
residual networks](https://arxiv.org/abs/1603.05027) 

- wide resnet에서는 bottleneck layer는 고려하지 않는다. bottleneck은 layer를 더 깊게 쌓고자 하는 것인데 이 논문은 그게 초점이 아니니까.
- residual block의 respresentation power를 늘리는 방법으로는 다음 세 가지가 있음. 세번째는 3x3 이상은 안쓰므로 탈락
  - block 마다 conv를 더 넣음
  - 각 conv의 feature plane을 더 넣음
  - conv의 filter 사이즈를 늘림
<img src="https://www.dropbox.com/s/7h5whxvdthu8y18/Screenshot%202018-10-12%2018.15.28.png?dl=1">

- Type of convolutions in residual block
  - block 내에 feature plane의 개수는 다 똑같음
  - original basic block은 B(3, 3)임. 3x3 conv + 3x3 conv를 의미함.
  - 이 3x3을 1x1로 대체할 수 있을까해서 여러가지로 실험함.

<img src="https://www.dropbox.com/s/h112dtgbhh7qw0p/Screenshot%202018-10-12%2018.36.18.png?dl=1">

- 결국 사용하는 residual block의 모양은 다음 그림에서 마지막과 같음. dropout이 conv 사이에 들어감
<img src="https://www.dropbox.com/s/f5dwsef7crx97f7/Screenshot%202018-10-12%2018.46.05.png?dl=1">
- 논문에서 사용하는 기호들
  - l: deeping factor(number of convoluion in block)
  - d: total number of blocks
  - k: widening factor --> multiply the number of features
  - n: total number of layers
  - 앞으로 WRN-n-k 이런식으로 표현할 것임. 

- 전반적인 wide resnet 구조는 다음과 같음
<img src="https://www.dropbox.com/s/ecxzfjhi878c8wd/Screenshot%202018-10-12%2018.54.00.png?dl=1">

<br/>

### Experiment Results
- CIFAR10에서 ZCA whitening 전처리를 함. 하지만 일부 실험에서는 그냥 mean/std normalization만 했음
- Type of convolutions in a block. parameter 수 비슷하게 유지
  - WRN-40-2 에서 테스트: B(1,3,1), B(3,1), B(1,3), B(3,1,1)
  - WRN-28-2: B(3,3)
  - WRN-22-2: B(3,1,3)
  - 결론: B(3,3) 사용

<img src="https://www.dropbox.com/s/rgf9m2qcgw0vlqr/Screenshot%202018-10-12%2019.01.14.png?dl=1">

- Number of convolutions per block
  - deeping factor l을 변화시켜보는 실험임
  - WRN-40-2 + 3x3 convolution을 사용함
  - parameter 수는 유지하면서 실험
  - B(3), B(3,3), B(3,3,3), B(3,3,3,3) --> B(3,3)이 최고 성능
  - l=3, 4가 안 좋은 건 residual connection이 상대적으로 감소해서 optimize가 어려운게 아닌가 싶음 

<img src="https://www.dropbox.com/s/9lr89qawsac7knc/Screenshot%202018-10-12%2019.03.40.png?dl=1">

- Width of residual blocks
  - 어떻게 보면 이 논문의 하이라이트
  - 다시 말하자면 width = feature plane의 수
  - n=40, 28, 22, 16 / k = 1, 2, 4, 8, 10, 12 를 실험해봄
  - CIFAR10에 대해서는 n=28 / k=10이 제일 좋은 정확도

 <img src="https://www.dropbox.com/s/8kaggunevtssp8r/Screenshot%202018-10-12%2021.02.35.png?dl=1">

 - 다른 resnet과의 비교
   - 이 실험에서는 dropout 사용 x
   - 일반적인 augmentation만 실행
   - 여전히 WRN-28-10이 제일 좋은 성능. ResNet-1001보다 0.92% 높음.
   - 비슷한 parameter 수를 가진 ResNet-1202와 WRN-40-4를 비교하면 학습속도가 5배 차이남.

<img src="https://www.dropbox.com/s/rilyppjh5hteue7/Screenshot%202018-10-12%2021.11.22.png?dl=1">

- dropout에 관련된 실험. dropout은 0.3을 사용
<img src="https://www.dropbox.com/s/bugro9g0uxbx40t/Screenshot%202018-10-12%2021.13.13.png?dl=1">

- dataset 별 최고 성능 모델. CIFAR10은 WRN-40-10 + dropout
<img src="https://www.dropbox.com/s/hmjuiunmjlzsev7/Screenshot%202018-10-12%2021.15.10.png?dl=1">

