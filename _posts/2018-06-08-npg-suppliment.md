---
layout: post
title: "A Natural Policy Gradient 보충자료"
subtitle: "피지여행 4번째 발표 준비를 위한 보충자료"
categories: paper
tags: rl
comments: true
---

# Natural Policy Gradient 보충 자료

## 1. CS294의 자료
cs294에서도 NPG에 대한 내용을 소개한다. cs294 lecture 13에 해당하는 내용이다. 링크는 다음과 같다. 

- 링크: [http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf)

사실 이 피피티를 이해하려면 TRPO에 대한 내용도 알아야한다. 여기서 말하는 NPG는 Natural Policy Gradient에서 이야기하는 알고리즘이라기보다는 TRPO 저자 혹은 강의의 강사가 이야기하는 NPG이다. surrogate loss function이 들어가며 step size에 대한 구체적인 수식도 집어넣었다. 이 부분들을 감안하더라도 이 부분은 NPG에 대한 이해를 돕기 때문에 살펴보는 것에 의미가 있다.

### 1.1 첫번째 페이지
<img src="https://www.dropbox.com/s/mp134cb8kl6lkdb/Screenshot%202018-06-09%2000.04.04.png?dl=1">

여기서 이야기하는 $$L$$은 NPG 논문에서 이야기하던 $$\eta$$와는 다르다. 어떻게 다른지는 TRPO 논문을 참고하거나 CS294 강의를 들어보기 바란다. 일단 그 부분은 이해했다치고 이 페이지에서 하고 있는 것을 설명해보자. 

NPG가 하고 싶은 건 steepest direction을 찾고 그 방향대로 policy를 update하고 싶은 것이다. steepest direction이라는 것의 정의는 현재 parameter space 상의 point로부터 일정 거리만큼 움직였을 때, 그 중 가장 작은 함수 값을 가지는 지점으로 가는 방향이다. 이 때, 일정한 거리라는 것을 parameter space 상에서 정의하기 위해 coordinate 변화에 invariant한 metric을 사용해서 나타냈었다. 즉 다음과 같은 식으로 나타낼 수 있다는 것이다.

$$\vert d\theta \vert^2=\sum_{ij}(\theta)d\theta_id\theta_i=d\theta^TG(\theta)d\theta$$ 

여기서는 결과적으로는 같지만 다른 방식으로 표현한다. optimize하고자 하는 함수에 대해서는 1st order로 local approximation을 취한다. 그리고 일정한 거리라는 것을 표현하기 위해 distribution의 변화를 의미하는 KL-divergence를 가져온다. 이 KL-divergence는 2nd-order로 local approximation한다. 현재 parameter space상의 점에서는 KL-divergence의 미분값이 0이 되므로 결국 2차 미분인 Hessian 항만 남는다. 그게 다음 수식이 의미하는 것이다.

<img src="https://www.dropbox.com/s/g48q5fsgw72h6tb/Screenshot%202018-06-09%2000.48.33.png?dl=1">

이 문제는 KL-divergence constraint가 있는 optimization 문제이지만 lagrange multiplier를 사용해서 penalty 문제로 전환할 수 있다. 그게 다음 부분이다. 

<img src="https://www.dropbox.com/s/kkxwdmkna0gn72r/Screenshot%202018-06-09%2000.48.03.png?dl=1">

이 문제를 다 풀고 다면 다음과 같은 update rule을 구할 수 있다. 어떻게 구하는지는 이제 살펴보자.
<img src="https://www.dropbox.com/s/8e2t1e46mneg7tf/Screenshot%202018-06-09%2000.49.40.png?dl=1">


### 1.2 두번째 페이지
<img src="https://www.dropbox.com/s/vzaerazk3dq9425/Screenshot%202018-06-09%2000.05.18.png?dl=1">

이 페이지에서 놀라운 사실은 Fisher Information Matrix가 사실 KL divergence의 Hessian이라는 사실이다!!. 그동안 왜 Fisher를 쓰는지 고민이 많았는데 이 부분을 보니까 갑자기 이해가 되는 듯 하다.

그리고 NPG에서 중요한 말인 "covariant"가 나온다. natural gradient $$H^{-1}g$$는 축의 변화에 invariant한 steepest direction을 가리킨다(이게 NPG가 하고 싶은 일이다). 이 다음부터는 이게 왜 covariant한지에 대한 증명이 나온다. 괴롭더라도 한 번 이해해놓으면 좋다. 

### 1.3 세번째 페이지
<img src="https://www.dropbox.com/s/pujcolzzuie1gvx/Screenshot%202018-06-09%2000.05.52.png?dl=1">


### 1.4 네번째 페이지
<img src="https://www.dropbox.com/s/ffgp47bobvreplc/Screenshot%202018-06-09%2000.06.28.png?dl=1">

### 1.5 다섯번째 페이지
<img src="https://www.dropbox.com/s/k606vh5iqv1jdn4/Screenshot%202018-06-09%2000.06.57.png?dl=1">

### 1.6 여섯번째 페이지
<img src="https://www.dropbox.com/s/b9jd2rzww6wha1b/Screenshot%202018-06-09%2000.07.32.png?dl=1">
