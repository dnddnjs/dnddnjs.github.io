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

[다음 글](https://math.stackexchange.com/questions/2239040/show-that-fisher-information-matrix-is-the-second-order-gradient-of-kl-divergenc/2239159)에서도 FIM이 KL-divergence의 hessian과 동일하다($$\theta'$$이 $$\theta$$일 경우에 그렇다. Taylor expansion을 통해서 나오는 Hessian은 $$\theta$$위치에서의 Hessian 값이다.)라고 이야기한다. 

<img src="https://www.dropbox.com/s/mwrnf0c8rc2alyw/Screenshot%202018-06-09%2007.57.12.png?dl=1">

그리고 NPG에서 중요한 말인 "covariant"가 나온다. natural gradient $$H^{-1}g$$는 축의 변화에 invariant한 steepest direction을 가리킨다(이게 NPG가 하고 싶은 일이다). 이 다음부터는 이게 왜 covariant한지에 대한 증명이 나온다. 괴롭더라도 한 번 이해해놓으면 좋다. 

### 1.3 세번째 페이지
<img src="https://www.dropbox.com/s/pujcolzzuie1gvx/Screenshot%202018-06-09%2000.05.52.png?dl=1">
"covariant"하다는 것의 의미를 설명하는 부분이다. covariant를 이야기하기 위해 리만 공간에서의 거리(distance)를 이야기한다. 리만 공간에서 $$v$$와 $$v+\delta v$$사이의 거리는 다음과 같이 표현할 수 있다. 이 때, G를 metric tensor라고 이야기한다.

<img src="https://www.dropbox.com/s/acrieq6cvo42ldf/Screenshot%202018-06-09%2008.02.10.png?dl=1">

좌표계가 변함에 따라 거리라는 양은 변하지 않아야한다. 다음 예시를 살펴보자. 일반적인 Cartesian coordinate와 polar coordinate에서의 거리를 이야기해보자. Cartesian coordinate에서 metric tensor는 identity matrix이다. (거리를 $$dx^2+dy^2$$으로 나타낸다는 말이다) Cartesian coordinate에서의 x, y는 다음과 같이 $$r, \theta$$에 대해서 나타낼 수 있다. $$\delta x$$로 미분할 때는 $$x$$를 $$r$$과 $$\theta$$로 각각 편미분해서 더하는 것이다.

<img src="https://www.dropbox.com/s/6opt0zgyp6tn41g/Screenshot%202018-06-09%2008.08.33.png?dl=1">

Cartesian coordinate에서의 거리는 각 component의 제곱의 합이다. 이것을 Polar coordinate로 변환한다. 그러면 Polar coordinate에서의 metric tensor를 구할 수 있다. 이는 특수한 경우에 "거리"라는 개념이 coordinate에 따라 invariant 하다라는 것을 보인것이다. 이 때, "metric tensor"라는 것이 각 coordinate마다 거리를 계산하는데 사용된다. 즉, 이 metric tensor가 coordinate의 변화에 invariant한 metric이 되어야한다. 

<img src="https://www.dropbox.com/s/yo76t1a5ojxby7o/Screenshot%202018-06-09%2008.10.54.png?dl=1">


### 1.4 네번째 페이지
<img src="https://www.dropbox.com/s/ffgp47bobvreplc/Screenshot%202018-06-09%2000.06.28.png?dl=1">

이제 gradient의 covariant에 대해 생각해보자. natural gradient가 정말 covariant한가? 같은 벡터를 다른 system에서 다른 coordinate로 표현한 경우를 생각한다. 이는 다음과 같이 표현할 수 있다. 같은 벡터이며 같은 거리를 움직였다고 생각해보자.

<img src="https://www.dropbox.com/s/zajf90pytb2ho1a/Screenshot%202018-06-09%2008.16.53.png?dl=1">

이제 v의 미분을 생각해보자. 단순한 chain rule을 통해 다음과 같이 쓸 수 있다. 
<img src="https://www.dropbox.com/s/vhaw3kcn123hj7n/Screenshot%202018-06-09%2008.20.07.png?dl=1">

두 개의 system에서 $$\delta v$$와 $$\delta w$$는 같다고 가정을 하였다. 앞에서 이야기한 거리의 정의를 가져와서 쓰면 다음과 같다. $$\delta v$$를 $$w$$에 대해서 치환한다.

<img src="https://www.dropbox.com/s/129o7f00c60ww56/Screenshot%202018-06-09%2008.22.36.png?dl=1">

이 때, 함수 f의 w에 대한 gradient는 다음과 같다.
<img src="https://www.dropbox.com/s/cgmh1htfz36vtv8/Screenshot%202018-06-09%2008.25.16.png?dl=1">


### 1.5 다섯번째 페이지
<img src="https://www.dropbox.com/s/k606vh5iqv1jdn4/Screenshot%202018-06-09%2000.06.57.png?dl=1">

distance에 대한 식과 gradient 식을 사용해서 natural gradient가 covariant임을 보이자. NPG에서 봤듯이 natural gradient는 다음과 같이 쓸 수 있다. 두 개의 natural gradient가 같다면 covariant라고 할 수 있다.

<center><img src="https://www.dropbox.com/s/ojp2nih0gwt98pp/Screenshot%202018-06-09%2008.26.42.png?dl=1" width="200px"></center>

이는 다음과 같이 증명할 수 있다. 만약 서로 다른 두 coordinate에 대해 natural gradient가 같다면 같은 $$\delta v$$와 $$\delta w$$의 관계식을 만족할 것이다. 실제로 만족한다! 즉, natural gradient는 covariant한 것이다.

<img src="https://www.dropbox.com/s/a9o2c35hl5acfk6/Screenshot%202018-06-09%2008.30.26.png?dl=1">

그렇다면 왜 Natural Policy Gradient에서는 Natural gradient가 covariant하지 않았던 걸까? 논문에서는 metric tensor의 문제라고 했는데 어떤 문제인걸까? 그것을 여기의 수식과 어떻게 연결할 수 있을까?

### 1.6 여섯번째 페이지
<img src="https://www.dropbox.com/s/b9jd2rzww6wha1b/Screenshot%202018-06-09%2000.07.32.png?dl=1">

이것이 NPG의 알고리즘이다. 마지막 식이 나오는 과정은 다음과 같다. 

<img src="https://www.dropbox.com/s/taryh1wvqsqi1tx/Screenshot%202018-06-09%2018.03.36.png?dl=1">

<img src="https://www.dropbox.com/s/uook5mwpwgf7q89/Screenshot%202018-06-09%2018.04.07.png?dl=1">

<img src="https://www.dropbox.com/s/deh443s2xilfoc5/Screenshot%202018-06-09%2018.04.24.png?dl=1">

## 2. Newton's Method
--

optimize할 함수를 2nd order로 approximation해서 optimize하는 방법이다. Natural Gradient 방법론에서는 1nd order로 objective funciton을 approximation 했다는 것을 기억하자. Newton's method에서는 Hessian이 positive이면 convex 함수가 되어서 최소값을 구할 수 있다. 그러면 parameter를 이 최소값 지점으로 업데이트를 하고 이 과정을 반복하면 gradient 방법론 보다 훨씬 빠르게 수렴할 수 있다. 

[ppt 출처](https://www.cs.ccu.edu.tw/~wtchu/courses/2014s_OPT/Lectures/Chapter%209%20Newton's%20Method.pdf)


<img src="https://www.dropbox.com/s/2dhqtkindfqzsdx/Screenshot%202018-06-10%2011.09.50.png?dl=1">