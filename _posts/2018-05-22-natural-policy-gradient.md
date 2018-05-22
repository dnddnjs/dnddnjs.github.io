---
layout: post
title: "A Natural Policy Gradient"
subtitle: "피지여행 4번째 논문: NPG"
categories: paper
tags: rl
comments: true
---

# A Natural Policy Gradient [2001]

<img src="https://www.dropbox.com/s/hjnb3xkotjghw3t/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-22%2023.02.25.png?raw=1">

- 논문 저자: Sham Kakade
- 논문 링크: [https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)
- 함께 보면 좋을 논문: 
	- [Policy Gradient Methods for
Reinforcement Learning with Function
Approximation (2000)](hhttps://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
	- [Natural Gradient Works Efficiently in Learning(1998)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf)
- 논문을 보는 이유: TRPO와 NPG는 관련이 많기 때문에 TRPO를 더 잘 이해하기 위해 봄

## 1. Abstract
---

- natural gradient method를 policy gradient에 적용
- natural gradient는 steepest descent direction을 가짐
- gradient descent는 parameter를 한 번에 많이 update 할 수 없는 반면, natural gradient는 가장 좋은 action을 고르도록 학습이 됌 (sutton 논문에서와 같이 compatible value function을 사용할 경우 policy iteration에서 policy improvement 1 step의 과정에서)
- simple MDP와 tetris MDP에서 테스트함. 성능이 많이 향상


## 2. Discussion
---

- natural gradient method는 policy iteration에서와 같이 greedy action을 선택하도록 학습됌
- line search와 함께 쓰면 natural gradient method는 더 policy iteration 같아짐
- greedy policy iteration에서와는 달리 performance improvement가 보장됌
- 하지만 F(Fisher information matrix)가 asymtotically Hessian으로 수렴하지 않음. asymtotically conjugate gradient method(Hessian을 approx.로 구하는 방법)가 더 좋아 보일 수 있음
- 하지만 Hessian이 항상 informative하지 않고 tetris에서 봤듯이 natural gradient method가 더 효율적일 수 있음(pushing the policy toward choosing greedy optimal actions)
- conjugate gradient method가 좀 더 maximum에 빠르게 수렴하지만, performance는 maximum에서 거의 안변하므로 좋다고 말하기 어려움(?). 이 부분에 대해서 추가적인 연구 필요.

## 3. Introduction
---

- direct policy gradient method는 future reward의 gradient를 따라 policy를 update함
- 하지만 gradient descent는 non-covariant(1차 미분이므로 이렇게 표현하지 않나 싶음)
- 이 논문에서는 covarient gradient를 제시함 = natural gradient
- natural gradient와 policy iteration의 연관성을 설명하겠음: natural policy gradient is moving toward choosing a greedy optimal action (이런 연결점을 보이는 것이 왜 중요한 것일까?)


## 4. A Natural Gradient
---
### 4.1 환경에 대한 설정
- MDP: tuple $$(S, s_0, A, R, P)$$
- $$S$$: a finite set of states
- $$s_0$$: a start state
- $$A$$: a finite set of actions
- $$R$$: reward function $$R: S \times A -> [0, R_{max}]$$
- $$\pi(a;s, \theta)$$: stochastic policy parameterized by $$\theta$$
- 모든 정책 $$\pi$$는 ergodic: stationary distribution $$\rho^{\pi}$$이 잘 정의되어있음
- 이 논문에서는 sutton의 pg 논문의 두 세팅(start-state formulation, average-reward formulation) 중에 두 번째인 average-reward formulation을 가정 
- performance or average reward: $$\eta(\pi)=\sum_{s,a}\rho^{\pi}(s)\pi(a;s)R(s,a)$$
- state-action value: $$Q^{\pi}(s,a)=E_{\pi}[\sum_{t=0}^{\infty}R(s_t, a_t)-\eta(\pi)\vert s_0=s, a_0=a]$$
- 정책이 $$\theta$$로 parameterize되어있으므로 performance는 $$\eta(\pi_{\theta})$$인데 $$\eta(\theta)$$로 쓸거임

### 4.2 Natural Gradient
- 서튼 pg 논문의 policy gradient theorem에 따라 exact gradient of the average reward는 다음과 같음

$$\nabla\eta(\theta)=\sum_{s,a}\rho^{\pi}(s)\nabla\pi(a;s,\theta)Q^{\pi}(s,a)$$

- steepest descent direction of $$\eta(\theta)$$는 $$\eta(\theta + d\theta)$$를 최소화하는 $$d\theta$$로 정의
- gradient descent에서는 $$\vert d\theta \vert^2$$가 일정 크기 이하인 것으로 제약조건을 줌(held to small constant)
- 다음은 Natural Gradient Works Efficiently in Learning을 참조
	- Euclidian space에서는 gradient가 steepest direction을 가리키지만 Riemannian space에서는 그렇지 않다.
	-  