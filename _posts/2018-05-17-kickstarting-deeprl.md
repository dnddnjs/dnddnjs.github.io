---
layout: post
title: "Kickstarting Deep Reinforcement Learning"
subtitle: "Kickstart: 선생님과 제자"
categories: paper
tags: rl
comments: true
---

## 논문 제목: Kickstartking Deep Reinforcement Learning [2018 March]

<img src="https://www.dropbox.com/s/7q8mudzrp3g5vwa/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-17%2017.46.47.png?raw=1">
- 논문 저자: Simon Schmitt (DeepMind)
- 논문 링크: [https://arxiv.org/pdf/1803.03835.pdf](https://arxiv.org/pdf/1803.03835.pdf)
- 함께 보면 좋을 논문: [policy distillation(2015)](https://arxiv.org/pdf/1511.06295.pdf)

## 1. Abstract
---
- 이전에 학습한 "teacher" agent가 새롭게 학습하는 "student" agent를 kickstart
- 이를 위해 policy distillation, population-based training의 아이디어를 차용
- 본 논문에서 제시하는 방법은 teacher와 student agent의 architecture에 제약을 가하지 않음. student가 스스로 teacher를 surpass 하도록 함
- kickstart된 새로운 agent의 data efficiency 향상
- 각각의 task에 최적화된 여러 teacher로부터 하나의 student가 kickstarting 가능
- 바닥부터 학습한 agent에 비해서 10배 빠른 학습, 42% 높은 성능

## 2. Conclusion
---
- policy-based RL 환경에서 implement하기 쉬움
- policy distillation에 비교해서 student가 스스로 teacher로부터의 조언을 가지고 learning objective를 균형잡음. 따라서 teacher보다 더 좋은 성능을 냄.
- 이전에 학습한 agent의 지식을 흡수하고 사용하는 새로운 agent --> 새로운 연구 방향
- 혼자서 학습하지 못하는 complex task도 학습 가능

## 3. Introduction
---
- 환경에서 여러가지를 경험하고 그 경험을 모으는 과정이 time-consuming & risk가 있음
- complex task를 학습하려 할 때 이런 점이 문제가 됌. 거의 billion 단위의 학습 step 소요.
- 다른 agent를 teacher로 해서 어떻게 하면 배울 수 있을까가 관심사
	- supervised learning에서는 weight transfer라는 게 있긴 함(보통 transfer 	learning 이라고 부르는 것). 하지만 강화학습에서는 잘 안되었음(왜인지 궁금하다!)
	- 그래서 그 잘 안되는 문제를 해결한게 이 논문에서 제시하는 "kickstarting" 방법
	- expert 자체(teacher의 다른 표현이다)가 아닌 그 경험 자체로부터 학습하던 imitation 	learning 과는 다름(imitation learning이 문제가 있었으니 다른 방법인 kickstarting을	제시했겠지? 그렇다면 imitation learning의 문제가 무엇이었을까? DARLA 논문을 읽어보면 	좋을 듯)
- policy distillation과 population based training의 아이디어를 합쳐봄(각각의 아이디어가 뭔지를 아는 것도 좋을 듯). policy distillation 과는 달리 student와 teacher architecture에 제약을 안주고 자동으로 teacher가 student에게 영향을 주도록 함.
- experiment는 뒤에서 언급

## 4. Kickstarking RL
---
### 4.1 Knowledge transfer 
- 기본적으로 pre-trained agent가 이용가능하다고 가정 (혹은 특정 specific task에 특화된 expert agent)
- 특정되어있지 않은 구조를 가진 새로운 agent를 학습하고 싶은데 teacher를 이용해서 (1) faster learning (2) higher performance 를 가지게 하고 싶음
- 간단히 말하자면 student가 샘플링한 trajectory에 대해서 student policy와 teacher policy의 차이를 나타내는 auxiliary loss function을 고려한다. 
	- 이 loss function은 기존 objective function에 더해져서 뉴럴넷 학습에 사용된다. 
	- 대신 auxiliary loss function에 weight를 곱해준다. 
	- weight의 역할은 전체 학습 비중이 점점 teacher의 policy를 따라하는게 아닌 expected reward를 높이는 방향으로 이동하도록 한다.
- kickstarting의 핵심은 knowledge transfer mechanism 이다.
	- 가장 유명한 knowledge transfer mechanism이 policy distillation이다.
	- teacher policy: $$\pi_T$$
	- teacher가 generate한 trajectory: $$(x_t)_{t>=0} $$
	- student policy: $$\pi_S$$ parameterized by $$w$$
	- student policy를 teacher policy에 가깝게 만드는 loss function: distill loss, 다음과 같다. 
	- $$H$$ 은 cross entropy를 의미함.

$$l_{distill}(w,x,t)=H(\pi_T(a|x_t) \| \pi_S(a|x_t, w))$$
	   
- 하지만 단지 이 loss function은 student가 teacher를 모방하게 만들 뿐임.
- student가 teacher로부터 도움을 받으면서 스스로 standard RL objective를 높이도록 하고 싶음
- 보통 expected return을 많이 objective로 사용함: $$E_{\pi_S}[R]$$,  $$R=\sum_{t>=0}\gamma^tr_t$$
- 따라서 expected return에 대한 loss term과 distill loss를 weighted sum한다. $$\lambda_k >= 0$$

$$l_{kick}^k=l_{RL}(w, x, t) + \lambda_kH(\pi_T(a | x_t) \| \pi_S(a | x_t, w))$$

- policy distillation과는 달리 trajectory를 student policy에 따라 sampling 함
- auxiliary loss는 다른 관점에서 보면 A3C의 entropy regularization과 같은 맥락으로 볼 수 있음
	- A3C loss: $$D_{KL}(\pi_S(a | x_t, w) \| U)$$, $$U$$는 uniform distribution
	- distill loss: $$D_{KL}(\pi_T(a | x_t, w) \| \pi_S(a | x_t, w))$$
	
- 다음 그림에서 첫 번째 그림은 모든 task를 한꺼번에 학습하는 보통의 RL agent 그림임. 두 번째 그림은 student 하나, teacher 하나인 에이전트임. 세 번째는 student 하나, teacher 3인 상황에서의 학습을 그린 것임. knowledge transfer의 흐름을 보기. 
<img src="https://www.dropbox.com/s/jd85p8mjbkp6yta/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-17%2018.06.09.png?raw=1">

### 4.2 Kickstarting Actor-Critic
- A3C의 loss를 다시 표현해보겠음
	- $$v_{t+1}$$ : value function target
	- $$V(x_t | \theta)$$ : value approximation computed by critic network
	- critic의 loss function: $$\|V(x_t | \theta)-v_{t}\|_2^2$$
	 
$$l_{A3C}(w, x, t)=log\pi_S(a_t | s_t, w)(r_t + \gamma v_{t+1} - V(x_t | \theta)) - \beta H(\pi_S(a | x_t, w))$$

- 이 때, A3C Kickstarting loss는 다음과 같음

$$l_{A3C}(w, x, t) + \lambda_kH(\pi_T(a | x_t) \| \pi_S(a | x_t, w))$$

### 4.3 Population based training
- Kickstarting에서 중요한 것은 바로 loss에서 $$\lambda_k$$의 자동 스케줄링임
- 기존에는 사람이 직접 schedule을 짰는데 그러면 또 추가로 전문지식이 필요함
- 만약 teacher가 한 agent가 아니고 여러 agent면 손으로 schedule 짜기가 더 어려움
- population based training이 이것을 자동으로 해줌
	- 다양한 hyper parameter를 가지는 population을 만듬
	- 이 중에서 랜덤으로 골라서 학습
	- 성능이 다른 놈보다 월등히 높은 놈이 있으먼 선택
	- 선택한 놈의 hyper parameter로 바로 대체하기보다는 조금 이동함