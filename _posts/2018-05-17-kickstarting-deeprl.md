---
layout: post
title: "Kickstarting Deep Reinforcement Learning"
subtitle: "Kickstart: 선생님과 제자"
categories: paper
tags: rl
comments: true
---

<img src="https://www.dropbox.com/s/b0dp6tse95zudzw/Screenshot%202018-10-09%2020.54.11.png?dl=1">
- 논문 저자: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun(Microsoft Research)
- 논문 링크: [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)


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

$$l_{distill}(w,x,t)=H(\pi_T(a \vert x_t) \Vert \pi_S(a \vert x_t, w))$$
	   
- 하지만 단지 이 loss function은 student가 teacher를 모방하게 만들 뿐임.
- student가 teacher로부터 도움을 받으면서 스스로 standard RL objective를 높이도록 하고 싶음
- 보통 expected return을 많이 objective로 사용함: $$E_{\pi_S}[R]$$,  $$R=\sum_{t>=0}\gamma^tr_t$$
- 따라서 expected return에 대한 loss term과 distill loss를 weighted sum한다. $$\lambda_k >= 0$$

$$l_{kick}^k=l_{RL}(w, x, t) + \lambda_kH(\pi_T(a \vert x_t) \Vert \pi_S(a \vert x_t, w))$$

- policy distillation과는 달리 trajectory를 student policy에 따라 sampling 함
- auxiliary loss는 다른 관점에서 보면 A3C의 entropy regularization과 같은 맥락으로 볼 수 있음
	- A3C loss: $$D_{KL}(\pi_S(a \vert x_t, w) \Vert U)$$, $$U$$는 uniform distribution
	- distill loss: $$D_{KL}(\pi_T(a \vert x_t, w) \Vert \pi_S(a \vert x_t, w))$$
	
- 다음 그림에서 첫 번째 그림은 모든 task를 한꺼번에 학습하는 보통의 RL agent 그림임. 두 번째 그림은 student 하나, teacher 하나인 에이전트임. 세 번째는 student 하나, teacher 3인 상황에서의 학습을 그린 것임. knowledge transfer의 흐름을 보기. 
<img src="https://www.dropbox.com/s/jd85p8mjbkp6yta/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-17%2018.06.09.png?raw=1">

### 4.2 Kickstarting Actor-Critic
- A3C의 loss를 다시 표현해보겠음
	- $$v_{t+1}$$ : value function target
	- $$V(x_t \vert \theta)$$ : value approximation computed by critic network
	- critic의 loss function: $$\Vert V(x_t \vert \theta)-v_{t}\Vert_2^2$$
	 
$$l_{A3C}(w, x, t)=log\pi_S(a_t \vert s_t, w)(r_t + \gamma v_{t+1} - V(x_t \vert \theta)) - \beta H(\pi_S(a \vert x_t, w))$$

- 이 때, A3C Kickstarting loss는 다음과 같음

$$l_{A3C}(w, x, t) + \lambda_kH(\pi_T(a \vert x_t) \Vert \pi_S(a \vert x_t, w))$$

### 4.3 Population based training
- Kickstarting에서 중요한 것은 바로 loss에서 $$\lambda_k$$의 자동 스케줄링임
- 기존에는 사람이 직접 schedule을 짰는데 그러면 또 추가로 전문지식이 필요함
- 만약 teacher가 한 agent가 아니고 여러 agent면 손으로 schedule 짜기가 더 어려움
- population based training이 이것을 자동으로 해줌
	- 다양한 hyper parameter를 가지는 population을 만듬
	- 이 중에서 랜덤으로 골라서 학습
	- 성능이 다른 놈보다 월등히 높은 놈이 있으먼 선택
	- 선택한 놈의 hyper parameter로 바로 대체하기보다는 조금 이동함

	
## 5. Experiment
- IMPALA 에이전트로 DMLab-30 task에서 테스트함
- IMPALA 에이전트
	- visual input에 대해서는 convolution + lstm
	- language input에 대해서는 lstm
	- 두 개의 output을 concat, 그 다음 fully connected로 actor, critic output
	- small agent는 2개의 conv layer
	- large agent는 15개의 conv layer
- DMLab-30 task
	- 딥마인드에서 만든 30개의 간단한 task 들

<img src="https://www.dropbox.com/s/1tstdy9c0tivlgk/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-18%2010.16.04.png?raw=1">

- 1 high-capacity learner worker
- 150 actor worker
- worker는 task당 5개로 distributed (어마어마한 실험 환경임)
- 학습 평가는 사람이 했을 때의 점수와 비교함
- 첫 번째 그림은 바닥부터 학습한 에이전트와 single teacher를 통해 학습한 student agent의 성능을 비교한 그래프임. 주황색 그래프가 teacher agent가 되었다고 생각했을 때 teacher의 final score에 도달하는 시간에 비해 student는 1/10의 속도로 도달함.
- 두 번째 그래프는 kickstarting distillation weight의 evolution을 보여줌
<img src="https://www.dropbox.com/s/7jn1jg0ibvx9nvi/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-18%2010.21.47.png?raw=1">

- distillation weight의 scheduling의 효과는 다음 그래프에서 볼 수 있음. 생각보다 critical 한 것 같음. constant로 사용할 경우 supervised learning인 policy distillation과 거의 차이 없음. 

<img src="https://www.dropbox.com/s/ozhs48i8so4cevm/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-18%2010.29.04.png?raw=1">

## 6. Evaluation
- 기존 knowledge transfer 방법이 거의 imitation learning에 가까웠다면 kickstarting은 reinforcement learning과 적절히 섞은 점이 장점
- 기존 A3C나 IMPALA 같은 에이전트에 쉽게 implement 가능
- 추가적으로 weight scheduling이 필요한데 이게 성능에 영향을 많이 주는 것을 봐서는 현재 방법이 아닌 다른 방법으로 추가적으로 성능 개선이 가능하지 않을까 싶음
- teacher가 있는 상황이 많지 않을 것 같다는 생각이 듬(현실적일까..?)
- multiple teacher를 두는 것은 자원이 충분하지 않은 상황에서 오히려 전반적인 프로세스의 크기를 키우는 게 아닐까 싶음. 
- 바닥부터 스스로 학습하면서 자아를 분리해서 하나는 teacher로 하나는 student가 되는 방법은 어떨까 싶음

