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

(개인생각) 뉴럴넷을 사용할 경우 gradient가 steepest direction이 아닌 경우가 많다. 이럴 경우 natural gradient가 steepest direction이 된다는 연구가 이뤄지고 있었다. 강화학습의 policy gradient은 objective function의 gradient를 따라 policy를 업데이트한다. 이 때, policy는 parameterized 되는데 이 경우에도 gradient 대신에 natural gradient가 좋다는 것을 실험해보는 논문인 것 같다. 

2차미분을 이용한 다른 방법들과의 비교가 생각보다 없는 점이 아쉽다.(Hessian을 이용한다거나 conjugate gradient method를 이용한다거나). 또한 natural gradient 만으로 업데이트하면 policy의 improvement보장이 안될 수 있다. policy의 improvement를 보장하기 위해 line search도 써야하는데 line search를 어떻게 쓰는지에 대한 자세한 언급이 없다.

natural policy gradient 논문은 natural gradient + policy gradient를 처음 적용했다는데 의의가 있다. 하지만 이 논문이 문제 삼은 gradient는 non-covariant하다(이게 어떤 문제가 있는지 모르겠다)라는 문제를 natural gradient를 통해 해결하지 못했다(논문에서 이 부분이 명확히 표현된건지 모르겠다). NPG의 뒤를 잇는 논문이 "covariant policy search"와 "natural actor-critic"에서 covariant하지 못하다는 것을 문제 삼는다. 

또한 논문은 pg의 두 가지 세팅 중에 average-reward setting(infinite horizon)에서만 NPG를 다룬다. "covariant policy search" 논문에서는 average-reward setting과 start-state setting 모두에 대해서 npg를 적용한다. 

natural gradient + policy gradient를 처음 제시했다는 것은 좋지만 npg 학습의 과정을 자세하게 설명하지 않았고 다른 2차 미분 방법들과 비교를 많이 하지 않은 점이 아쉬운 논문이다(인용된 논문들을 잘 안봐서 그럴지도 모른다).


## 2. Discussion
---

- natural gradient method는 policy iteration에서와 같이 greedy action을 선택하도록 학습됌
- line search와 함께 쓰면 natural gradient method는 더 policy iteration 같아짐
- greedy policy iteration에서와는 달리 performance improvement가 보장됌
- 하지만 F(Fisher information matrix)가 asymtotically Hessian으로 수렴하지 않음. asymtotically conjugate gradient method(Hessian의 inverse를 approx.로 구하는 방법)가 더 좋아 보일 수 있음
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
이 논문에서 제시하는 학습 환경은 다음과 같다.

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
#### 4.2.1 Policy gradient Theorem
서튼 pg 논문의 policy gradient theorem에 따라 exact gradient of the average reward는 다음과 같다. 다음 수식이 어떻게 유도되었는지, 어떤 의미인지 모른다면 서튼 pg 논문을 통해 제대로 이해하는 것이 좋다.

$$\nabla\eta(\theta)=\sum_{s,a}\rho^{\pi}(s)\nabla\pi(a;s,\theta)Q^{\pi}(s,a)$$

steepest descent direction of $$\eta(\theta)$$는 $$\eta(\theta + d\theta)$$를 최소화하는 $$d\theta$$로 정의된다. 이 때, $$\vert d\theta \vert^2$$가 일정 크기 이하인 것으로 제약조건을 준다(held to small constant). Euclidian space에서는 $$\eta(\theta)$$가 steepest direction이지만 Riemannian space에서는 natural gradient가 steepest direction이다. 

#### 4.2.2 Natural gradient 증명
Riemannian space에서 거리는 다음과 같이 정의된다. $$G(\theta)$$는 특정한 양수로 이루어진 matrix이다.

$$\vert d\theta \vert^2=\sum_{ij}(\theta)d\theta_id\theta_i=d\theta^TG(\theta)d\theta$$

이 수식은 Natural Gradient Works Efficiently in Learning 논문에서 증명되어있다. 다음은 natural gradient 증명이다. 

steepest direction을 구할 때 $$\theta$$의 크기를 제약조건을 준다. 제약조건은 다음과 같다. 

$$\vert d\theta \vert^2 = \epsilon^2$$

그리고 steepest vector인 $$d\theta$$는 다음과 같이 정의할 수 있다. 

$$d\theta = \epsilon a$$

$$\vert a \vert^2=a^TG(\theta)a = 1$$

이 때, $$a$$가 steepest direction unit vector이 되려면 다음 수식을 최소로 만들어야 한다. (이 수식은 잘 모르겠지만 $$\theta$$에서의 1차근사를 가정하는게 아닌가 싶다.

$$\eta(\theta + d\theta) = \eta(\theta) + \epsilon\nabla\eta(\theta)^Ta$$

위 수식이 제약조건 아래 최소가 되는 $$a$$를 구하기 위해 Lagrangian method를 사용한다. Lagrangian method를 모른다면 [위키피디아](https://en.wikipedia.org/wiki/Lagrange_multiplier)를 참고하는 것을 추천한다. 위 수식이 최소라는 것은 $$\nabla\eta(\theta)^Ta$$가 최소라는 것이다. 

$$\frac{\partial}{\partial a_i}(\nabla\eta(\theta)^Ta - \lambda a^TG(\theta)a)=0$$

따라서 $$(\nabla\eta(\theta)^Ta - \lambda a^TG(\theta)a)=0$$는 상수이다. 상수를 미분하면 0이므로 이 식을 $$a$$로 미분한다. 그러면 다음과 같다. steepest direction을 구한 것이다.

$$\nabla\eta(\theta) = 2 \lambda G(\theta)a$$

$$a=\frac{1}{2\lambda}G^{-1}\nabla\eta(\theta)$$

이 때, 다음 식을 natural gradient라고 정의한다.

$$\bar{\nabla}\eta(\theta) = G^{-1}\nabla\eta(\theta)$$

natural gradient를 이용한 업데이트는 다음과 같다. 

$$\theta_{t+1}=\theta_t - \alpha_tG^{-1}\nabla\eta(\theta)$$

여기까지는 natural gradient의 증명이었다. 이 natural gradient를 policy gradient에 적용한 것이 natural policy gradient이다. natural policy gradient는 다음과 같이 정의된다.

$$\bar{\nabla}\eta(\theta) = F^{-1}\nabla\eta(\theta)$$

$$G$$ 대신 $$F$$를 사용했는데 $$F$$는 Fisher information matix이다. 수식은 다음과 같다.

$$F(\theta) = E_{\rho^\pi(s)}[F_s(\theta)]$$

$$F_s(\theta)=E_{\pi(a;s,\theta)}[\frac{\partial log \pi(a;s, \theta)}{\partial \theta_i}\frac{\partial log \pi(a;s, \theta)}{\partial\theta_j}]$$

왜 G가 F가 되는지는 아직 잘 모르겠다. 거리라는 개념을 표현하려면 

## 5. The Natural Gradient and Policy Iteration
---
### 5.1 Theorem 1
sutton pg 논문에 따라 $$Q^{\pi}(s,a)$$를 approximation한다. approximate하는 함수 $$f^{\pi}(s,a;w)$$는 다음과 같다.(compatible value function)

$$f^{\pi}(s,a;w)=w^T\psi^{\pi}(s,a)$$

$$\psi^{\pi}(s,a) = \nabla log\pi(a;s,\theta)$$

$$w$$는 원래 approximate하는 함수 $$Q$$와 $$f$$의 차이를 줄이도록 학습한다(mean square error). 수렴한 local minima의 $$w$$를 $$\bar{w}$$라고 하겠다. 에러는 다음과 같은 수식으로 나타낸다. 

$$\epsilon(w,\pi)\equiv\sum_{s, a}\rho^{\pi}(s)\pi(a;s,\theta)(f^{\pi}(s,a;w)-Q^{\pi}(s,a))^2$$

위 식이 local minima이면 미분값이 0이다. $$w$$에 대해서 미분하면 다음과 같다. 

$$\sum_{s, a}\rho^{\pi}(s)\pi(a;s,\theta)\psi^{\pi}(s,a)(\psi^{\pi}(s,a)^T\bar{w}-Q^{\pi}(s,a))=0$$

$$(\sum_{s, a}\rho^{\pi}(s)\pi(a;s,\theta)\psi^{\pi}(s,a)\psi^{\pi}(s,a)^T)\bar{w}=\sum_{s, a}\rho^{\pi}(s)\pi(a;s,\theta)\psi^{\pi}(s,a)Q^{\pi}(s,a))$$

이 때, 위 식의 우변은 $$\psi$$의 정의에 의해 policy gradient가 된다. 또한 왼쪽 항에서는 Fisher information matrix가 나온다.

$$F(\theta)=\sum_{s,a}\pi(a;s,\theta)\psi^{\pi}(s,a)\psi^{\pi}(s,a)=E_{\rho^\pi(s)}[F_s(\theta)]$$

따라서 다음과 같다.

$$F(\theta)\bar{w}=\nabla\eta(\theta)$$

$$\bar{w}=F(\theta)^{-1}\nabla\eta(\theta)$$

이 식은 natural gradient 식과 동일하다. 이 식은 policy가 update 될 때, value function approximator의 parameter 방향으로 이동한다는 것을 의미한다. function approximation이 정확하다면 그 parameter의 natural policy gradient와 inner product가 커야한다. 

### 5.2 Theorem 2: Greedy Polict Improvement
natural policy gradient가 단순히 더 좋은 행동을 고르도록 학습하는게 아니라 가장 좋은 (greedy) 행동을 고르도록 학습한다는 것을 증명하는 파트이다. 이것을 일반적인 형태의 policy에 대해서 증명하기 전에 exponential 형태의 policy에 대해서 증명하는 것이 Theorem 2이다.

policy를 다음과 같이 정의한다.

$$\pi(a;s,\theta) \propto exp(\theta^T\phi_{sa})$$

$$\bar{\nabla}\eta(\theta)$$가 0이 아니고 $$\bar{w}$$는 approximation error를 최소화한 $$w$$라고 가정한다. 이 상태에서 natural gradient update를 생각해보자. policy gradient는 gradient ascent임을 기억하자.

$$\theta_{t+1}=\theta_t + \alpha_t\bar{\nabla}\eta(\theta)$$

이 때 $$\alpha$$가 learning rate로 parameter를 얼마나 업데이트하는지를 결정한다. 이 값을 무한대로 늘렸을 때 policy가 어떻게 업데이트되는지 생각해보자. 

$$\pi_{\infty}(a;s)=lim_{\alpha\rightarrow\infty}\pi(a;s,\theta+\alpha\bar{\nabla}\eta(\theta))-(1)$$

function approximator는 다음과 같다. 

$$f^{\pi}(s,a;w)=w^T\psi^{\pi}(s,a)$$

Theorem 1에 의해 위 식은 아래와 같이 쓸 수 있다.


$$f^{\pi}(s,a;w)=\bar{\nabla}\eta(\theta)^T\psi^{\pi}(s,a)$$

$$\theta$$의 정의에 의해 $$\psi$$는 다음과 같다.

$$\psi^{\pi}(s,a)=\phi_{sa}-E_{\pi(a';s,\theta)}[\phi_{sa'}]$$

function approximator는 다음과 같이 다시 쓸 수 있다.

$$f^{\pi}(s,a;w)=\bar{\nabla}\eta(\theta)^T(\phi_{sa}-E_{\pi(a';s,\theta)}[\phi_{sa'}])$$

greedy policy improvement가 Q function 값 중 가장 큰 값을 가지는 action을 선택하듯이 여기서도 function approximator의 값이 가장 큰 action을 선택하는 상황을 가정해본다. 이 때 function approximator의 argmax는 다음과 같이 쓸 수 있다.

$$argmax_{a'}f^{\pi}(s,a)=argmax_{a'}\bar{\nabla}\eta(\theta)^T\phi_{sa'}$$

(1) 식을 다시 살펴보자. policy의 정의에 따라 다음과 같이 쓸 수 있다. 

$$\pi(a;s,\theta + \alpha\bar{\nabla}\eta(\theta)) \propto exp(\theta^T\phi_{sa} + \alpha\bar{\nabla}\eta(\theta)^T\phi_{sa})$$

$$\bar{\nabla}\eta(\theta) \neq 0$$이고 $$\alpha\rightarrow\infty$$이면 exp안의 항 중에서 뒤의 항이 dominate하게 된다. 여러 행동 중에 $$\bar{\nabla}\eta(\theta)^T\phi_{sa}$$가 가장 큰 행동이 있다면 이 행동의 policy probability가 1이 되고 나머지는 0이 된다. 따라서 다음이 성립한다.

$$\pi_{\infty}=0$$ 

if and only if 

$$a \notin argmax_{a'}\bar{\nabla}\eta(\theta)^T\phi_{sa'}$$

이 결과로부터 natural policy gradient는 단지 더 좋은 action이 아니라 best action을 고르도록 학습이 된다. 하지만 non-covariant gradient(1차미분) 에서는 그저 더 좋은 action을 고르도록 학습이 된다. 하지만 이 natural policy gradient에 대한 결과는 infinite learning rate 세팅에서만 성립함. 좀 더 일반적인 경우에 대해서 살펴보자.

#### 5.3 Theorem 3 
Theorem 2에서와는 달리 일반적인 policy를 가정하자(general parameterized policy). Theorem 3는 이 상황에서 natural gradient를 통한 업데이트가 best action를 고르는 방향으로 학습이 된다는 것을 보여준다. 

natural gradien에 따른 policy parameter의 업데이트는 다음과 같다. $$\bar{w}$$는 approximation error를 minimize하는 $$w$$이다.

$$\delta\theta = \theta' - \theta = \alpha\bar{\nabla}\eta(\theta)=\alpha\bar{w}$$

policy에 대해서 1차근사를 하면 다음과 같다. 

$$\pi(a;s,\theta')=\pi(a;s,\theta)+\frac{\partial\pi(a;s,\theta)^T}{\partial\theta}\delta\theta + O(\delta\theta^2)$$

$$=\pi(a;s,\theta)(1+\psi(s,a)^T\delta\theta) + O(\delta\theta^2)$$

$$=\pi(a;s,\theta)(1+\alpha\psi(s,a)^T\bar{w}) + O(\delta\theta^2)$$

$$=\pi(a;s,\theta)(1+\alpha f^{\pi}(s,a;\bar{w}) + O(\delta\theta^2)$$

policy 자체가 function approximator의 크기대로 업데이트가 되므로 local하게 best action의 probability는 커지고 다른 probability의 크기는 작아질 것이다. 하지만 만약 greedy improvement가 된다하더라도 그게 performance의 improvement를 보장하는 것은 아니다. 하지만 line search와 함께 사용할 경우 improvement를 보장할 수 있다. 

## 6. Metrics and Curvatures
---
다음 식에 해당하는 G는 Fisher Information Matrix만 사용할 수 있는 것이 아니다.

$$\vert d\theta \vert^2=\sum_{ij}(\theta)d\theta_id\theta_i=d\theta^TG(\theta)d\theta$$

이 파트에서는 FIM과 다른 metric 사이의 관계를 다룬다. 

- In the different setting of parameter estimation, the Fisher information converges to the ```Hessian```, so it is [asymptotically efficient](https://en.wikipedia.org/wiki/Efficiency_(statistics)#Asymptotic_efficiency)
- 이 논문의 경우, 아마리 논문의 'blind separation case'와 유사한데 이 때는 꼭 asymtotically efficient하지 않다. 이 말은 즉 2nd order 수렴이 보장되지 않는다는 것이다.
- [Mackay](http://www.inference.org.uk/mackay/ica.pdf) 논문에서 hessian에서 data independant한 term을 metric으로 가져오는 방법을 제안했다. 그래서 performance를 2번 미분해보면 다음과 같다. 하지만 다음 식에서는 모든 항이 data dependent하다(Q가 있으니까). 첫 번째 항이 그나마 FIM과의 관련성이 있을 수 있지만 Q 값이 curvature에 weight를 주는 방식 때문에 다르다고 할 수 있다.

$$
\nabla^2\eta(\theta)=\sum_{sa}\rho^{\pi}(s)(\nabla^2\pi(a;s)Q^{\pi}(s,a)+\nabla\pi(a;s)\nabla Q^{\pi}(s,a)^T+\nabla Q^{\pi}(s,a)\nabla\pi(a;s)^T)
$$


- hessian은 보통 positive definite가 아닐수도 있다. 따라서 local maxima가 될 때까지 Hessian이 사용하기 별로 안좋다. 그리고 local maxima에서는 Hessian보다는 Conjugate methods가 더 효율적이다. 

이 파트에서는 무엇을 말하고 있는지 알기가 어렵다. FIM과 Hessian이 관련이 있다는 것을 알겠다. 하지만 asymtotically efficient와 같은 내용을 모르므로 내용의 이해가 어려웠다.


## 7. Experiment
---
논문에서는 natural gradient를 simple MDP와 tetris MDP에 대해서 테스트했다. practice에서는 Fisher information matrix는 다음과 같은 식으로 업데이트한다.

$$f\leftarrow f+\nabla log \pi(a_t; s_t, \theta)\nabla log \pi(a_t; s_t, \theta)^T$$

T length trajectory에 대해서 f/T를 통해 F의 estimate를 구한다.

### 7.1 Linear Quadratic regulator
에이전트를 테스트할 환경은 다음과 같은 dynamics를 가지고 있다. $$u(t)$$는 control signal로서 에이전트의 행동이라고 생각하면 된다. $$\epsilon$$은 noise distribution으로 환경에 가해지는 노이즈이다. 에이전트의 목표는 적절한 $$u(t)$$를 통해 
x(t)를 0으로 유지하는 것이다. 제어분야에서의 LQR controller 문제이다.

$$
x(t+1) = 0.7x(t)+u(t)+\epsilon(t)
$$

x(t)를 0으로 유지하기 위해서 $$x(t)^2$$를 cost로 잡고 이 cost를 최소화하도록 학습한다. 이 시스템을 linear라고 부르는 것은 아래 그림과 같이 선형의 형태를 띄기 때문이다. 이 논문에서 실험할 때는 이 그림에서의 system에 noise를 더해준 것이다. [그림 출처](https://stanford.edu/class/ee363/lectures/dlqr.pdf)

<img src='https://www.dropbox.com/s/vz0q97lcek4oti5/Screenshot%202018-06-08%2014.21.10.png?dl=1'>

이 실험에서 사용한 parameterized policy는 다음과 같다. parameter가 $$\theta_1$$과 $$\theta_2$$ 밖에 없는 상당히 간단한 policy이다. 

$$
\pi(u;x,\theta) \propto exp(\theta_1 s_1 x^2 + \theta_2 s_2 x)
$$

이 policy를 간단히 numpy와 matplotlib를 이용해서 그려봤다. $$\theta_1$$과 $$theta_2$$를 (0.5, 0.5), (1, 0), (0, 1)로 하고 $$s_1$$과 $$s_2$$는 1로 두었다. x는 -1에서 1까지의 범위로 그렸다. x를 0으로 유지하려면 u(t)가 -와 +가 둘 다 가능해야할 것 같은데 위 식으로만 봐서는 action이 하나이고 그 action일 확률을 표시하는 것처럼 나왔다. 아마 -1과 +1이 u(t)가 될 수 있는데 그 중 +1을 선택할 확률이 위와 같이 되는게 아닌가 싶다.
<center><img src='https://www.dropbox.com/s/v69qyrwn7zurk8c/Screenshot%202018-06-08%2014.57.07.png?dl=1' width='500px'></center>

다음 그림은 1-d LQR을 학습한 그래프이다. cost가 $$x^2$$이기 때문에 cost가 0으로 갈수록 agent는 0에서 안정적으로 머무른다고 볼 수 있다. 6개의 선 중에서 오른쪽 세 개가 일반적인 gradient 방법을 사용해서 학습한 결과이다. 그리고 왼쪽의 세 개의 선이 natural policy gradient를 통해 학습한 학습 곡선이다. 일반 gradient 방법보다 natural gradient가 훨씬 빠르게 학습한다(time 축이 log scale인 것을 감안하자).

하지만 문제가 있다. npg를 학습한 세 개의 곡선은 $$\theta$$를 rescale 한 것이다. $$\theta$$앞에 곱해지는 숫자에 따라 학습의 과정이 다르다. 이 것은 coordinate에 따라 steepest gradient가 다르게 측정된다는 것이다. 즉, co-variant gradient가 아니라는 뜻이다. 이 논문에서는 natural gradient를 통해 gradient가 covariant하도록 만들고 싶었는데 실패한 것이다. 하지만 여전히 의의가 있는 것은 기존 gradient 방법들보다 훨씬 빠르게 학습한다는 것이다.


<center><img src="https://www.dropbox.com/s/jb6cyzn7613x4bs/Screenshot%202018-06-08%2014.43.13.png?dl=1" width="300px"></center>

### 7.2 Tetris
두 개의 실험 중에서 tetris만 살펴보려한다. tetrix는 linear function approximator와 greedy policy iteration을 사용할 경우 performance가 갑자기 떨어지는 현상이 있다. 밑의 그림에서 A의 spike가 있는 그래프가 이 경우이다. 그 밑에 낮게 누워있는 그래프는 일반적인 policy gradient 방법이다. 하지만 Natural policy gradient를 사용할 경우 B 그림에서 오른쪽 그래프와 같이 성능개선이 뚜렷하다. Policy Iteration 처럼 성능이 뚝 떨어지지 않고 안정적으로 유지한다. 또한 그림 C에서 보는 것처럼 오른쪽 그래프인 일반적인 gradient 방법보다 훨씬 빠르게 학습하는 것을 볼 수 있다.

<img src="https://www.dropbox.com/s/644zpk53bqn31o1/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-05-23%2015.26.46.png?raw=1">
