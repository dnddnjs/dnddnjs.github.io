---
layout: post
title: "CIFAR-10 정복 시리즈 3: Shake-Drop"
subtitle: "Shake-Drop regularization"
categories: cifar10
tags: dl
comments: true
---

## CIFAR-10 정복하기 시리즈 소개
CIFAR-10 정복하기 시리즈에서는 딥러닝이 CIFAR-10 데이터셋에서 어떻게 성능을 높여왔는지 그 흐름을 알아본다. 또한 코드를 통해서 동작원리를 자세하게 깨닫고 실습해볼 것이다. 

- CIFAR-10 정복하기 시리즈 목차(클릭해서 바로 이동하기)
  - [CIFAR-10 정복 시리즈 0: 시작하기](https://dnddnjs.github.io/cifar10/2018/10/07/start_cifar10/)
  - [CIFAR-10 정복 시리즈 1: ResNet](https://dnddnjs.github.io/cifar10/2018/10/09/resnet/)
  - [CIFAR-10 정복 시리즈 2: PyramidNet](https://dnddnjs.github.io/cifar10/2018/10/24/pyramidnet/)
  - [CIFAR-10 정복 시리즈 3: Shake-Shake](https://dnddnjs.github.io/cifar10/2018/10/25/shake_shake/)

- 관련 코드 링크
  - [pytorch cifar10 github code](https://github.com/dnddnjs/pytorch-cifar10) 

<br>

## CIFAR-10 정복 시리즈 3: Shake-Drop
이전 포스트에서는 ResNet의 구조를 변형시킨 모델을 알아봤다. PyramidNet은 학습할 때 error rate가 거의 0이라고 볼 수 있다. 학습 error rate에 비해 테스트 error rate는 여전히 높기 때문에 regularization에 대해 생각해봐야한다. CIFAR은 학습 데이터양이 적은데 비해 네트워크의 representation power는 높다. 따라서 **overfit**이 일어나기 쉽다. CIFAR에서의 overfit 문제를 해결하고자 하는 것이 **Shake-Shake regularization**[^0]이다. Shake-Shake는 네트워크의 forward pass와 backward pass에서 noise를 주는 방식이다. 하지만 Shake-Shake는 **ResNeXt**[^1]의 구조의 네트워크에만 적용할 수 있다. 이 post에서는 Shake-Shake를 살펴보도록 하겠다. 

1. [ResNeXt](#resnext)
2. [FractalNet](#fractalnet)
3. [Shake-Shake](#shake-shake)
4. [Code Review](#code-review)
5. [Squeeze and Excitation](#squeeze-and-excitation)


<br/>

딥러닝에서 regularization은 overfit을 방지하기 위한 방법으로 많이 사용되고 있다. 그동안 사용되어왔던 regularization 효과를 가지는 방법들로는 weight decay, dropout, batch-normalization, SGD 등이 있다. PyramidNet 포스트에서 살펴봤던 ResDrop 또한 regularization에 해당한다. 네트워크 자체는 점점 강력해지지만 generalization 성능은 그만큼 따라오지 않기 때문에 이 이외에 추가적인 노력이 이어졌다. 기존 residual block는 2 branch로 구성되어있다. 한 branch는 idenity mapping이고 다른 branch는 nonlinear computation이 이뤄진다. **ResNeXt**는 이런 기본적인 구성을 벗어나서 2개의 branch 이상의 n개의 branch를 사용한다. **FractalNet**[^2]의 경우도 ResNeXt와 유사하게 여러 개의 subpath를 사용한다. FractalNet은 drop path라는 regularization 방법을 사용한다. **Shake-Shake**는 ResNeXt와 drop path를 적절히 합친 것이라고 볼 수 있다. 따라서 Shake-Shake를 살펴보기 전에 ResNeXt와 FractalNet을 간단히 살펴보겠다. 

<br>

## ResNeXt
ResNeXt는 기본적으로 **multi-branch ResNet이**라고 보면 된다. 기존에 residual block을 design할 때 activation의 순서를 바꿔보거나(pre-activation ResNet) 혹은 convolution의 filter 수를 변화시켰다(WideResNet, PyramidNet). 하지만 ResNeXt는 그 이외에 **cardinality**라는 개념을 소개한다. 다음 그림에서 왼쪽이 일반적인 residual block이다. 오른쪽이 ResNeXt의 residual block이다. Shortcut connection은 그대로 하나이지만 residual 부분이 여러개인 것을 볼 수 있다. Cardinality는 residual의 개수이다. 여러 path의 output은 summation으로 합친다.

<figure>
  <img src="https://www.dropbox.com/s/whhjpdfwzkcoahu/Screenshot%202018-11-23%2019.58.46.png?dl=1" width="400px">
  <figcaption>
    https://arxiv.org/pdf/1611.05431.pdf
  </figcaption>
</figure>

<br>

ResNeXt의 multi-branch는 GoogLeNet의 **Inception module**[^3]과 상당히 유사하다. 다음 그림의 Inception module이다. ResNeXt의 residual block은 Inception module과 다르게 각 path마다 모두 동일한 구조를 지니며 dimension이 모두 같다. Inception module은 hyper parameter가 많기 때문에 디자인하기 어렵다면 ResNeXt는 단순히 몇 개의 path를 사용하는지만 설정하기 때문에 상당히 간편하다. 

<figure>
  <img src="https://www.dropbox.com/s/frvr8g5vaojayw7/Screenshot%202018-11-23%2020.04.08.png?dl=1">
  <figcaption>
    https://arxiv.org/pdf/1409.4842.pdf
  </figcaption>
</figure>

<br>


## FractalNet
FractalNet은 Residual을 학습시키는 기존의 ResNet 변형체들과 다른 방식이다. FractalNet은 Residual을 학습하는 방식을 사용하지 않아도 네트워크를 깊게 쌓을 수 있다는 것을 보여준다. 아래 그림이 FractalNet의 fractal block의 구조를 보여준다. 가장 왼쪽은 fractal block을 형성하는 기본적인 방법을 보여준다. 보통 $$f_C$$로는 convolution을 사용하는데 확장할 때는 하나의 convolution이 오른쪽에 두 개로 합쳐진다. 그 다음 왼쪽에 다른 하나의 convolution을 붙이고 그 출력들을 join 연산을 통해 합친다. 이렇게 만든 fractal block은 가운데 그림과 같다. Residual block에서 볼 수 있는 residual과 identity mapping의 구조는 볼 수 없다. 

<figure>
  <img src="https://www.dropbox.com/s/kngx40hcgf8bg45/Screenshot%202018-11-23%2020.25.40.png?dl=1" width="500px">
  <figcaption>
    https://arxiv.org/pdf/1605.07648.pdf
  </figcaption>
</figure>

<br>

기존 residual block에서는 2 branch가 identity mapping과 residual learning 이라는 각자의 역할을 수행했다. 하지만 fractal block의 경우 여러 path가 존재하는데 서로 중복된 역할을 할 수 있다. Dropout이 이러한 **co-adaptation** 문제를 해결하려고 하나의 neuron 단위에 적용되었다. Fractal block에서는 **co-adapatation** 문제를 해결하기 위해 path를 drop 해버리는 drop path를 사용한다. Drop path의 작동하는 예시는 다음 그림과 같다. Drop path는 두 가지 방식으로 작동한다. Local 방식은 다음 그림에서 형광색에 해당하는 join layer에서 랜덤하게 인풋을 drop해버린다. Global 방식은 두 번째, 네 번째 그림에서 보듯이 전체 block 내부에서 하나의 path만 선택한다. 이렇게 path를 drop해버리는 것으로 regularization 효과를 볼 수 있다. 

<figure>
  <img src="https://www.dropbox.com/s/ednae8p9cag0hin/Screenshot%202018-11-23%2020.26.01.png?dl=1" width="500px">
  <figcaption>
    https://arxiv.org/pdf/1605.07648.pdf
  </figcaption>
</figure>

<br>

다음은 FractalNet의 실험결과이다. 20 layers에 38.6M 사이즈의 FractalNet을 보면 CIFAR-10에서 augmentation이 없을 경우 10.18 %의 error rate를 얻는 것을 볼 수 있다. 하지만 drop-path와 dropout을 사용할 경우 3% 정도의 성능이 향상된다. Data augmentation을 적용한 CIFAR-10에 대해서도 0.6 % 정도의 성능 향상을 볼 수 있다.

<figure>
  <img src="https://www.dropbox.com/s/2tqzbx4lxti64g5/Screenshot%202018-11-23%2023.53.07.png?dl=1">
  <figcaption>
    https://arxiv.org/pdf/1605.07648.pdf
  </figcaption>
</figure>

<br>

## Shake-Shake
Shake-Shake는 ResNeXt와 Drop-path를 합친 것이라고 볼 수 있다. ResNeXt에서 여러 branch의 output을 합칠 때 단순히 summation으로 합친다. 하지만 Shake-Shake에서는 stochastic affine transform을 통해서 합치겠다는 것이 아이디어이다. 다음 그림이 Shake-Shake의 작동 방식을 알려준다. ResNeXt의 경우 32개의 branch까지도 사용했는데 Shake-Shake에서는 2개의 branch만 사용한다. 이 2개의 branch를 사용해서 regularization 하는 것이 핵심이다. Shake-Shake는 forward pass에서 한 번, backward pass에서 한 번 **stochastic affine transform**을 수행한다. 이 affine transform은 일종의 **augmentation**이라고 볼 수 있다. 

<figure>
  <img src="https://www.dropbox.com/s/t2ijf2ahf5dkxa1/Screenshot%202018-10-13%2019.32.12.png?dl=1" width="500px">
  <figcaption>
    https://arxiv.org/pdf/1705.07485.pdf
  </figcaption>
</figure>

<br>
Shake-Shake에서 특정 block의 forward는 다음 수식과 같다. $$\alpha$$는 확률변수로서 0에서 1 사이의 랜덤한 숫자이다. 하나의 path는 $$\alpha$$를 곱하고 다른 하나의 path는 $$1-\alpha$$를 곱해서 더한 것이 residual이 된다. $$\alpha$$는 학습할 때 매 mini-batch마다 새로 뽑는다. Backward pass에서도 비슷하게 $$\beta$$라는 0에서 1 사이의 확률변수를 뽑아서 두 개의 다른 path로 가는 gradient에 그 값을 곱해준다. 위 그림에서 두 번째가 backward pass를 의미한다. 세 번째는 test 할 때를 말하는 것인데 test 할 때는 두 개의 path에 0.5씩 곱한 다음에 더한다. Shake-Shake는 drop path와 같이 하나의 path를 없애버리는 방식이 아니라 두 개의 path를 랜덤하게 섞어버리는 방식을 사용했다. 이러한 방식을 Shake-Shake 만의 novelty라고 볼 수 있다. 


$$x_{i+1} = x_i + \alpha_i F(x_i, W_i^{(1)}) + (1- \alpha_i )F(x_i, W_i^{(2)})$$

<br>

Forward pass와 backward pass에서는 각각 $$\alpha$$와 $$\beta$$라는 random number를 뽑아야한다. 이 때 새로운 $$\alpha, \beta$$를 뽑는 방법에는 여러 가지가 있다. Pass 할 때마다 새로운 random number를 뽑는 것을 "Shake"라고 하며 $$\alpha$$와 $$\beta$$를 따로 따로 pass마다 새로 뽑는 것을 **Shake-Shake**라고 한다. Shake-Shake 방식이 제일 성능이 좋기 때문에 논문의 이름이 Shake-Shake Regularization이 된 것이다. 다음 표는 여러가지 random number 추출 방식에 따른 성능 비교를 보여준다. Shake-Shake Image 방식이 **2.86%**로 가장 높은 성능을 달성한 것을 볼 수 있다. Level이라는 것이 있는데 Batch는 $$\alpha, \beta$$를 하나의 mini-batch 안에서 공유하겠다는 것이고 Image는 $$\alpha, \beta$$를 mini-batch안의 image마다 다르게 사용하겠다는 것을 뜻한다. 

<figure>
  <img src="https://www.dropbox.com/s/1cso45sjfpj7aap/Screenshot%202018-11-24%2000.17.41.png?dl=1" width="500px">
  <figcaption>
    https://arxiv.org/pdf/1705.07485.pdf
  </figcaption>
</figure>

Shake-Shake 모델은 3개의 stage를 가지는데 각 stage는 4개의 residual block으로 구성된다. 따라서 네트워크 전체의 깊이는 26이 된다. 위 표에서 Model 부분에 26 2x96d라고 써져있는데 이건 네트워크가 26층의 깊이를 가지며 2개의 branch를 사용하고 첫 residual block의 width가 96이라는 것을 의미한다. 2점대의 error rate라는 꽤나 인상적인 결과를 보여주는 Shake-Shake의 코드를 한 번 살펴보자.

<br>

## Code Review
코드에서는 Shake-Shake 26 2-32d 모델을 살펴볼 것이다. Shake-Shake의 residual block은 다음과 같다. 각 부분을 따로 살펴보겠다. 

~~~python
class ShakeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(ShakeBlock, self).__init__()
        self.residual_branch1 = ResidualBranch(in_channels, out_channels, stride)
        self.residual_branch2 = ResidualBranch(in_channels, out_channels, stride)

        if down_sample is not None:
            self.down_sample = SkippingBranch(in_channels, stride)
        else:
            self.down_sample = nn.Sequential()

        self.shake_shake = ShakeShake.apply

    def forward(self, x):
        residual = x
        out1 = self.residual_branch1(x)
        out2 = self.residual_branch2(x)
        
        batch_size = out1.size(0)
        if self.training:        
            alpha = torch.rand(batch_size).to(device)
            beta = torch.rand(batch_size).to(device)
            beta = beta.view(batch_size, 1, 1, 1)
            alpha = alpha.view(batch_size, 1, 1, 1)
            out = self.shake_shake(out1, out2, alpha, beta)
        else:
            alpha = torch.Tensor([0.5]).to(device)
            out = self.shake_shake(out1, out2, alpha)

        skip = self.down_sample(residual)
        return out + skip
~~~

<br>

Residual branch는 따로 class로 정의를 해놓았다. 각각의 branch는 self.residual_branch1과 self.residual_branch2로 정의한다. 

~~~python
self.residual_branch1 = ResidualBranch(in_channels, out_channels, stride)
self.residual_branch2 = ResidualBranch(in_channels, out_channels, stride)

~~~

<br>
residual branch의 코드는 다음과 같다. 일반적인 Residual block에서 residual branch에 해당하는 부분만 들어있다. Residual branch는 ReLU-Conv3x3-BN-ReLU-Conv3x3-BN-Mul 으로 구성된다. 

~~~python
class ResidualBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
    
        x = self.relu(x)
        x = self.conv2(x)
        out = self.bn2(x)
        return out

~~~

<br>
다시 ShakeBlock으로 돌아간다. Residual Branch를 통해 정의된 residual branch들은 각각 out1, out2의 출력을 낸다. nn.Module을 상속할 경우 self.training을 통해 학습 중인지 아닌지를 알아낼 수 있다. 만약 학습 중이라면 self.training이 True가 되고 이 때 alpha와 beta를 image 단위로 랜덤하게 뽑아야한다. torch.rand라는 함수를 사용해서 alpha와 beta를 sampling한 다음에 feature dimension에 맞춰준다. out1과 out2에 alpha를 적용하는 함수가 self.shake_shake이며 custom module로 따로 정의되어있다. 이 때 beta도 함께 인자로 넣어주는데 pytorch에서 forward pass에서의 값을 저장해놓고 backpropagation을 하기 때문에 forward pass에서 beta의 정보를 넣어줘야한다. self.shake_shake에서 반환된 out은 shortcut과 더힌다.

~~~python
def forward(self, x):
    shortcut = x
    out1 = self.residual_branch1(x)
    out2 = self.residual_branch2(x)
    
    batch_size = out1.size(0)
    if self.training:        
        alpha = torch.rand(batch_size).to(device)
        beta = torch.rand(batch_size).to(device)
        beta = beta.view(batch_size, 1, 1, 1)
        alpha = alpha.view(batch_size, 1, 1, 1)
        out = self.shake_shake(out1, out2, alpha, beta)
    else:
        alpha = torch.Tensor([0.5]).to(device)
        out = self.shake_shake(out1, out2, alpha)

    shortcut = self.down_sample(shortcut)
    return out + shortcut
~~~

<br>

self.shake_shake는 ShakeShake라는 클래스를 통해 정의된다. ShakeShake는 Shake-Shake 코드의 핵심이라고 할 수 있다. 이 코드를 작성할 때 pytorch tutorial[^4]과 pytorch discuss[^5]를 참고했다. 원래 backpropagation 할 때는 forward pass에서 곱해졌던 상수값을 기억해서 gradient에 곱해준다. 하지만 Shake-Shake에서는 forward pass와 backward pass에서 다른 상수값을 사용하기 때문에 이와 같이 custom을 해야 한다. ctx.save_for_backward에 인자로 넣으면 backward 할 때 그 값들을 호출할 수 있다. backward 함수에서 아까 저장했던 tensor를 불러온다. 불러온 $$\beta$$값을 각각의 branch로 내려가는 두 개의 gradient에 곱해준다. 한 gradient에는 $$\beta$$를 곱하고 한 branch에는 $$1 - \beta$$를 곱해준다. 


~~~python
class ShakeShake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, alpha, beta=None):
        ctx.save_for_backward(input1, input2, alpha, beta)
        out = alpha * input1 + (1 - alpha) * input2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, alpha, beta = ctx.saved_tensors
        grad_input1 = beta * grad_output
        grad_input2 = (1 - beta) * grad_output
        return grad_input1, grad_input2, None, None
~~~

<br>

ShakeBlock의 forward pass에서 shortcut을 downsampling 하는데 보통 downsample의 방식과는 다르다. 보통 residual block에서 shortcut connection을 down sample 할 때 feature map의 사이즈를 반으로 줄이고 channel 수를 2배로 늘린다. ResNet에서는 max-pooling으로 사이즈를 반으로 줄이고 zero-padding으로 channel 수를 늘렸다. Shake-Shake에서는 특이하게도 입력을 2개의 branch를 만들어서 channel 방향으로 concatenate 한다. 첫 번째 branch는 들어온 입력을 average pooling한 이후에 1x1 convolution을 통과시킨다. 두 번째 branch에서는 입력을 왼쪽 위로 1 step만큼 shift한 이후에 padding을 통해 원래 입력의 feature map size를 유지한다. 그 이후에 1x1 convolution을 통과시킨다. 두 branch의 output인 out1과 out2를 channel 방향으로 concatenate 한다. 

~~~python
class SkippingBranch(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(SkippingBranch, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=stride, 
                   padding=0)  
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, 
                 stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, 
                 stride=1, padding=0, bias=False)
  

    def forward(self, x):
        out1 = self.avg_pool(x)
        out1 = self.conv1(out1)
    
        shift_x = x[:, :, 1:, 1:]
        shift_x= F.pad(shift_x, (0, 1, 0, 1))
    
        out2 = self.avg_pool(shift_x)
        out2 = self.conv2(out2)
    
        out = torch.cat([out1, out2], dim=1)
        return out
~~~

<br>

Shake-Shake 네트워크의 전체 구조는 ShakeResNet에 정의되어 있다. ResNet의 전체 네트워크 코드와 동일하다.
~~~python
class ShakeResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10):
        super(ShakeResNet, self).__init__()
        self.in_channels = 16
        self.num_layers= num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                 stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # feature map size = 32x32x32
        self.stage1 = self.get_layers(block, 16, 32, stride=1)
        # feature map size = 32x32x64
        self.stage2 = self.get_layers(block, 32, 64, stride=2)
        # feature map size = 32x32x128
        self.stage3 = self.get_layers(block, 64, 128, stride=2)
    
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def get_layers(self, block, in_channels, out_channels, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False
    
        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, stride, down_sample)])
      
        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x
~~~

<br>

일반적인 ResNet과 또 다른 점은 학습 epoch 수이다. Shake-Shake는 forward pass와 backward pass에 일종의 노이즈를 주입하기 때문에 regularization 효과를 보는 대신 학습이 느려진다. 따라서 기존 ResNet과 같이 일정 update step마다 learning rate를 0.1배 하는 것은 맞지 않다. 대신 **cosine annealing**[^6]을 사용한다. Cosine annealing은 learning rate를 cosine 함수의 형태로 decay 하겠다는 것이다. 다음 그림이 cosine annealing에서 learning rate가 iteration에 따라 어떻게 감소하는지를 보여준다. 처음 몇 epoch 동안은 높은 learning rate로 빠르게 local minimum을 찾고 그 이후 learning rate를 decay하면서 minimum에 가까이 다가가고 마지막 epcoh 동안에는 천천히 움직이다가 학습을 마무리한다. 

<figure>
  <img src="https://www.dropbox.com/s/6rcn2w05zye3yhz/Screenshot%202018-11-26%2000.13.33.png?dl=1" width="400px">
  <figcaption>
    https://towardsdatascience.com/https-medium-com-reina-wang-tw-stochastic-gradient-descent-with-restarts-5f511975163
  </figcaption>
</figure>

<br>

Cosine annealing은 코드로 다음과 같이 구현할 수 있다. PyTorch의 lr_scheduler에서 custom learning rate scheduling을 할 수 있는 LambdaLR을 사용한다. 결국 cosin_annealing 함수를 호출하는 것이다. 이 함수에서는 lr_max에서 lr_min 까지 decay하는 함수의 형태를 정의하고 있다. Shake-Shake에서 첫 learning rate 곧 lr_max는 0.2이고 annealing을 1800 epoch 동안 수행한다. 

~~~python
def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


def cosine_annealing_scheduler(optimizer, epochs, lr):
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            epochs,
            lr,  # since lr_lambda computes multiplicative factor
            0))

    return scheduler
~~~

<br>

Shake-Shake 네트워크의 학습 과정은 다음과 같다. 다른 네트워크에 비해 1800 epoch을 학습하기 때문에 학습이 오래 걸린다. 

<img src="https://www.dropbox.com/s/llsful1i8g03qg4/Screenshot%202018-11-26%2000.22.14.png?dl=1">

<br>

## Squeeze and Excitation


<br> 

### 참고문헌
[^0]: https://arxiv.org/pdf/1705.07485.pdf
[^1]: https://arxiv.org/pdf/1611.05431.pdf
[^2]: https://arxiv.org/pdf/1605.07648.pdf
[^3]: https://arxiv.org/pdf/1409.4842.pdf
[^4]: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
[^5]: https://discuss.pytorch.org/t/why-input-is-tensor-in-the-forward-function-when-extending-torch-autograd/9039
[^6]: https://arxiv.org/pdf/1608.03983.pdf
