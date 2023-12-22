# Federated-Learning-Frameworks-Comparison

"**FederatedLearningFrameworksComparison**" 是一个专注于对比和分析联邦学习的两大开源框架——Flower和TensorFlow Federated (TFF)的调研项目。它包括详尽的文档研究、框架特性评估、以及实际Demo测试。通过深入探讨每个框架的设计哲学、性能表现和易用性，本项目旨在提供一个全面的视角，帮助研究者和开发者了解这些先进工具在处理数据安全和隐私问题时的差异和优势。

## 调研背景

在当今数字化时代，数据和机器学习的应用无处不在，从智能医疗到个性化推荐系统。然而，这种广泛的数据利用引发了对个人隐私的严重关切。为解决这一问题，联邦学习应运而生，它作为隐私计算的一种重要形式，允许在不共享用户原始数据的情况下，跨多个设备和平台进行模型训练。

本次调研将围绕两个主要的开源联邦学习框架：**Flower 和 TensorFlow Federated (TFF) **展开。

### 联邦学习

联邦学习是一种革命性的机器学习方法，它允许不同设备和组织在保持数据隐私的前提下共同训练模型。这种方法的核心在于模型的训练发生在本地，只有模型更新（而非原始数据）被发送到中心服务器进行聚合。这样不仅保护了数据隐私，还减少了数据传输的需要，提高了效率。

联邦学习具有下面这些特性：

+ **多方参与**：两个或两个以上的参与方合作构建共享的机器学习模型，每个参与方都拥有一部分数据用以训练模型

+ **不交换数据**：在联邦学习训练过程中，任意一个参与方的任意原始数据不会离开该参与方，不会被直接交换和收集

+ **保护传输信息**：在联邦学习训练过程中，训练所需的信息需要经过保护后在个参与方之间进行传输，使得各参与方无法基于传输的信息推测出其他参与方的数据

+ **近似无损**：模型的量能要充分接近理想模型（即各参与方通过

  直接合并数据训练得到的模型）的性能

我们可以将联邦学习分为三种主要类型：**横向联邦学习**、**纵向联邦学习**以及**联邦迁移学习**

#### 横向联邦学习

**本次调研运行的Demo均为横向联邦学习框架**

这里简要介绍一下横向联邦学习

横向联邦学习适用于数据持有者拥有相似特征但不同样本的情况，例如不同用户的手机数据

![img](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/a9d03be053ee240b0ccb63184db9fcda.png)

横向联邦学习以数据的特征维度为导向，取出参与方特征相同而用户不完全相同的部分进行联合训练。在此过程中，通过各参与方之间的样本联合，**扩大了训练的样本空间，从而提升了模型的准确度和泛化能力**。

当然，想要在普通的模型测试中**模拟横向联邦非常简单**，我们只需将同一数据集分割成几份，作为不同的客户端上的数据即可

#### 客户-服务器架构

横向和纵向是从参与方拥有的数据类型出发，对联邦学习进行分类

我们还需要关注具体架构

联邦学习有两种常用的架构：**客户-服务器架构**以及**对等网络架构**

**客户-服务器架构**也被称为主-从（master-worker）架构或者轮辐式（hub-and-spoke）架构。在这种系统中，具有同样数据结构的 K 个参与方（也叫作客户或用户）在服务器（也叫作参数服务器或者聚合服务器）的帮助下，协作地训练一个机器学习模型

![2c025f8a64564cdbabacbfe4277f5469](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/2c025f8a64564cdbabacbfe4277f5469.png)

**步骤1**：各参与方在本地计算模型梯度，并使用同态加密、差分隐私或秘密共享等加密技术，对梯度信息进行掩饰，并将掩饰后的结果（简称为加密梯度） 发送给聚合服务器。

**步骤2**：服务器进行安全聚合（secure aggregation）操作，如使用基于同态加密的加权平均。

**步骤3**：服务器将聚合后的结果发送给各参与方。

**步骤4**：各参与方对收到的梯度进行解密，并使用解密后的梯度结果更新各自的模型参数。

当然除了聚合梯度，服务器端**也可以对收到的模型参数进行聚合**

![image-20231220143615042](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231220143615042.png)

#### 对等网络架构

对等网络架构中不存在中央服务器或协调方

![image-20231220143953129](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231220143953129.png)

每一个训练方负责只使用本地数据训练同一个机器学习模型（如DNN）

训练方使用安全链路（channels）在相互之间传输模型参数信息

### 开源框架简介

#### Flower

<img src="https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/2023-12-22-flower.webp" alt="2023-12-22-flower" style="zoom: 25%;" />

​	

Flower 是一个轻量的联邦学习框架，提出于 2020 年。一直以来，因为设计良好，方便扩展受到了比较多的关注。

框架设计主要追求下面目标：

1. 可拓展，支持大量的客户端同时进行模型训练；
2. 使用灵活，支持异构的客户端，通信协议，隐私策略，支持新功能的开销小；

**官网**：[Flower Framework main](https://flower.dev/docs/framework/index.html)

**github源码链接**：[adap/flower: Flower: A Friendly Federated Learning Framework (github.com)](https://github.com/adap/flower)

当然，之所以选择调研flower，也是因为它的官方教程较为完善、社区活跃。

#### TFF



<img src="https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/tf-logo-card-16x9.png" alt="img" style="zoom:25%;" />

TensorFlow Federated（TFF）是一个开源框架，用于对分散数据实验机器学习和其他计算。TensorFlow Federated使开发人员能够表达和模拟联邦学习系统。

**官网**：[TensorFlow Federated (google.cn)](https://tensorflow.google.cn/federated?hl=zh-cn)

**github源码链接**：[tensorflow/federated: A framework for implementing federated learning (github.com)](https://github.com/tensorflow/federated)

## 框架分析

我们先从两个框架的官方文档以及论文入手，分析这些框架的特点以及优势。当然，**更多特性需要我们后续运行Demo之后才能观察到**。

### Flower

想要深入了解Flower这个开源框架，还是需要我们自己去阅读背后的论文

***FLOWER: A FRIENDLY FEDERATED LEARNING FRAMEWORK***

论文链接如下：

[2007.14390.pdf (arxiv.org)](https://arxiv.org/pdf/2007.14390.pdf)

从论文的摘要部分我们可以了解到，Flower框架的**核心优势在于它支持大规模的联合学习实验**，这对于在现实世界条件下，尤其是在设备性能和网络连接存在巨大差异的情况下，进行研究和实验至关重要。这种框架**能够处理高达数百万级别的客户端**，这对于研究和开发具有深远的意义，因为它允许研究者在真实世界的复杂环境中测试和优化他们的FL算法。此外，它还提供了一个平滑的过渡路径，使研究者能够从大规模模拟实验轻松迁移到真实设备上的实际应用。这表明Flower框架非常注重实用性和灵活性，适合在多种环境下进行FL研究和实践。

Flower框架的设计理念强调了在实际环境中，特别是在边缘计算环境中实施联合学习的挑战。与其他集中在云计算或数据中心的框架不同，**Flower更加注重边缘设备的异构性和环境因素**。这种方法能更好地反映真实世界中设备的多样性和网络条件的不确定性，使FL算法更加健壮和实用。

![image-20231220161913275](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231220161913275.png)

Flower框架**采用客户端-服务器架构**，支持不同类型的设备和操作系统。它提供了高度模块化的设计，允许在不同级别上定制FL算法。Flower还支持跨平台部署，包括移动设备和嵌入式系统，当然**这些内容无法在本次调研中测试到**。

### FFT

谷歌作为联邦学习的提出者，在其深度学习框架TensorFlow的基础上开发出了一套联邦学习的框架Tensorflow Federated（TFF）

所以与Flower框架相比，TFF**对tensorflow深度学习框架的支持性更好，同时也更具专业性**

![image-20231220164731851](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231220164731851.png)

TensorFlow Federated使开发人员能够表达和模拟联邦学习系统。如图所示，每个手机在本地训练模型（A），将它们的更新汇总到（B），然后形成改进的共享模型（C）。

在浏览官方网站与文档之后，我也体会到了下面几点优势

1. **灵活性和扩展性**：TFF 提供灵活的 API，支持广泛的机器学习模型和算法，易于扩展和自定义。
2. **与 TensorFlow 的集成**：TFF 与 TensorFlow 紧密集成，使得开发者可以轻松利用 TensorFlow 的功能和生态系统。
3. **函数式编程模型**：提高代码的抽象度和复用性。
4. **灵活的模型和数据处理**：支持复杂的机器学习模型和数据处理流程。

TFF同样是**采用客户端-服务器架构**，专注于保持数据的隐私性和局部性。

而且，TFF与原生的python库不同，它有许多自己定义的操作和数据结构

例如在定义函数前，我们可以使用`@tf.function`装饰器，将普通的Python函数转换为TensorFlow图（Graph）操作。这种转换有以下几个优点：

1. **性能提升**：图执行允许TensorFlow进行更多的优化，例如并行化和分布式执行，从而提高运行效率。
2. **可移植性**：图操作可以跨不同的平台和硬件（如CPU、GPU、TPU）无缝运行，增强了代码的可移植性。
3. **自动微分支持**：使用图可以更容易地利用TensorFlow的自动微分（autodiff）功能，这对于训练深度学习模型尤其重要。

当然，我们也可以通过阅读TFF背后的论文来深入了解它

论文链接如下：

[1902.01046.pdf (arxiv.org)](https://arxiv.org/pdf/1902.01046.pdf)

![image-20231222084543366](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231222084543366.png)

论文中的这张图可以算是google提出的联邦学习技术的核心：图中展示了联邦学习服务器架构中的不同参与者。流程开始于协调者（Coordinator），它负责整个过程的协调。协调者创建主聚合器，它是一个长期存在的参与者，用于创建和管理聚合器。聚合器是短暂的参与者，它们负责从选择器（Selector）处接收设备的模型更新。选择器是连接设备的接口，负责选择哪些设备参与本轮学习。整个架构设计为循环流程，以支持持续的模型训练和更新。

## 环境搭建

在分析代码和运行教程Demo前，我们需要分别搭建这两个开源框架的环境

在本次调研中，为了对比不同开源框架的效率，我们需要选用相同的深度学习框架、神经网络模型以及数据集

**深度学习框架我们选用Tensorflow**

### Flower

首先我们使用conda包管理创建一个测试用的环境，并安装Tensorflow框架

```shell
conda create -n Flower python=3.8
conda activate Flower
pip install tensorflow
```

使用以下指令为环境安装flower

``` shell
pip install flwr
pip install flwr_datasets
```

最后可以执行`pip list`来验证安装完整性

![image-20231220172807606](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231220172807606.png)



### TFF

同样，我们使用conda包管理创建一个测试用的环境

```shell
conda create -n TFF python=3.7
conda activate TFF
```

需要注意的是，tensorflow的版本需要依靠TFF的版本来指定，且安装TFF时会自动安装对于的tensorflow，所以可以先不装tensorflow

**由于Demo中使用cpu来跑，所以并不需要太在意tensorflow与你的CUDA版本对应问题**

使用以下指令为环境安装TFF

``` shell
pip install tensorflow-federated==0.13.1
```

这里去掉了官方教程指令中的`--quiet`选项，目的是为了更清晰地观察安装进度，如果安装中途出错更好定位原因

![image-20231220210309912](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231220210309912.png)

## 代码剖析及改进

两个开源框架均选择**训练MNIST数据集**

### Flower

Flower的测试代码非常简洁和直观，同时也能**更好地反应服务端和客户端之间的通信和交互**

客户端的代码为`Flower_client.py`

首先加载模型和数据集

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
```

接着是训练过程

``` python
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=30, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}

```

包括了从服务端获取并设置参数、训练以及评估模型等过程

最后则是**连接服务器端口**的部分

``` python
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())
```

因为给出了地址，所以能够非常好地模拟实际场景中客户端与服务端的通信

服务端的代码为`Flower_server.py`

```
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
```

只有非常简短的几行，可见**Flower框架的API高度集成**，搭建联邦学习时非常方便

其中比较重要的部分即权值平均

那么由于原Demo跑的数据集为cifar10

我们需要**修改客户端代码，使其训练mnist**

``` python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

同时我们还需要修改模型，**使用与TFF的Demo相同的模型结构**

``` python
model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### TFF

TFF的教程Demo我选择的是《自定义联合算法第2部分-实现联合平均算法》

教程地址：[自定义联合算法，第 2 部分：实现联合平均  | TensorFlow Federated (google.cn)](https://tensorflow.google.cn/federated/tutorials/custom_federated_algorithms_2?hl=zh-cn)

首先是对一些库函数的引入，这里要注意的是，后续使用到TFF库时都会使用缩写`tff`

``` python
import tensorflow_federated as tff
```

导入用于训练的MNIST数据集

``` python
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
```

我们将客户端数量定义为2，保持其与Flower的Demo中的客户端个数相同

``` python
NUM_EXAMPLES_PER_USER = 2
```

接下来使用`get_data_for_digit()`方法将原MNIST数据集按客户端数量分为`NUM_EXAMPLES_PER_USER`份

``` python
def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence
```

每一份为由字典组成的列表，**字典的键为图像数组，值为对应label**

在定义损失函数之前，我们还需要定义输入和输出的类型，即**TFF命名元组**

与 Python 不同，针对类似元组的容器，TFF 具有单个抽象类型构造函数 `tff.StructType`。命名元组在TFF中广泛用于表示模型的参数、输入输出数据格式以及在联合计算中传递的数据。每个元素的名称提供了额外的语义信息，有助于理解和维护复杂的联合学习算法。

然后是损失函数部分

``` python
@tf.function
def forward_pass(model, batch):
    predicted_y = tf.nn.softmax(
        tf.matmul(batch['x'], model['weights']) + model['bias'])
    return -tf.reduce_mean(
        tf.reduce_sum(
            tf.one_hot(batch['y'], 10) * tf.math.log(predicted_y), axis=[1]))

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    return forward_pass(model, batch)
```

首先，模型的输出通过`tf.matmul`（矩阵乘法）和加上偏差后，使用`tf.nn.softmax`函数进行softmax处理。这一步是为了将模型输出转换为概率分布。损失函数使用的是交叉熵损失，这在多类分类问题中很常见。具体来说，它计算了实际标签的one-hot编码和预测概率分布之间的交叉熵。最后通过`tf.reduce_mean`计算所有样本的平均损失值。

初始化参数部分与普通的机器学习模型类似，这里就跳过分析

模型训练函数如下

``` python
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`. Must
    # be defined outside the @tf.function.
    model_vars = collections.OrderedDict([
      (name, tf.Variable(name=name, initial_value=value))
      for name, value in initial_model.items()
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function
    def _train_on_batch(model_vars, batch):
        # Perform one step of gradient descent using loss from `batch_loss`.
        with tf.GradientTape() as tape:
            loss = forward_pass(model_vars, batch)
        grads = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(
            zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
        return model_vars

    return _train_on_batch(model_vars, batch)
```

该模型使用keras库中的**随机梯度下降（SGD）优化器**

函数`_train_on_batch`被定义为一个`@tf.function`，这样可以让TensorFlow优化执行效率。这个函数执行模型的一次前向传递计算损失（使用`forward_pass`），然后使用梯度带（`tf.GradientTape`）计算梯度，并应用这些梯度来更新模型变量。

有了模型训练函数，我们就可以开始模拟在客户端本地的训练了

``` python
def local_train(initial_model, learning_rate, all_batches):

    # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    return tff.sequence_reduce(all_batches, initial_model, batch_fn)
```

模型初始化以及训练模块均使用刚刚定义的`initial_model`和`batch_train`函数

以及本地评估函数：

``` python
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    # TODO(b/120157713): Replace with `tff.sequence_average()` once implemented.
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))
```

接着我们需要实现联合训练，实现联合训练的最简单方法是进行本地训练，然后对模型进行平均。

``` python
SERVER_FLOAT_TYPE = tff.type_at_server(tf.float32)


@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE,
                           CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
  return tff.federated_mean(
      tff.federated_map(local_train, [
          tff.federated_broadcast(model),
          tff.federated_broadcast(learning_rate), data
      ]))
```

它接受三个参数：服务器上的模型（`SERVER_MODEL_TYPE`），服务器上的浮点数（学习率，`SERVER_FLOAT_TYPE`），以及客户端的数据类型（`CLIENT_DATA_TYPE`）。

函数返回联合平均的结果。`tff.federated_map`将`local_train`函数应用于每个客户端，`tff.federated_broadcast`用于将模型和学习率从服务器广播到各个客户端。然后，这些局部训练的结果被收集起来，并通过`tff.federated_mean`计算它们的平均值，从而完成一次联合训练迭代。

当然，我们可以像下面的原版教程一样，使用损失值来评估最后的效果

``` python
model = initial_model
learning_rate = 0.1
for round_num in range(5):
  model = federated_train(model, learning_rate, federated_train_data)
  learning_rate = learning_rate * 0.9
  loss = federated_eval(model, federated_train_data)
  print('round {}, loss={}'.format(round_num, loss))
```

重点来了，我们**想要和Flower框架的Demo一样输出准确率**，方便进行对比

新增一个计算准确率的函数`compute_accuracy`

``` python
def compute_accuracy(model, federated_data):
    total_accuracy = 0.0
    total_samples = 0

    for client_data in federated_data:
        for batch in client_data:
            x, y = batch['x'], batch['y']
            predicted_y = tf.nn.softmax(tf.matmul(x, model['weights']) + model['bias'])
            correct_prediction = tf.equal(tf.argmax(predicted_y, axis=1), y)
            batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            total_accuracy += batch_accuracy.numpy() * len(y)
            total_samples += len(y)

    return total_accuracy / total_samples

```

这样我们可以通过获取相同epoch下两个框架的准确率和损失，来对比训练效率

## 运行Demo

### Flower

首先启动服务端

``` shell
python Flower_server.py
```

重新开一个进程，运行如下指令启动客户端0

``` shell
python Flower_client.py
```

同样，等待几秒后启动客户端1

``` shell
python Flower_client.py
```

服务端显示如下![image-20231222132012014](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231222132012014.png)

客户端显示如下

![image-20231222131934588](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231222131934588.png)

如下图所示，最后的准确率达到了`0.8521`

![image-20231222132103889](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231222132103889.png)

### TFF

安装Demo运行所需特定版本的依赖库

``` shell
pip install protobuf==3.19.0
```

执行如下指令

``` shell
python TFF_test.py
```

运行结果如下

![image-20231222132310148](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231222132310148.png)

可以看到，30轮后准确率达到了`0.6843`

## 分析与总结

下面是实验结果的对比表格

| 特性/框架      | Flower 实验结果  | TFF 实验结果     |
| -------------- | ---------------- | ---------------- |
| **客户端数量** | 2                | 2                |
| **训练数据集** | MNIST            | MNIST            |
| **模型结构**   | 简单的多层感知机 | 简单的多层感知机 |
| **准确率**     | 0.8521           | 0.6843           |
| **训练轮次**   | 30轮             | 30轮             |

通过这些结果，我们可以看到在相同条件下，Flower框架在这次实验中的准确率表现优于TFF。这可能归因于Flower在客户端管理和模型更新策略上的不同实现。当然，这样的实验结果并不足以全面评估两个框架的性能，还需要考虑实验的详细设置和参数调整等因素。

从Demo的形式上来讲，Flower的Demo更能模拟现实场景的联邦学习中服务端与客户端的交互。

总结来说，Flower框架在易用性、扩展性和表现突出，而TFF则在与TensorFlow的集成度、专业性以及对复杂模型的处理能力方面有优势。

下面是对两个框架的特性的总结表格

| 框架           | Flower                        | TFF                                   |
| -------------- | ----------------------------- | ------------------------------------- |
| **设计目标**   | 大规模联邦学习实验            | 与TensorFlow紧密集成的联邦学习        |
| **易用性**     | 高度集成的API，易于搭建和开始 | 需要熟悉TensorFlow生态系统            |
| **扩展性**     | 支持大量客户端，适合边缘设备  | 灵活的API，适合多种机器学习模型和算法 |
| **专业性**     | 强调实用性和灵活性            | 提供高度专业化的联邦学习元素          |
| **性能**       | 面向实际环境的优化            | 优化了复杂数据处理和模型训练          |
| **社区和支持** | 活跃的社区和完善的官方文档    | 谷歌支持和广泛的技术资源              |