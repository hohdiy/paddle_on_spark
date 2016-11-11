# Paddle On Spark： 一种paddle的分布式运行环境
这篇文档中，我们将介绍paddle是如何on在spark上的，包含特点介绍和代码使用样例。
本次即将推出的是基于spark 1.6的，非GPU的paddle训练，预计是11月下旬发布。

另外透露一下，我们将在后续提供一个notebook环境（类似于jupyter），基于web的方式使用spark和paddle的能力，并且带有图形交互功能，更加便捷。

## 特点介绍：
 1. Paddle被按照spark.ml中定义的[pipeline](http://spark.apache.org/docs/latest/ml-pipeline.html)接口进行封装，和其他spark机器学习算法在使用上几乎没有区别。
 2. Paddle支持直接使用spark的DataFrame对象作为训练输入，无需事先进行格式转换。这样你就可以方便地利用spark进行数据处理并直接开始paddle训练。
 3. Paddle的训练进程（trainer和pserver）可以直接在spark集群上启动并执行分布式训练任务。
 4. Paddle训练出来的模型也可以使用DataFrame对象作为预测输入，进行分布式预测，预测结果也是DataFrame。这个模型也可以通过save函数保存到集群上。

总而言之，你可以把paddle当做spark.ml原生的机器学习算法使用，方便快捷。

## 代码样例：(Python code)
```python
# notebook language indicator.
%pyspark

# use paddle config lib.
from config_parser import *

# read ImageNet training data into DataFrame.
df = sqlContext.loadImageTable("hdfs://xxx-hdfs.baidu.com:54310/javis/ImageNet/2012-small/train", 224)

# split data into train and test.
train, test = df.randomSplit([0.7, 0.3], 0)

# define paddle training network configs. reference: http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/layers_index.html.
def configs():
    ......

# new a paddle
paddle = Paddle()

# make and set configs
config = parse_config(configs, '')
paddle.setConfig(config)

# set label and features.
paddle.setLabelCol("label")
paddle.setFeaturesCol("image")
paddle.setCluster(trainerNum = 10, pServerNum= 1)

# train model
model = paddle.fit(train)

# waiting some minutes ....

model.setPredictionCol("predict")
model.setFeaturesCol("image")

# predict
imageWithPredict = model.transform(test)
# show 10 lines
imageWithPredict.show(10)

# model save
model.save("hdfs://xxx-hdfs.baidu.com:54310/javis/ImageNet/cnn.model")
```

## Paddle 网络配置样例
```python
# define paddle training network configs.
def configs():
    Settings(
        algorithm='sgd',
        learning_method='momentum',
        batch_size=32,
        learning_rate=1,
        learning_rate_decay_a=0.5,
        learning_rate_decay_b=1500000 * 10,
        learning_rate_schedule="discexp",
        num_batches_per_send_parameter=1,
    )

    Inputs("label", "image")
    Outputs("cost")

    lr = 0.01 / 128.0
    dr = 0.0005 * 128.0

    Layer(
        name="image",
        type="data",
        size=150528,
    )

    Layer(
        name="conv",
        type="exconv",
        active_type="relu",
        bias=Bias(learning_rate=lr * 2,
                  momentum=0.9,
                  initial_mean=0,
                  initial_std=0.0),
        inputs=Input("image",
                     learning_rate=lr,
                     momentum=0.9,
                     decay_rate=dr,
                     initial_mean=0.0,
                     initial_std=0.059,
                     conv=Conv(filter_size=7,
                               channels=3,
                               padding=0,
                               stride=2,
                               groups=1)),
        num_filters=16,
        partial_sum=110 * 110,
        shared_biases=True
    )

    Layer(
        name="pool",
        type="pool",
        inputs=Input("conv",
                     pool=Pool(pool_type="max-projection",
                               channels=16,
                               size_x=32,
                               start=0,
                               stride=8))
    )

    Layer(
        name="fc",
        type="fc",
        active_type="relu",
        bias=Bias(learning_rate=lr * 2,
                  momentum=0.9,
                  initial_mean=0,
                  initial_std=0),
        inputs=Input("pool",
                     learning_rate=lr,
                     momentum=0.9,
                     decay_rate=dr,
                     initial_mean=0.0,
                     initial_std=0.01),
        size=16
    )

    Layer(
        name="output",
        type="fc",
        size=1000,
        active_type="softmax",
        bias=Bias(learning_rate=lr * 2,
                  momentum=0.9,
                  initial_mean=0,
                  initial_std=0),
        inputs=[
            Input("fc",
                  learning_rate=lr,
                  momentum=0.9,
                  decay_rate=dr,
                  initial_mean=0.0,
                  initial_std=0.001)],
    )

    Layer(
        name="label",
        type="data",
        size=1,
    )

    Layer(
        name="cost",
        type="multi-class-cross-entropy",
        inputs=["output", "label"],
    )
```

