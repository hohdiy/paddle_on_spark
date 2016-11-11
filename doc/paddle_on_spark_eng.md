# Paddle On Spark： A distributed running environment for paddle
In this document, we will describe how paddle works on spark, including feature descriptions and code examples.
The upcoming launch is based on the spark 1.6, non-GPU paddle training, is expected to be released on late November.

In addition, we will provide a notebook environment(similar to jupyter), which is web-based、graphical interactive and easy to use.

## Features：
 1. Paddle is encapsulated according to the [pipeline] (http://spark.apache.org/docs/latest/ml-pipeline.html) interface defined in spark.ml, and the other spark machine learning algorithms are practically unused the difference.
 2. Paddle supports direct use of the DataFrame object spark as a training input, without prior conversion. So you can easily use spark for data processing and start paddle training directly.
 3. Paddle training process (trainer and pserver) can be directly in the spark cluster to start and perform distributed training tasks.
 4. Paddle training out of the model can also use the DataFrame object as a predictive input, distributed prediction, prediction results are DataFrame. This model can also be saved to the cluster.

All in all, you can put paddle as a spark.ml native machine learning algorithm to use, convenient and quick.

## Code Example：(Python code)
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

## Paddle Network Configs
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

