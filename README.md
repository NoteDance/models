# Introduction:
Neural network models built with TensorFlow.


# Installation:
Download models from https://github.com/NoteDance/models and then unzip it to the site-packages folder of your Python environment.


# Train:
```python
from models.ViT import ViT
vit=ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    pool='cls',
    channels=3,
    dim_head=64,
    drop_rate=0.1,
    emb_dropout=0.1
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer='adam',loss=loss_fn)
model.fit(x_train, y_train, epochs=5)
```


# Distributed training:
```python
from models.ViT import ViT

strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
    vit=ViT(
      image_size=224,
      patch_size=16,
      num_classes=1000,
      dim=768,
      depth=12,
      heads=12,
      mlp_dim=3072,
      pool='cls',
      channels=3,
      dim_head=64,
      drop_rate=0.1,
      emb_dropout=0.1
  )
  optimizer = tf.keras.optimizers.Adam()

def train_step(inputs):
  images, labels = inputs
  with tf.GradientTape() as tape:
    predictions = vit(images)
    loss = compute_loss(labels, predictions)
  gradients = tape.gradient(loss, vit.weights)
  optimizer.apply_gradients(zip(gradients, vit.weights))
  return loss

@tf.function(jit_compile=True)
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

for epoch in range(EPOCHS):
  total_loss = 0.0
  num_batches = 0
  for x in train_dist_dataset:
    total_loss += distributed_train_step(x)
    num_batches += 1
  train_loss = total_loss / num_batches

  template = ("Epoch {}, Loss: {}")
  print(template.format(epoch + 1, train_loss)
```


# Build models:
Here are some examples of building various neural networks, all in a similar way.

CLIP_large:
```python
from models.CLIP import CLIP
clip=CLIP(
    embed_dim=1024,
    image_resolution=224,
    vision_layers=14,
    vision_width=1024,
    vision_patch_size=32,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12
  )
```

DiT_B_4:
```python
from models.DiT import DiT_B_4
dit=DiT_B_4()
```

Llama2_7B:
```python
from models.Llama2 import Llama2
llama=Llama2()
```

ViT
```python
from models.ViT import ViT
vit=ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    pool='cls',
    channels=3,
    dim_head=64,
    drop_rate=0.1,
    emb_dropout=0.1
)
```


# Assign the trained parameters to the model:
The assign_param function allows you to assign trained parameters, such as downloaded pre-trained parameters, to the parameters of a neural network. These parameters should be stored in a list.
```python
from models.assign_param import assign_param
assign_param(model.weights,param)
```
