# Finetuning

This is an example on fine-tuning Gemma. For an example on how to run a pre-trained Gemma model, see the [sampling](https://gemma-llm.readthedocs.io/en/latest/sampling.html) tutorial.

To fine-tune Gemma, we use the [kauldron](https://kauldron.readthedocs.io/en/latest/) library which abstract most of the boilerplate (checkpoint management, training loop, evaluation, metric reporting, sharding,…).

```
!pip install -q gemma
```

```
# Common imports
import os
import optax
import treescope

# Gemma imports
from kauldron import kd
from gemma import gm
```

By default, Jax do not utilize the full GPU memory, but this can be overwritten. See [GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html):

```
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
```

## Data pipeline

First create the tokenizer, as it’s required in the data pipeline.

```
tokenizer = gm.text.Gemma3Tokenizer()

tokenizer.encode('This is an example sentence', add_bos=True)
```

```
[<_Gemma2SpecialTokens.BOS: 2>, 1596, 603, 671, 3287, 13060]
```

First we need a data pipeline. Multiple pipelines are supported including:

- [HuggingFace](https://kauldron.readthedocs.io/en/latest/api/kd/data/py/HuggingFace.html)
    
- [TFDS](https://kauldron.readthedocs.io/en/latest/api/kd/data/py/Tfds.html)
    
- [Json](https://kauldron.readthedocs.io/en/latest/api/kd/data/py/Json.html)
    
- …
    

It’s quite simple to add your own data, or to create mixtures from multiple sources. See the [pipeline documentation](https://kauldron.readthedocs.io/en/latest/data_py.html).

We use `transforms` to customize the data pipeline, this includes:

- Tokenizing the inputs (with [`gm.data.Tokenize`](https://gemma-llm.readthedocs.io/en/latest/api/gm/data/Tokenize.html))
    
- Creating the model inputs (with [`gm.data.Tokenize`](https://gemma-llm.readthedocs.io/en/latest/api/gm/data/Tokenize.html)))
    
- Adding padding (with [`gm.data.Pad`](https://gemma-llm.readthedocs.io/en/latest/api/gm/data/Pad.html)) (required to batch inputs with different lengths)
    

Note that in practice, you can combine multiple transforms into a higher level transform. See the `gm.data.ContrastiveTask()` transform in the [DPO example](https://github.com/google-deepmind/gemma/tree/main/examples/dpo.py) for an example.

Here, we try [mtnt](https://www.tensorflow.org/datasets/catalog/mtnt), a small translation dataset. The dataset structure is `{'src': ..., 'dst': ...}`.

```
ds = kd.data.py.Tfds(
    name='mtnt/en-fr',
    split='train',
    shuffle=True,
    batch_size=8,
    transforms=[
        # Create the model inputs/targets/loss_mask.
        gm.data.Seq2SeqTask(
            # Select which field from the dataset to use.
            # https://www.tensorflow.org/datasets/catalog/mtnt
            in_prompt='src',
            in_response='dst',
            # Output batch is {'input': ..., 'target': ..., 'loss_mask': ...}
            out_input='input',
            out_target='target',
            out_target_mask='loss_mask',
            tokenizer=tokenizer,
            # Padding parameters
            max_length=200,
            truncate=True,
        ),
    ],
)

ex = ds[0]

treescope.show(ex)
```

```
Disabling pygrain multi-processing (unsupported in colab).
{
    'input': i64[8 200],
    'loss_mask': bool_[8 200 1],
    'target': i64[8 200 1],
}
```

We can decode an example from the batch to inspect the model input. We see that the `<start_of_turn>` / `<end_of_turn>` where correctly added to follow Gemma dialog format.

```
text = tokenizer.decode(ex['input'][0])

print(text)
```

```
<start_of_turn>user
Would love any other tips from anyone, but specially from someone who’s been where I’m at.<end_of_turn>
<start_of_turn>model
J'apprécierais vraiment d'autres astuces, mais particulièrement par quelqu'un qui était était déjà là où je me trouve.
```

## Trainer

The [kauldron](https://kauldron.readthedocs.io/en/latest/) trainer allow to train Gemma simply by providing a dataset, a model, a loss and an optimizer.

Dataset, model and losses are connected together through a `key` strings system. For more information, see the [key documentation](https://kauldron.readthedocs.io/en/latest/intro.html#keys-and-context).

Each key starts by a registered prefix. Common prefixes includes:

- `batch`: The output of the dataset (after all transformations). Here our batch is `{'input': ..., 'loss_mask': ..., 'target': ...}`
    
- `preds`: The output of the model. For Gemma models, this is `gm.nn.Output(logits=..., cache=...)`
    
- `params`: Model parameters (can be used to add a weight decay loss, or monitor the params norm in metrics)
    

```
model = gm.nn.Gemma3_4B(
    tokens="batch.input",
)
```

```
loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
    logits="preds.logits",
    labels="batch.target",
    mask="batch.loss_mask",
)
```

We then create the trainer:

```
trainer = kd.train.Trainer(
    seed=42,  # The seed of enlightenment
    workdir='/tmp/ckpts',  # TODO(epot): Make the workdir optional by default
    # Dataset
    train_ds=ds,
    # Model
    model=model,
    init_transform=gm.ckpts.LoadCheckpoint(  # Load the weights from the pretrained checkpoint
        path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
    ),
    # Training parameters
    num_train_steps=300,
    train_losses={"loss": loss},
    optimizer=optax.adafactor(learning_rate=1e-3),
)
```

Trainning can be launched with the `.train()` method.

Note that the trainer like the model are immutables, so it does not store the state nor params. Instead the state containing the trained parameters is returned.

```
state, aux = trainer.train()
```

```
Configuring ...
Initializing ...
Starting training loop at step 0
```

## Checkpointing

To save the model params, you can either:

- Activate checkpointing in the trainer by adding:
    
    ```
    trainer = kd.train.Trainer(
        workdir='/tmp/my_experiment/',
        checkpointer=kd.ckpts.Checkpointer(
            save_interval_steps=500,
        ),
        ...
    )
    ```
    
    This will also save the optimizer, step, dataset state,…
    
- Manually save the trained params:
    
    ```
    gm.ckpts.save_params(state.params, '/tmp/my_ckpt/')
    ```
    

## Evaluation

Here, we only perform a qualitative evaluation by sampling a prompt.

For more info on evals:

- See the [sampling](https://gemma-llm.readthedocs.io/en/latest/sampling.html) tutorial for more info on running inference.
    
- To add evals during training, see the Kauldron [evaluator](https://kauldron.readthedocs.io/en/latest/eval.html) documentation.
    

```
sampler = gm.text.ChatSampler(
    model=model,
    params=state.params,
    tokenizer=tokenizer,
)
```

We test a sentence, using the same formatting used during fine-tuning:

```
sampler.chat('Hello! My next holidays are in Paris.')
```

```
'Salut ! Mes vacances suivantes seront à Paris.'
```

The model correctly translated our prompt to French!
