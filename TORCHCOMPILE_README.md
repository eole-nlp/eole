# Notes on the usage of Torch Compile and cudagraphs

Starting at version 0.5 we have refactored the code to be torch.compile friendly

torch.compile is very tricky to tune and the set up changes regularly on each new version of pytorch.

We decided to implement 4 modes:

* First you need to turn on the torch compiler with a new environment variable:
`EOLE_TORCH_COMPILE="1"` default value is "0"

* Then you need to pick a mode ("2" by default)
`EOLE_COMPILE_MODE="2"`

* The first two values trigger torch compile at the Transformer-Decoder level 
- Mode = 0 : Decoder Level - cudagraphs True
- Mode = 1 : Decoder Level - cudagraphs False
- The first compilation will be time consuming (about 60-80 seconds for a 24 layer model)
- Each run after will take about 10 seconds to warmup but you will get the fastest speed in terms of throughput: eg > 1300 tok/sec with Gemma3-1b on a RTX5090
- Mode "1" is mostly for educational purpose, compilation is the same but without triton.cudagraphs

* The last two values trigger torch compile at Transformer DecoderLayer level
- Mode = 2 : Decoder Layer Level - cudagraphs True
- Mode = 3 : Decoder Layer Level - cudagraphs False
- The first compilation will be very fast (10 sec first run, then 2 sec) because it will compile only one layer and reuse it for the next 23 layers.
- The only downside is that it does not optimize the potential fused ops between layers. Comparative speed vs full above will be around 950 tok/sec


## What strategy did we implement ? (subject to change)

* We use eager mode (but adding cuda kernels for rmsnorm, rope, activation) for the prefill forward.
* Based on the incoming batch `batch_size`, we warmup all batch sizes from `1` to `batch_size` with sequence length `1`
* Then we process the batch

The main reason is that currently Eole reduce the batch size when sequences are finished and eliminated fomr the batch. We could adopt another strategy and keep the batch size constant.

* We also changed the way the KV cache is set up. We used to handle a dynamically growing cache which is a bit cumbersome with torch.compile
* Selecting the `max_length` upfront is key and now counts for both the prefill and decoded tokens. (cf benchmark scripts or recipes)
