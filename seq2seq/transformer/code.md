finish all part of question sin this repo following these guideleinsat like the water brand?
[7, 0, 30, 0, 29, 16, 56, 94, 16, 30, 94, 31, 15, 0, 31, 94, 21, 16, 20, 8, 94, 31, 15, 8, 94, 37, 0, 31, 8, 29, 94, 4, 29, 0, 23, 7, 56]

Time to implement this ourselves! In seq2seq/tokenizer/character_tokenizer.py, you’ll see a CharacterTokenizer class. Your first task is to implement this tokenizer. Your subtasks are:

Fill the self.vocab dictionary with our mapping of
char
→
index
char→index (so the keys of self.vocab will be all the characters from self.characters, and the values will be indices starting from
0
0).
Implement the encode function, which will take a string text and output a Tensor containing a list of token indices.
Implement the decode function, which will take a Tensor containing a list of token indices and return the original string.
After implementing your tokenizer, test with:

python -m unittest tests/test_tokenizer/test_character_tokenizer.py

Byte-Pair Encoding (BPE)
While character-based tokenization is a valid way of encoding your text, it has limitations. In particular, each character doesn’t have any specific meaning inherently attached to it. If we see words like “old”, “older”, and oldest”, it would help to repeat a single token for “old” that allows models to see that these words are related. Additionally, character-based tokenization leads to very, very long sequences, which will eventually require more compute to process.

But if we went to the other extreme and tried to give every English word a separate token, our vocab size would be huge (!) and we still run into the “old”/”older” problem. Byte-pair encoding tokenizes subwords to strike a balance between the two, where “old” and “older” might share a subword token for “old”, with another token for “er”.

We don’t require you to implement BPE, but here’s how it works conceptually: we will iteratively replace the most commonly occuring sequence with a new token.

You have a large corpus of data you would like to build your BPE with.
You start with one token for each unique byte in our data. Assume we start with 100 tokens.

Dasari? Is that like the water brand?
[7, 0, 30, 0, 29, 16, 56, 94, 16, 30, 94, 31, 15, 0, 31, 94, 21, 16, 20, 8, 94, 31, 15, 8, 94, 37, 0, 31, 8, 29, 94, 4, 29, 0, 23, 7, 56]

Now, we sweep through the data for all pairs of tokens and note down which pair occurs most frequently. In this case, [0, 31] corresponding to at occurs most frequently, so it’ll become the 101th token
a
t
→
101
at→101.
We retokenize our original data with our new vocab, and repeat the process until we reach a desired vocab size or until there are no pairs of tokens that occur more than once.
At the end, we have a bigger vocab than just the characters:
a
→
0
b
→
1
…
a
t
→
101
…
a→0
b→1
…
at→101
…
Takeaway
Tokenization enables us to encode our textual inputs into a number-based representation. Important things to note:

When we want to batch our sequences for training, we need to make sure all of the sequences are the same length. If the maximum-length sequence in our batch has length
L
L, then we will pad our remaining sequences with a special pad token [PAD], which has some token index like 50267. This is handled for you in the given Datasets.
To denote when our sequences start and end, we’ll add additional special tokens at the start and end for Start Of Sequence and End of Sequence: [SOS] and [EOS] respectively (also included for you).
With batching, our sequence shape has now been converted as:

input text
→
(
B
,
T
)
input text→(B,T)
where
B
B denotes batch size and
T
T denotes “time” or “# of tokens”. You should interpret this as: we have
B
B sequences of length
T
T.

Embedding
From our tokenization, we have a tensor representation for any arbitrary French or English input. Now here comes the fun part - we want to create an embedding for our text. As a reminder, an embedding is some sort of tensor that contains all the essential information about our text.

Generally, this will be in a shape like
(
B
,
T
,
C
)
(B,T,C), containing the batch size (
B
B), the number of time-steps (
T
T), and the number of channels (
C
C). What does this mean? Well, let’s break it down:

Batch size and time-steps we know from the previous step. Number of channels (
C
C) gives us the length of each embedding. We can choose this to be 256, 512, or really any number - powers of two are common ones for efficient computation.

To be very clear, this means that for every token, we have a
C
C-length vector that represents it. At the start of our model, we don’t have any contextual information. So compute an embedding, we just use a linear transformation!

Let’s take a look at an example to make this explicitly clear:

Word to embed: “hello”

Batch size: 1 (we only have one input text to embed)
Number of time-steps: 5 (this is the length of our tokenized word, which has 5 characters)
Number of channels: 3 (this is the length of each embedding vector, chosen by us!)
Overview of Embeddings
nn.Embedding Simplification
If we look closely at the matrix multiplication, we can notice that for each token, the “multiplication” is just choosing the row in W corresponding to that token!

So, this “linear transformation” is just a lookup table, where we have
V
V vectors (V being the vocab size), and we look up the vector for each token and pile them together in a
T
×
C
T×C matrix. This is so common that PyTorch provides us with a nn.Embedding object that creates the differentiable lookup table for us!

At the start of our model, all of these
V
V vectors contain completely random numbers, and through gradient descent come to represent important information about our tokens.

As a summary, using an Embedding layer, we’ve converted our tokens into token embedding vectors containing token information. We are ready to build our Transformer now!

Shape progress:

Tokenize: input text
→
(
B
,
T
)
Token Embeddings:
(
B
,
T
)
→
(
B
,
T
,
C
)
Tokenize: input text→(B,T)
Token Embeddings: (B,T)→(B,T,C)
Transformer
Remember that our translation task has two steps:

Encode all the information from our French input.
Use this encoded information to autoregressively generate an English translation.
So, let’s first focus on encoding the French input. At this point, we’ve tokenized the French, then created an embedding for it. We consider our input to the Transformer Encoder to be
(
B
,
T
,
C
)
(B,T,C) and the Transformer Encoder will spit out an output of the same shape
(
B
,
T
,
C
)
(B,T,C)! While going through the Transformer Encoder, each embedding has gathered contextual information from surrounding tokens (by attending to them). Let’s take a look at how this attention process works and implement it.

Attention
The goal of attention is to learn contextual information about this token based on all other tokens. Recall what our embeddings look like:

Attention Embeddings
Also recall what our goal is: for each embedding, we want to search up related embeddings and extract the most important information from them.

Our YouTube analogy is apt! If I want to watch a Mario Kart video on YouTube, I’ll use a query “Mario Kart gameplay”, which will check how similar my query is to a bunch of YouTube titles, which we call keys, and each title has a video value associated with it.

Youtube Analogy
How do we replicate this hash table idea with our embeddings? Well, we use our handy linear transformation tool and just learn (in the machine learning sense) how to make queries, keys, and values!

Full Attention
If we do this for every embedding (see below), we’ll again end up with 5 “out” embeddings, each of which is a weighted sum of our values. If you understand this, you know exactly how the entirety of attention works. Now, just like how we use many kernels in a CNN, we’ll apply this process on these embeddings many times (this is where the “multi-head” term comes from).

MHA Attention
And this is where we get our full scaled dot-product attention equation from the paper:

attention
σ
(
Q
K
⊤
)
qk
_
length
⋅
V
attention=
qk_length
​

σ(QK
⊤
)
​
⋅V
After this diagram, we’ve covered scaled dot-product attention and multi-head attention blocks as described in the Attention Is All You Need paper. You’re ready to implement them yourselves!

Task: Implement the full Transformer Encoder. Subtasks:

Implement all parts of MultiHeadAttention in seq2seq/transformer/attention.py. This includes: init, split_heads, combine_heads, scaled_dot_product_attention, forward. Follow the paper closely and use the diagrams for guidance. An implementation of positional encoding is provided for you.
Implement the FeedForwardNN in seq2seq/transformer/attention.py. All this entails is adding two Linear layers that transform your embeddings of size
(
B
,
T
,
C
)
(B,T,C) to some intermediate shape
(
B
,
T
,
hidden_dim
)
(B,T,hidden_dim) with a ReLU operation, then transforming them back to
(
B
,
T
,
C
)
(B,T,C).
Implement the Encoder in seq2seq/transformer/encoder.py. You’ll need the modules from attention.py. In particular, implement EncoderLayer and then Encoder.
After this step, run:

python -m unittest tests/test_transformer/test_encoder.py

You should be passing all these tests (these are purely sanity checks, not correctness checks, which will come during training).

Shape progress:

Tokenize: input French text
→
(
B
,
T
enc
)
Token Embeddings:
(
B
,
T
enc
)
→
(
B
,
T
enc
,
C
)
Encoder:
(
B
,
T
enc
,
C
)
→
(
B
,
T
enc
,
C
)
Tokenize: input French text→(B,T
enc
​
)
Token Embeddings: (B,T
enc
​
)→(B,T
enc
​
,C)
Encoder: (B,T
enc
​
,C)→(B,T
enc
​
,C)
For your reference, here are the Encoder and Attention figures from the original Attention Is All You Need paper:

MHA Attention
Decoder
Great! We’ve successfully encoded all the information from our French input into a set of contextual embeddings
(
B
,
T
enc
,
C
)
(B,T
enc
​
,C)

Now, we need to use this encoded information to autoregressively generate an English translation. This is the job of the Decoder.

The Decoder is also a stack of layers, just like the Encoder. However, it’s a bit more complex because it has to manage two different inputs:

The Encoder’s output embedding: This is the fully contextualized French sentence (our
(
B
,
T
enc
,
C
)
(B,T
enc
​
,C) tensor). This is where the Decoder gets its information about what to translate.

The target sequence: This is the English translation generated so far. For example, if we’re translating “Je suis un” to “I am a”, and we want to predict the next word (“student”), we feed “I am a”
(
B
,
T
dec
,
C
)
(B,T
dec
​
,C) into the decoder as the “target sequence”.

The Decoder’s job is to take these two inputs and produce an output
(
B
,
T
dec
,
C
)
(B,T
dec
​
,C) that represents the best prediction for the next token at each position.

Attention Modifications for Decoding
In the decoder, we have to make some modifications to our attention mechanism: masked-attention and cross-attention.

Masked Multi-Head Attention (Self-Attention): This first attention block operates on the target (English) embeddings. It’s almost identical to the Encoder’s self-attention, with one crucial difference: we must prevent the decoder from “cheating” by looking at future tokens.

When we’re at position
i
i (e.g., trying to predict the word after “am”), we should only have access to tokens at positions
≤
i
≤i (“I”, “am”). We can’t look at position
i
+
1
i+1 (“a”), because we’re trying to predict it!

We accomplish this by applying a look-ahead mask to the attention scores (the
Q
K
⊤
QK
⊤
matrix) before the softmax. This mask sets all values corresponding to future positions to
−
∞
−∞. Remember what the softmax function looks like! If we want attention scores for future tokens to be zero after the softmax, we must set their output after the
Q
K
⊤
QK
⊤
matrix to be
−
∞
−∞.

Masked Self-Attention
One other place where we need to use this masking is to mask out our pad tokens. Remember that we add our pad tokens when batching multiple sequences together of varying length. If we allow the encoder and the decoder to attend to pad tokens, these tokens will end up dominating our training and lead to divergent models.

The functions to create boolean causal and pad masks for the encoder and decoder are provided in seq2seq/transformer/transformer.py as make_pad_mask and make_causal_mask.

Cross-Attention (Encoder-Decoder Attention): Remember that self-attention allows us to attend to tokens within our own sequence. For the Decoder, we also want to look at the French input by attending to the French embedding from the Encoder..

Recall our YouTube analogy. Now, the Query (
Q
Q) comes from the output of our Decoder’s masked self-attention (from step 1). This
Q
Q represents: “I am an English token, this is my context so far. What information do I need from the French sentence to make my next prediction?”

The Keys (
K
K) and Values (
V
V) come from the Encoder’s output (the contextualized French embeddings). Remember that keys inform us about which values are relevant: keys and values are tied together and must originate from the same source.

The Decoder’s query (“I am a…”) searches against all the French keys. It might find that its query is most similar to the key for the “suis” token embedding while also attending to the other tokens, and it will then take a weighted sum of the French values to produce its output. This is how it learns the alignment between the two languages.

Cross-Attention
Task: Implement the full Transformer Decoder. Subtasks:

Modify scaled_dot_product_attention in seq2seq/transformer/attention.py. If you haven’t added this yet, handle the optional mask argument. If this mask is provided, you must “fill” the attention scores (the matmul(q, k) / ... result) with a very large negative number (like -float('inf')) at all positions where the mask is 1. This must be done before applying the softmax.

Implement all parts of the Decoder in seq2seq/transformer/decoder.py.

Implement DecoderLayer. This will involve creating instances of your MultiHeadAttention (one for masked self-attention, one for cross-attention) and your FeedForwardNN. Don’t forget the three residual connections and LayerNorms! You’ll notice you are given both a tgt_mask and a src_mask here. tgt_mask has both the causal mask and the pad mask applied for the English input into the Decoder. src_mask has the pad mask applied to it.

You’ll need to think about where to input the src_mask vs the tgt_mask (hint: the only function that actually deploys any masks is the scaled_dot_product_attention function)

Remember that our LM task will be decoder-only, so we don’t want to do cross-attention in this case. When enc_x is None, make sure to skip the cross-attention step in your DecoderLayer.

Implement Decoder. This will be a ModuleList of your DecoderLayers, just like in the Encoder. It will also need to handle the target embeddings and positional encoding. Its forward method will be more complex, as it takes both the target sequence tgt and the enc_x (encoded French).

After this step, run:

python -m unittest tests/transformer/test_decoder.py

You should be passing all these tests. Once this is done, you have a full Transformer! We’ve implemented the joining of the Encoder and Decoder for you in seq2seq/transformer/transformer.py (take a look!).

Tokenize: input French text (source)
→
(
B
,
T
enc
)
Token Embeddings:
(
B
,
T
enc
)
→
(
B
,
T
enc
,
C
)
Encoder:
(
B
,
T
enc
,
C
)
→
(
B
,
T
enc
,
C
)
Tokenize: input English text (target)
→
(
B
,
T
dec
)
Token Embeddings:
(
B
,
T
dec
)
→
(
B
,
T
dec
,
C
)
Decoder Masked Self-Attention:
(
B
,
T
dec
,
C
)
→
(
B
,
T
dec
,
C
)
Decoder Cross-Attention:
(
B
,
T
enc
,
C
)
+
(
B
,
T
dec
,
C
)
→
(
B
,
T
dec
,
C
)
Tokenize: input French text (source)→(B,T
enc
​
)
Token Embeddings: (B,T
enc
​
)→(B,T
enc
​
,C)
Encoder: (B,T
enc
​
,C)→(B,T
enc
​
,C)
Tokenize: input English text (target)→(B,T
dec
​
)
Token Embeddings: (B,T
dec
​
)→(B,T
dec
​
,C)
Decoder Masked Self-Attention: (B,T
dec
​
,C)→(B,T
dec
​
,C)
Decoder Cross-Attention: (B,T
enc
​
,C)+(B,T
dec
​
,C)→(B,T
dec
​
,C)
Here is the full Transformer architecture now (including the Decoder) for your reference (from Attention Is All You Need paper):

Cross-Attention
Training
Once we are finished with our architecture, we want to implement our training loops for both the NMT and the LM tasks.

NMT Task
Fill in the TODO sections in scripts/train_nmt.py.

You’ll need to take a look at seq2seq/tokenizer/bpe_tokenizer.py for details on what index to ignore.
Fill in the tgt_output. Think carefully about what should be the target input and target output for our model (hint: remember that the Decoder is autoregressive!)
LM Task
We’ve implemented the LM training script for you! Just add the same line that you added in the NMT task in the TODO line in scripts/train_lm.py.

Tracking experiments
We’ve added simple wandb logging to your training scripts. Make sure to fill in your entity names in both scripts to track your experiments!

Start training!
Set your devices to be different values (based on which GPUs are free on honeydew according to nvidia-smi).
Train! You can run:

tmux # open two different panes or two different tmux sessions
uv run scripts/train_nmt.py
uv run scripts/train_lm.py

and your model should start training!

Checkpoints will start saving in your working directory. Once you have checkpoints, you can test your models with:

uv run scripts/decode_nmt.py
uv run scripts/decode_lm.py

Take a look at these scripts. In both cases, we basically start with a BOS token, produce next token distributions, sample from them, and add to our list of accumulated tokens so far. If we either hit our max_len or an EOS token, we stop decoding! Feel free to explore different decoding schemes, including greedy, top-
k
k, top-
p
p (nucleus), and beam search.

Submission
Once you feel as though your model outputs are satisfactory when running the decode scripts, feel free to submit:

On the HW 4 grading assignment, please submit the text outputs from the decode_nmt.py and decode_lm.py scripts.
And you’re done! You’ve successfully built and trained Transformer-based models :)