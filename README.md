# softmax_offbyone_triton
## The `softmax_offbyone` or `QuiteAttn` operator avoids the negative noisy heads by adding a constant term(1.0) in softmax denominator.

$$\text{softmax}_1(x) = {\exp x_i \over {1+\sum_j \exp x_j}}$$

$$\text{QuiteAttn}（Q， K， V） = \text{sofxmax}_1({QK^T\over \sqrt d}) V$$

And you can read this post [https://www.evanmiller.org/attention-is-off-by-one.html](https://www.evanmiller.org/attention-is-off-by-one.html) for more details.

------



Then, this implementation is heavily based on triton tutorial[06-fused-softmax](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
And the most important thing is the code change is really small:

Standard sofmax:

```python
 # initialize m->0, l->exp(0)=1
 m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - tl.inf
 l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
```

softmax_offbyone:

```python
 # initialize m->0, l->exp(0)=1
 m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
 l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
```



