# Mamba-1, from the equations to these arrays

This implementation uses the real, diagonal selective SSM of **Mamba-1**.
The architecture is credited to Albert Gu and Tri Dao. The code here is an
independent educational NumPy implementation.

## Layout

| Symbol | Meaning | Shape |
| --- | --- | --- |
| B | batch size in shapes | scalar |
| L | sequence length | scalar |
| D | model width in shapes | scalar |
| E | expanded width, `expand * d_model` | scalar |
| N | state size per expanded channel | scalar |
| R | time-step projection rank | scalar |
| x | normalized block input | B × L × D |
| u, z, Δ | scan input, gate input, positive time step | B × L × E |
| A | negative continuous diagonal state coefficients | E × N |
| Bₜ, Cₜ | input-dependent write/read vectors | B × L × N |
| D_skip | learned direct-path scale | E |
| h | recurrent state | B × E × N |

`B` and `D` in mathematical expressions also conventionally denote SSM
parameters. The names `B_t`, `D_skip` disambiguate them from dimensions here.
The actual scan function uses the conventional arguments `A, B, C, D`.

## Forward

1. RMSNorm the input and linearly project it to `2E` features, split into
   a convolution branch and a gate branch `z`.
2. Apply a causal, left-padded, depthwise convolution to the first branch,
   then SiLU to obtain `u`.
3. Project `u` into `R+2N` features and split into low-rank time-step features,
   `B_t`, and `C_t`. The same `B_t/C_t` vectors are shared across channels.
4. Expand the `R` time-step features to `E` channels and apply softplus:

   $$\Delta_t=\operatorname{softplus}(r_t W_\Delta+b_\Delta),\quad A=-\exp(A_{\log}).$$

5. Update and read the state:

   $$a_{t,e,n}=\exp(\Delta_{t,e} A_{e,n}),$$
   $$h_{t,e,n}=a_{t,e,n}h_{t-1,e,n}
      +\Delta_{t,e}B_{t,n}u_{t,e},$$
   $$y_{t,e}=\sum_n C_{t,n}h_{t,e,n}+D_{\mathrm{skip},e}u_{t,e}.$$

6. Gate `y` by `SiLU(z)`, project `E → D`, and add the residual input.

The language model stacks these pre-norm residual blocks, applies a final
RMSNorm, and projects to vocabulary logits. The default head reuses the
embedding matrix; its gradients accumulate from input and output uses.

**Discretization detail.** The exponential is applied to `A`. The input update
uses `Δ B u`, matching the official real selective-scan reference. This is a
simplified input discretization; it is not the exact ZOH input integral
`A^(-1)(exp(Δ A)-I)B u`. Do not replace one with the other while claiming
identical arithmetic.

## Reverse-time differentiation

The model contains no autograd engine. Given the incoming readout derivative
`g_y` and a recurrent carry, the scan backward computes:

$$g_h=g_{\mathrm{carry}}+g_y C_t,$$
$$g_{C_t,n}=\sum_e g_{y,e}h_{t,e,n},\qquad
g_{D,e}=\sum_{\mathrm{batch}}g_{y,e}u_{t,e},$$
$$g_{u,e}=g_{y,e}D_e+\sum_n g_{h,e,n}\Delta_{t,e}B_{t,n},$$
$$g_{B_t,n}=\sum_e g_{h,e,n}\Delta_{t,e}u_{t,e},$$
$$g_{\Delta,e}=\sum_n g_{h,e,n}
 (a_{t,e,n}h_{t-1,e,n}A_{e,n}+B_{t,n}u_{t,e}),$$
$$g_{A,e,n}=\sum_{\mathrm{batch}}g_{h,e,n}a_{t,e,n}
 h_{t-1,e,n}\Delta_{t,e},\qquad
g_{\mathrm{carry}}=g_h a_t.$$

Accumulate `g_A` across time. Chain through `A=-exp(A_log)` using
`g_A_log = g_A * A` and through softplus using its sigmoid derivative.
The reverse function also accepts a derivative of the final state and returns
the derivative of the initial state. This is verified independently with
finite differences.

For RMSNorm, with `q=(mean(x²)+eps)^(-1/2)` and `g=grad_y*weight`:

$$g_x=gq-xq^3\operatorname{mean}(gx).$$

The remaining operations use ordinary matrix, convolution, SiLU, residual,
embedding scatter-add, and cross-entropy derivatives. Start at
`MambaLM.loss_and_grad`, then follow `MambaMixer.backward` into
`selective_scan_backward`.

## Why the prefix scan works

Each token defines an affine map `f_t(h)=a_t*h+b_t`. The composition is

$$ (a_2,b_2)\circ(a_1,b_1)=(a_2a_1,b_2+a_2b_1). $$

Composition is associative, so all prefixes can be computed by a scan. Our
`affine_prefix` demonstrates a vectorized Hillis–Steele implementation with
logarithmically many stages but **O(L log L) total work**. Floating-point
parenthesization changes, so tiny rounding differences are expected.
This demonstration is not Mamba's fused work-efficient GPU algorithm.

## Memory and compute

For fixed model dimensions, the sequential scan performs O(L E N) arithmetic
per layer, plus O(L) dense projections and short convolutions. A token decoder
stores `E*N` SSM values and `(K-1)*E` raw convolution values per layer:

$$\text{state bytes}=B\cdot n_{\mathrm{layers}}\cdot E
    \cdot(N+K-1)\cdot\text{dtype bytes}.$$

This is independent of processed sequence length. `prefill` also returns
`B*L*vocab_size` logits, whose storage grows with `L`. `step` returns only
the current logits. Model weights are additional fixed storage.

Training explicitly stores all `L+1` SSM states, using O(B L E N) values per
layer, plus other activations and gradients. There is no training recomputation
or fused kernel. The optional prefix demo materializes full coefficient
and state tensors. These memory scopes are deliberately distinct.

## Initialization and scope

`A_log` begins at `log(1..N)` per channel and `D_skip` at one. Initial positive
time steps are sampled log-uniformly in `[0.001,0.1]`; the inverse softplus
sets the projection bias. Dense weights use variance-scaled Gaussian samples,
and output projections are scaled by `1/sqrt(n_layers)`. Embeddings use standard
deviation 0.2 for these tiny train-from-scratch demos. The numerical dtype
is consistently float32 or float64; there is no official mixed-dtype policy.

The architecture follows Mamba-1, with independently selected educational
initialization and software layout. There are no Mamba-2/3 blocks, pretrained
weight converter, fused GPU kernels, dropout, or positional embeddings here.

## Sources

- [Gu & Dao, Mamba paper, arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- [Official Mamba-1 block](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)
- [Official selective-scan reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)

Architecture references were checked on 2026-10-23. No upstream source files
or trained weights are redistributed in this project.
