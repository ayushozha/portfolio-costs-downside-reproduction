# Direct-reinforcement experiment specification

This implementation is newly written and run in 2026. Its historical boundary
concerns scientific methods, numerical software releases and benchmark data,
not a fictional execution date. The root frozen protocol supplies actual data,
split indices, releases, seeds and settings before any scientific run.

## Observations and actions

Let a risky-asset relative at source row `i` be the positive vector `y[i]`.
An action for this return is chosen from the flattened log relatives in
`y[i-L:i]`, with `L=20` in the proposed protocol. Thus the current earned
relative never enters its action's observation. Training observations alone
determine column means and standard deviations; those constants are retained
for validation and test. The price-relative data's absent calendar labels are
not reconstructed or invented.

The policy contains two ReLU hidden layers (32 and 16 units), followed by a
softmax over risky assets and a cash action. Cash earns a unit relative. Its
parameters directly optimize portfolio rewards; it is not a return classifier.
It has no preceding-action input, convolutional encoder or portfolio-vector
memory. Accordingly, it is a cost/risk ablation inspired by pre-cutoff direct
reinforcement research, not an exact Jiang et al. architecture reproduction.

## Self-financing trading

Each independent train, validation or test trajectory starts with one unit of
wealth in cash. If the preceding target was `a` and preceding return was `z`,
the drifted proportions immediately before the next trade are
`q = a*z / sum(a*z)`. For a new target `w`, the post-cost wealth multiplier is
the unique root of

`mu + c*sum_risky(abs(mu*w - q)) = 1`.

The cash component incurs no fee. Both sales and purchases of risky assets
incur the same proportional coefficient `c`. The monotone root is solved by
48 bisection steps on `[(1-c)/(1+c), 1]`. This accounts for trading in units of
pre-trade wealth rather than applying an approximate `1-c*turnover` to
unscaled target weights. Portfolio net log growth is
`r = log(mu) + log(sum(w*y))`. There is no terminal liquidation, price impact,
bid-ask spread, borrowing, short selling, leverage or positive cash yield.
Fees are modeled assumptions, not claims about actual historical brokerage.

## Objective and complete adjacent-action derivative

The objective is the mean of `r - lambda*min(r,0)^2`. This penalizes squared
negative daily log growth relative to a zero target; it is not CVaR,
drawdown optimization, Sortino-ratio optimization or a constraint guaranteeing
risk improvement.

Writing `s = sign(mu*w_risky - q_risky)` and
`D = 1 + c*sum(s*w_risky)`, the derivatives away from trade-sign boundaries are
`d(log(mu))/dw_risky = -c*s/D` and
`d(log(mu))/dq_risky = c*s/(mu*D)`. For a derivative vector `g` with respect
to drifted proportions, the preceding-action contribution is
`(z/sum(a*z)) * (g - sum(g*q))`. These expressions are included in the
backpropagation, alongside the softmax, two ReLU layers and downside term.
The chosen derivative is zero at exact absolute-value and ReLU kinks.

Each contiguous 128-return training block is sampled using a seeded
`RandomState`. A noninitial block includes its preceding policy action in
the graph, omits that predecessor's own reward, and propagates the first
included trading cost through that preceding action. Therefore a sampled
block is not artificially treated as beginning in cash. Internal block
actions receive complete adjacent-cost derivatives. Sampling overlapping
blocks changes the weighting of observations near the partition edges;
this is disclosed rather than presented as an unbiased full-trajectory
gradient estimator. There is no environmental state affected by decisions
other than holdings and the adjacent self-financing cost.

## Optimizer and model selection

Adam follows Algorithm 1 of Kingma and Ba's **arXiv:1412.6980v1** exactly in
its innovation-fraction notation: `beta1=0.1`, `beta2=0.001`,
`m = beta1*g + (1-beta1)*m`, `v = beta2*g^2 + (1-beta2)*v`, and bias
denominators `1-(1-beta)^t`. The learning rate is 0.0002 and epsilon is
1e-8. The code performs ascent on the reward instead of descent on its
negative. This is equivalent to the common later decay-coefficient notation
when innovation coefficients are converted, but the cited version and
implemented notation are explicit.

The proposed run makes 600 updates per seed and configuration. Update zero
and every 50 updates are evaluated on the fixed validation partition.
The checkpoint with the greatest matching validation objective is retained;
strict ties retain the earlier checkpoint. Return-only training and
validation use `c=0, lambda=0`; cost-aware uses `c=0.0025, lambda=0`; the
downside variants use `c=0.0025` with each prespecified lambda. The main
downside coefficient is 10; 1 and 100 are sensitivity analyses, not
post-test selections. Test outcomes do not select parameters or checkpoints.
No claim of global optimization follows from this nonconvex training.

## Evaluation and artifacts

The same selected actions are evaluated at fee coefficients 0, 0.001,
0.0025 and 0.005. Benchmarks are zero-yield cash, daily equal-weight
constant rebalancing, and initial equal-weight buy-and-hold. Buy-and-hold
incurs initial entry fees, then its target follows its own drifted holdings,
so subsequent turnover is zero. All strategies start from the same cash
condition; a fully invested initial buy incurs multiplier `1/(1+c)`.

Reported metrics include terminal wealth, cumulative and annualized return,
annualized volatility, zero-cash-rate Sharpe, zero-target Sortino, annualized
downside deviation, maximum drawdown, negative-return frequency, mean risky
dollar turnover and cumulative log fee drag. Annualized summaries assume
252 observations per year and their conventional square-root scaling;
they are descriptive rather than serial-dependence-adjusted inference.
Return-row counts and index ranges are retained because source matrices
do not contain per-observation trading dates.

Each test path records its original source row, net simple/log return,
wealth, exact multiplier, turnover and all target weights. Selected network
parameters, training-only normalization arrays, all validation checkpoints,
source hash, frozen-protocol hash, engine hash, actual UTC execution dates,
Python/NumPy versions and elapsed times are saved. Existing output
directories are never replaced. Five initialization/sampling seeds measure
algorithmic variability, not independent market replications.

## Focused implementation checks

Nine synthetic software tests check header/no-header row preservation,
exact self-financing accounting,
initial purchase fees, cash invariance, buy-and-hold turnover, absence of
future features, complete allocation and MLP finite-difference gradients,
and Adam's first update. They are not financial experiments and do not
establish profitable trading. The historical scientific runtime must pass
these same checks before executing the frozen market protocol.
