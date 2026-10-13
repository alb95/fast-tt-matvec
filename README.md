# fast-tt-matvec

MATLAB code for

> A. Bucci, S. Güttel, J. W. Pearson,
> *Fast matrix–vector products in tensor-train format via structured sketching.*

Given a TT operator `A` and a TT vector `v`, the routines here return a TT
approximation of `A*v` of prescribed rank `s` **without ever forming the
product**, whose cores would carry TT rank `r^A r^v`. The distinctive step is
that the Khatri–Rao sketch is contracted into the cores of the operator before
the operator is applied, which pays off when the mode sizes are large and the
target rank is well below `r^v`.

The repository contains the three routes analysed in the paper (sequential,
fused, pre-sketched), both orderings of successive randomized compression (SRC)
used for comparison, the three experiments that produce the figures, and the
data those figures are drawn from.

## Requirements

* MATLAB. The results in the paper were produced with R2026a; `pagemtimes` and
  `tensorprod` are used, so R2022a or later is needed.
* The [TT-Toolbox](https://github.com/oseledets/TT-Toolbox). Everything here
  takes and returns `tt_tensor` and `tt_matrix` objects. Pin a commit rather
  than tracking `master`: the semantics of `round` have changed in the past.

## Quick start

Start MATLAB **in the repository root** and run

```matlab
>> setup                     % puts the four folders on the path
>> test_orderings_agree      % seconds: the orderings return the same object
>> exp1_target_rank          % Figure 1
```

Each experiment writes its CSV and a log of the run into the current
directory, which is why the root is the place to launch them: `.gitignore`
ignores those names there, so a rerun cannot overwrite or pollute the
reference copies in `results/`. Reference copies of both, from the runs reported in the paper, are
committed in `results/`, so the figures can be rebuilt without re-running
anything: `pdflatex figures/fig2.tex` reads `results/response_time.csv`
directly.

## Layout

| folder | contents |
|---|---|
| `src/` | the six compression routes and their building blocks |
| `utils/` | random TT test problems, the timing harness, CSV output, the operation-count model |
| `experiments/` | one script per figure |
| `results/` | the CSVs and run logs behind the published figures |
| `figures/` | standalone `pgfplots` sources that redraw each figure from `results/` |
| `test/` | correctness checks, and a scratch script for exploring parameters |

## Figures

| figure | script | data | what it shows |
|---|---|---|---|
| 1 | `experiments/exp1_target_rank.m` | `results/exp1_target_rank_plot.csv`, `exp1_target_rank.csv` | running time of all six routes at `s = 6, 40, 100`, everything else fixed |
| 2 | `experiments/exp2_resonance.m` | `results/response_time.csv`, `response_acc.csv` | a near-resonant solve: `A*v` collapses to TT rank 3 from `r^v = 100` |
| 3 | `experiments/exp3_stream.m` | `results/stream.csv`, `stream_eps.csv`, `stream_residual.csv` | one pass over a stream of 100 realisations |

Sizes of the objects involved, from the logs in `results/*.txt`:

| | Figure 1 | Figure 2 | Figure 3 |
|---|---|---|---|
| parameters | `d=4, m=n=512, r^A=4, r^v=40` | `d=3, n=512, r^A≤16, r^v=100` | `d=4, n=300, N=100, r^A=4, r^v=56` |
| operator | 84 MB | 604 MB | 29 MB |
| argument | 13 MB | 42 MB | 1532 MB for the whole stream |
| cores of `A*v` | 211 MB | 10 499 MB, so never formed | — |

Figure 2 is the heavy one: `r^A r^v = 1600`, so a single interior core of the
product would occupy 10.5 GB. Forming it and the sequential ordering are both
skipped for that reason, which is why their curves are absent. Figure 3's
`mem_ours` and `mem_src` columns record the point of the streaming experiment:
4.3 MB against 1551 MB at the largest target rank.

## Names

The code predates the paper's terminology. The correspondence is:

| paper | function |
|---|---|
| sequential STTA | `stta_matvec_sequential` |
| fused STTA | `stta_matvec_fused` |
| pre-sketched STTA | `stta_matvec_presketched` |
| pre-sketched STTA, operator reused | `stta_matvec_presketched_reused` |
| standard SRC | `src_matvec_standard` |
| pre-sketched SRC | `src_matvec_presketched` |

`stta_matvec_*` return the approximant and are what the experiments time;
`stta_compress_*` return the sketches `Omega_k`, `Psi_k` from which
`stta_recover` builds it. The distinction between the last two STTA entries is
the one drawn by the two columns of the cost table in the paper:
`stta_matvec_presketched` builds the sketched operator inside the call, so the
preparation is charged to the product, whereas
`stta_matvec_presketched_reused` receives a sketched operator built once by
`stta_sketch_operator` and reuses it, which is the setting of repeated
application.

## How the timings were taken

`time_best_multi` times the routes **round-robin**: one call of each per pass,
several passes, and the minimum over passes. Running every repetition of one
route before starting the next would measure the first and the last route on a
processor in different thermal states — the drift measured over one of these
scripts was a factor 1.34, larger than several of the ratios the experiments
exist to establish. Alternating puts the readings to be compared seconds apart
instead of minutes, so the drift multiplies them all by nearly the same factor
and cancels in the ratio. The minimum rather than the mean is taken because the
noise is one-sided: a call can be delayed, never accelerated.

The number of inner calls is calibrated once per route so that each timed block
lasts at least 0.25 s, and five passes are made. Every route builds everything
it needs, the pre-sketch included, inside the timed call.

Absolute seconds are therefore context; the ratios are the result.

Alongside each time the scripts print `spread`, the ratio of the slowest to
the fastest of that routine's readings. The minimum is the right estimator
only as long as the readings measure the same work, one of them with less
interference than the others; a reading far faster than its siblings means
something structurally different happened, and no estimator can say which of
the two numbers is the cost. A spread near one is the evidence that the
minimum can be trusted. It is a diagnostic and not a rule: nothing is
discarded and nothing is rerun automatically, and `time_best_multi` only warns
when some routine's readings lie more than a factor 1.5 apart.

## Reproducibility

Each experiment seeds the generator at the top (`rng(0)`, `rng(4)`, `rng(11)`),
so a rerun on the same MATLAB version reproduces the same problem instance. The
logs in `results/*.txt` record the MATLAB version, thread count, platform and
date of the published runs.

## Licence

BSD 3-Clause, see `LICENSE`. If you use this code, please cite the paper; see
`CITATION.cff`.
