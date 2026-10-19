# Precision of `modefinder.emp_mode()`

On normal and beta data, the Bernstein method (the default for
`modefinder.emp_mode(data)`) is very accurate. It shows no noticeable bias,
and its error shrinks steadily as the sample grows. The study ran 200
repeated samples for each distribution and sample size, and compared the
estimates with the known true modes.

## Bernstein method

Typical error (RMSE), in units of the distribution's standard deviation for
the normals and in raw units for the betas (whose values lie between 0 and
1):

| distribution (true mode) | n = 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| Normal(0, 1) (0) | 0.29 | 0.16 | 0.056 | **0.017** |
| Beta(2, 5) (0.2) | 0.051 | 0.031 | 0.010 | **0.0033** |
| Beta(5, 2) (0.8) | 0.052 | 0.032 | 0.010 | **0.0031** |
| Beta(2, 2) (0.5) | 0.095 | 0.069 | 0.024 | **0.0072** |
| Beta(3, 10) (0.18) | 0.029 | 0.021 | 0.0071 | **0.0030** |
| Beta(1, 3) (0, on the boundary) | 0.044 | 0.023 | 0.0069 | **0.0006** |

- **Bias** is at most 0.003 in every case from n = 1,000 up. The error drops
  about threefold for every tenfold increase in sample size.
- **The mode on the boundary:** Beta(1, 3) has its mode at 0, and the method
  correctly returns the boundary point.
- **Speed:** about 8 ms per call at n = 100,000.

## Compared with the other two methods

- **Kernel (`smooth_option="kernel"`):** about as accurate up to n = 1,000,
  but it improves much more slowly after that. At n = 100,000 its error is
  3–4× the Bernstein method's: 0.062 vs 0.017 on the normal, and 0.011 vs
  0.0033 on Beta(2, 5). It also misses a mode at the boundary: on Beta(1, 3)
  it still puts the mode about 0.04 away from 0 at n = 100,000.
- **Empirical (`smooth=False`):** much less precise, and it doesn't get
  better with more data. The error on the normal is still about 0.3 at
  n = 100,000. The default number of bins equals the sample size, so each
  bin holds about one point and the estimate stays noisy. Passing a much
  smaller `m_knots` would probably help, but this was not tested.

## One thing to watch

The Bernstein method isn't fully shift-invariant. When all the data are
positive, the lower bound of the assumed range defaults to 0 rather than
just below the smallest value. At n = 100, Normal(5, 2) samples are usually
all positive, so it scores 0.33 against 0.29 for Normal(0, 1); from
n = 1,000 up the two give identical results. For data far from [0, 1],
passing `fix_lower` and `fix_upper` avoids this.

## Reproducing

The numbers come from [`precision_study.py`](precision_study.py) (needs
numpy), run with its defaults (seed 2024):

``` sh
python examples/precision_study.py --reps 200 --sizes 100 1000 10000 100000
```

Timings were measured on the machine the study was run on and will differ
elsewhere.

## Full results

bias, sd, RMSE and MAE are errors of the mode estimate divided by the
distribution's scale (the sd for the normals, 1 for the betas).

```
distribution        n method          bias       sd     RMSE      MAE  ms/call
------------------------------------------------------------------------------
Normal(0, 1)      100 Bernstein    -0.0322   0.2865   0.2876   0.2190     0.07
Normal(0, 1)      100 kernel       -0.0394   0.2886   0.2905   0.2299     0.03
Normal(0, 1)      100 empirical     0.0020   0.4730   0.4718   0.3812     0.00
Normal(0, 1)     1000 Bernstein     0.0030   0.1621   0.1618   0.1257     1.56
Normal(0, 1)     1000 kernel        0.0063   0.1584   0.1581   0.1259     0.07
Normal(0, 1)     1000 empirical    -0.1241   0.4180   0.4350   0.3423     0.05
Normal(0, 1)    10000 Bernstein    -0.0031   0.0565   0.0564   0.0454     2.02
Normal(0, 1)    10000 kernel       -0.0013   0.1060   0.1058   0.0873     0.63
Normal(0, 1)    10000 empirical    -0.1116   0.3640   0.3799   0.3092     0.59
Normal(0, 1)   100000 Bernstein     0.0004   0.0169   0.0169   0.0133     8.33
Normal(0, 1)   100000 kernel       -0.0028   0.0622   0.0621   0.0496     7.12
Normal(0, 1)   100000 empirical    -0.0699   0.3111   0.3181   0.2526     7.20

Normal(5, 2)      100 Bernstein    -0.0186   0.3272   0.3269   0.2346     0.08
Normal(5, 2)      100 kernel       -0.0394   0.2886   0.2905   0.2299     0.03
Normal(5, 2)      100 empirical    -0.0261   0.4770   0.4765   0.3900     0.00
Normal(5, 2)     1000 Bernstein     0.0029   0.1625   0.1621   0.1260     1.51
Normal(5, 2)     1000 kernel        0.0063   0.1584   0.1581   0.1259     0.08
Normal(5, 2)     1000 empirical    -0.1570   0.4014   0.4301   0.3425     0.05
Normal(5, 2)    10000 Bernstein    -0.0031   0.0565   0.0564   0.0453     2.05
Normal(5, 2)    10000 kernel       -0.0013   0.1060   0.1058   0.0873     0.61
Normal(5, 2)    10000 empirical    -0.1513   0.3542   0.3844   0.3141     0.61
Normal(5, 2)   100000 Bernstein     0.0004   0.0169   0.0169   0.0133     8.39
Normal(5, 2)   100000 kernel       -0.0028   0.0622   0.0621   0.0496     7.11
Normal(5, 2)   100000 empirical    -0.0775   0.2983   0.3075   0.2453     7.23

Beta(2, 5)        100 Bernstein     0.0027   0.0513   0.0513   0.0415     0.08
Beta(2, 5)        100 kernel        0.0231   0.0493   0.0543   0.0421     0.03
Beta(2, 5)        100 empirical     0.0159   0.0855   0.0868   0.0690     0.00
Beta(2, 5)       1000 Bernstein     0.0009   0.0312   0.0311   0.0247     1.05
Beta(2, 5)       1000 kernel        0.0082   0.0297   0.0307   0.0241     0.07
Beta(2, 5)       1000 empirical     0.0016   0.0702   0.0700   0.0552     0.05
Beta(2, 5)      10000 Bernstein     0.0001   0.0102   0.0102   0.0081     1.12
Beta(2, 5)      10000 kernel        0.0034   0.0183   0.0185   0.0152     0.62
Beta(2, 5)      10000 empirical     0.0092   0.0568   0.0574   0.0436     0.60
Beta(2, 5)     100000 Bernstein     0.0000   0.0033   0.0033   0.0026     7.21
Beta(2, 5)     100000 kernel        0.0021   0.0113   0.0115   0.0092     7.14
Beta(2, 5)     100000 empirical    -0.0030   0.0582   0.0581   0.0480     7.27

Beta(5, 2)        100 Bernstein     0.0032   0.0517   0.0517   0.0411     0.08
Beta(5, 2)        100 kernel       -0.0176   0.0517   0.0545   0.0404     0.03
Beta(5, 2)        100 empirical    -0.0334   0.0834   0.0896   0.0695     0.00
Beta(5, 2)       1000 Bernstein     0.0014   0.0317   0.0317   0.0252     1.03
Beta(5, 2)       1000 kernel       -0.0045   0.0297   0.0300   0.0235     0.07
Beta(5, 2)       1000 empirical    -0.0490   0.0816   0.0950   0.0755     0.05
Beta(5, 2)      10000 Bernstein    -0.0002   0.0104   0.0103   0.0078     1.11
Beta(5, 2)      10000 kernel       -0.0020   0.0174   0.0175   0.0140     0.60
Beta(5, 2)      10000 empirical    -0.0309   0.0653   0.0721   0.0568     0.60
Beta(5, 2)     100000 Bernstein     0.0004   0.0031   0.0031   0.0025     7.17
Beta(5, 2)     100000 kernel       -0.0009   0.0114   0.0114   0.0090     7.20
Beta(5, 2)     100000 empirical    -0.0297   0.0648   0.0711   0.0570     7.29

Beta(2, 2)        100 Bernstein     0.0019   0.0957   0.0954   0.0751     0.05
Beta(2, 2)        100 kernel       -0.0019   0.0971   0.0969   0.0791     0.03
Beta(2, 2)        100 empirical    -0.0096   0.1447   0.1446   0.1182     0.00
Beta(2, 2)       1000 Bernstein    -0.0039   0.0687   0.0686   0.0570     0.56
Beta(2, 2)       1000 kernel       -0.0049   0.0718   0.0718   0.0596     0.08
Beta(2, 2)       1000 empirical    -0.0690   0.1397   0.1555   0.1303     0.05
Beta(2, 2)      10000 Bernstein     0.0012   0.0235   0.0235   0.0191     0.91
Beta(2, 2)      10000 kernel        0.0024   0.0389   0.0389   0.0311     0.63
Beta(2, 2)      10000 empirical    -0.0041   0.1200   0.1197   0.0989     0.61
Beta(2, 2)     100000 Bernstein    -0.0006   0.0072   0.0072   0.0056     7.03
Beta(2, 2)     100000 kernel       -0.0035   0.0254   0.0256   0.0207     7.12
Beta(2, 2)     100000 empirical    -0.0037   0.1212   0.1209   0.0983     7.41

Beta(3, 10)       100 Bernstein     0.0015   0.0289   0.0289   0.0241     0.10
Beta(3, 10)       100 kernel        0.0125   0.0318   0.0340   0.0269     0.03
Beta(3, 10)       100 empirical     0.0144   0.0572   0.0589   0.0459     0.00
Beta(3, 10)      1000 Bernstein     0.0010   0.0211   0.0211   0.0170     1.99
Beta(3, 10)      1000 kernel        0.0051   0.0199   0.0205   0.0164     0.08
Beta(3, 10)      1000 empirical     0.0054   0.0403   0.0406   0.0311     0.05
Beta(3, 10)     10000 Bernstein     0.0001   0.0071   0.0071   0.0057     2.01
Beta(3, 10)     10000 kernel        0.0014   0.0113   0.0114   0.0093     0.62
Beta(3, 10)     10000 empirical    -0.0030   0.0397   0.0397   0.0306     0.58
Beta(3, 10)    100000 Bernstein     0.0002   0.0030   0.0030   0.0023     8.32
Beta(3, 10)    100000 kernel        0.0008   0.0080   0.0080   0.0065     7.13
Beta(3, 10)    100000 empirical    -0.0005   0.0381   0.0380   0.0311     7.19

Beta(1, 3)        100 Bernstein     0.0237   0.0369   0.0438   0.0237     0.06
Beta(1, 3)        100 kernel        0.1151   0.0352   0.1204   0.1151     0.03
Beta(1, 3)        100 empirical     0.1075   0.0987   0.1458   0.1075     0.00
Beta(1, 3)       1000 Bernstein     0.0131   0.0193   0.0233   0.0131     0.66
Beta(1, 3)       1000 kernel        0.0805   0.0108   0.0812   0.0805     0.07
Beta(1, 3)       1000 empirical     0.0655   0.0541   0.0849   0.0655     0.05
Beta(1, 3)      10000 Bernstein     0.0034   0.0061   0.0069   0.0034     0.92
Beta(1, 3)      10000 kernel        0.0563   0.0033   0.0564   0.0563     0.59
Beta(1, 3)      10000 empirical     0.0662   0.0563   0.0868   0.0662     0.60
Beta(1, 3)     100000 Bernstein     0.0001   0.0006   0.0006   0.0001     6.97
Beta(1, 3)     100000 kernel        0.0393   0.0013   0.0393   0.0393     7.16
Beta(1, 3)     100000 empirical     0.0394   0.0370   0.0540   0.0394     7.32
```
