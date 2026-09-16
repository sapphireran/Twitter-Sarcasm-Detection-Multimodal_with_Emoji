# Attention replay

[ok] uniform on identical rows                 max|err|=5.83e-10
[ok] context equals the repeated row           max|err|=7.36e-09
[ok] peak step (expected 3): 3  mass=0.279
[ok] masked step has zero weight               max|err|=0.00e+00
[ok] masked rows still sum to 1                max|err|=1.31e-08
[ok] tanh saturation: huge bias step0=0.238 vs aligned step5=0.237
[ok] weak content + modest bias: step0=0.256 step5=0.123
[ok] all-masked row stays finite
