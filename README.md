# CUDA-labs
This repository contains some code on my CUDA labs.

# Contents 
- [Element-wise vector minimum](#lab-1)

## <a href="#lab-1" id="lab-1" name="lab-1">Element-wise vector minimum</a>

See [lab_1/lab_1.cu](/lab_1/lab_1.cu).

Time elapsed comparison for different configurations:
|--------------------------------------|
| <<<32768, 1024>>> | 2.555904 ms      |
|--------------------------------------|
| <<<16384, 1024>>> | Time: 1.501184 ms|
|--------------------------------------|
| <<<8192, 512>>>   | 1.340416 ms      |
|--------------------------------------|
| <<<512, 512>>>    | 1.497088 ms      |
|--------------------------------------|
| <<<1, 32>>>       |  2.033664 ms     |
|--------------------------------------|

0.024469
0.434746