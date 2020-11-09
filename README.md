# NVIDIA GPU benchmark

Hello, I have prepared two speed tests for you on NVIDIA GPUs that I have access to.

**WARNING :** Instead of evaluating these GPUs alone, I recommend you to examine them with all their hardware, these GPUs may give different results in different applications or tests at different times.

* 1 . Speed test : I created four matrices with 10000 rows and 10000 columns on the GPU. First I multiply matrix a and b and assign it to variable y, then I multiply matrix c and d and assign it to variable z, and finally I multiply matrix y and z and assign it to variable x, and I did this operation 1000 times in total.

```python
import time, torch

bas = time.time()

a = torch.rand(10000, 10000, device=torch.device("cuda"))
b = torch.rand(10000, 10000, device=torch.device("cuda"))
c = torch.rand(10000, 10000, device=torch.device("cuda"))
d = torch.rand(10000, 10000, device=torch.device("cuda"))


for i in range(0,1000):

  y = a@b
  z = c@d
  x = y@z

son = time.time()

print("1.test result (second) : " + str(son-bas))
```

* 2 . Speed test : With C ++, I manually allocated two places in the GPU memory (10000 rows and 10000 columns) and assigned values to these reserved areas with loops. Then I multiplied these matrices with each other.

```c
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <math.h>
#include <stdio.h>

using namespace std;

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 10000;
    int SIZE = N*N;

    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    printf("END");

    return 0;
}
```


# Let's get to know the contestants :)

There are a total of 4 GPU bananas belonging to the Tesla series, let's examine them in order.

* 1-) [4 X NVIDIA Tesla V100 GPU](/4XTesla_V100.ipynb)

* 2-) [NVIDIA Tesla P4 GPU](Tesla_P4.ipynb)

* 3-) [NVIDIA Tesla P100 GPU](Tesla_P100.ipynb)

* 4-) [NVIDIA Tesla T4 GPU](Tesla_T4.ipynb)


#### 4 X NVIDIA Tesla V100 GPU

![a](https://user-images.githubusercontent.com/54184905/98583208-0f991080-22d5-11eb-9f0c-21a611229e78.png)

![Screenshot_2020-11-09_21-47-19](https://user-images.githubusercontent.com/54184905/98583318-39523780-22d5-11eb-859e-d342cc79aa9c.png)

![Screenshot_2020-11-09_21-48-35](https://user-images.githubusercontent.com/54184905/98583442-6ef72080-22d5-11eb-988e-f6a949f13d41.png)

* Yes, as you can see, we have a machine with 4 Tesla V100 GPUs(It has 64GB of video memory.) in total and we also have a 16-core Intel (R) Xeon (R) CPU.


#### NVIDIA Tesla P4 GPU

![Screenshot_2020-11-09_21-56-45](https://user-images.githubusercontent.com/54184905/98584704-54be4200-22d7-11eb-890a-475025962b4e.png)

![Screenshot_2020-11-09_21-57-27](https://user-images.githubusercontent.com/54184905/98584706-5556d880-22d7-11eb-84ac-ddffde0eb622.png)

![Screenshot_2020-11-09_21-57-52](https://user-images.githubusercontent.com/54184905/98584701-5425ab80-22d7-11eb-9a2d-43ebd42a97d2.png)

* We have a Tesla P4 GPU with 7.6GB of video memory, we also have an Intel (R) Xeon (R) CPU with 1 core.


#### NVIDIA Tesla P100 GPU

![Screenshot_2020-11-09_22-06-28](https://user-images.githubusercontent.com/54184905/98585334-28ef8c00-22d8-11eb-8023-7c4510ebbbf3.png)

![Screenshot_2020-11-09_22-06-34](https://user-images.githubusercontent.com/54184905/98585336-29882280-22d8-11eb-8d2f-740bb5227e00.png)

![Screenshot_2020-11-09_22-06-41](https://user-images.githubusercontent.com/54184905/98585332-2856f580-22d8-11eb-9027-1821d3407c5c.png)

* We have a Tesla P100 GPU with 16.2GB of video memory, we also have a 1-core Intel (R) Xeon (R) CPU.


#### NVIDIA Tesla T4 GPU

![Screenshot_2020-11-09_22-10-35](https://user-images.githubusercontent.com/54184905/98585648-a7e4c480-22d8-11eb-9dad-0b4c612b7b05.png)

![Screenshot_2020-11-09_22-10-48](https://user-images.githubusercontent.com/54184905/98585649-a7e4c480-22d8-11eb-9258-fffa0b199a14.png)

![Screenshot_2020-11-09_22-10-57](https://user-images.githubusercontent.com/54184905/98585644-a6b39780-22d8-11eb-939c-12d81b0c63e8.png)

* We have a Tesla T4 GPU with 15 GB of video memory, we also have a 1-core Intel (R) Xeon (R) CPU.


# Test 1 Results

1 . Let's recall what our test is. I created four matrices with 10000 rows and 10000 columns on the GPU. First I multiply matrix a and b and assign it to variable y, then I multiply matrix c and d and assign it to variable z, and finally I multiply matrix y and z and assign it to variable x, and I did this operation 1000 times in total.

```python
import time, torch

bas = time.time()

a = torch.rand(10000, 10000, device=torch.device("cuda"))
b = torch.rand(10000, 10000, device=torch.device("cuda"))
c = torch.rand(10000, 10000, device=torch.device("cuda"))
d = torch.rand(10000, 10000, device=torch.device("cuda"))


for i in range(0,1000):

  y = a@b
  z = c@d
  x = y@z

son = time.time()

print("1.test result (second) : " + str(son-bas))
```

#### Performance of GPUs, in seconds

* **1-) 4 X NVIDIA Tesla V100 GPU : 291.4778277873993 (Second), about 4.85 minutes.**

* **2-) NVIDIA Tesla P4 GPU : 1071.427838563919 (Second), about 17.85 minutes.**

* **3-) NVIDIA Tesla P100 GPU : 479.9311819076538 (Second), about 7.99 minutes.**

* **4-) NVIDIA Tesla T4 GPU : 1293.739860534668 (Second), about 21.56 minutes.**

**Our machine with 4 X NVIDIA Tesla V100 GPU won this race.**

![dsa](https://user-images.githubusercontent.com/54184905/98589101-d022f200-22dd-11eb-8dde-3863856880ea.png)


# Test 2 Results

2 . Let's recall what our test is. With C ++, I manually allocated two places in the GPU memory (10000 rows and 10000 columns) and assigned values to these reserved areas with loops. Then I multiplied these matrices with each other.

```c
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <math.h>
#include <stdio.h>

using namespace std;

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 10000;
    int SIZE = N*N;

    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    printf("END");

    return 0;
}
```


#### Test-2a, Performance of GPUs, in seconds :

**Test 2-a, let's first see which GPU will compile the Cuda file named matrixmul.cu. The files are compiled with nvcc (Cuda compiler).**

```python
# compilation test
bas = time.time()
!nvcc matrixmul.cu
son = time.time()

print("2.test-a result (second) : " + str(son-bas))
```

* **1-) 4 X NVIDIA Tesla V100 GPU : 1.413379192352295 (Second).**

* **2-) NVIDIA Tesla P4 GPU : 2.9613592624664307 (Second).**

* **3-) NVIDIA Tesla P100 GPU : 1.4539947509765625 (Second).**

* **4-) NVIDIA Tesla T4 GPU : 1.6754465103149414 (Second).**

**The machine with 4X NVIDIA Tesla V100 GPU won the race 2-a by a small margin.**
![tst2](https://user-images.githubusercontent.com/54184905/98590406-d2864b80-22df-11eb-9947-b4d40876ad23.png)


#### Test-2b, Performance of GPUs, in seconds :

**Test 2-b, which GPU will be able to finish running the compiled file first.**

```python
# run the compiled file, test
bas = time.time()
!./a.out
son = time.time()

print("2.test-b result (second) : " + str(son-bas))
```

* **1-) 4 X NVIDIA Tesla V100 GPU : 9.453376293182373 (Second).**

* **2-) NVIDIA Tesla P4 GPU : 8.686630487442017 (Second).**

* **3-) NVIDIA Tesla P100 GPU : 8.072553873062134 (Second).**

* **4-) NVIDIA Tesla T4 GPU : 8.99604868888855 (Second).**

**The machine with 4X NVIDIA Tesla P100 GPU won the race 2-b by a small margin.**

![dsaddd](https://user-images.githubusercontent.com/54184905/98591846-e468ee00-22e1-11eb-9048-969834a7d388.png)


# My own conclusions based on these results

* According to my observations, in short and simple operations, all GPUs, regardless of GPU video memory and CPU, can finish in a very short and close time.

* But in long and laborious calculations, high GPU memory and a good CPU allow it to stand out from other competitors.

* If we look at the graphics and results, today's winner is 4 X NVIDIA Tesla V100 GPUs :).

* **WARNING :** Instead of evaluating these GPUs alone, I recommend you to examine them with all their hardware, these GPUs may give different results in different applications or tests at different times.

* GPUs have been accessed via Google Colab and AWS.
