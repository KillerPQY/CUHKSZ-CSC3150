## Assignment 3 Report

### Environment

**The cluster** offered by CSC4008.

**CUDA** version:

```bash
[120090175@node21 bonus]$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

**GPU** information:
```bash
[120090175@node21 bonus]$ lspci | grep -i vga
03:00.0 VGA compatible controller: ASPEED Technology, Inc. ASPEED Graphics Family (rev 41)
af:00.0 VGA compatible controller: NVIDIA Corporation Device 1eb1 (rev a1)
[120090175@node21 bonus]$ lspci | grep -i nvidia
af:00.0 VGA compatible controller: NVIDIA Corporation Device 1eb1 (rev a1)
af:00.1 Audio device: NVIDIA Corporation Device 10f8 (rev a1)
af:00.2 USB controller: NVIDIA Corporation Device 1ad8 (rev a1)
af:00.3 Serial bus controller [0c80]: NVIDIA Corporation Device 1ad9 (rev a1)
```

### Design

First of all, we add a attribute "page_counter" for vm so that it is easy for us to implment LRU algorithm.

```c
  // record the unused times of each page entry
  u32 *page_counter;
```

Each time we use the page table, the unused times of each page entry should be updated. Thus, we implement a function to update the counter.

```c
__device__ void update_counter(VirtualMemory *vm, u32 page_num) {
  // increase the times of unused page
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->page_counter[i] += 1;
  }
  // reset the time of the used page
  vm->page_counter[page_num] = 0;
}
```

In order to realize page replacement, I use the MSB of the page entry as the valid bit and the LSB of the page entry as the dirty bit.

```c
// valid bit is used for determining the page
// dirty bit is used for changing page
// use MSB as valid bit; use LSB as dirty bit
// invalid := MSB is 1; dirty := LSB is 1
vm->invert_page_table[i] = 0x80000000;
vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
```

For **vm_read** function, we firstly extract the page number and offset according the address.

```c
  u32 page_num = addr / 32;
  u32 offset = addr % 32;
```

Then, find the page entry matched and check the valid bit

```c
for (int i = 0; i < vm->PAGE_ENTRIES; i++){
    // find the page entry matched
    if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_num) {
        // use mask to check the valid bit
        // valid
        if ((vm->invert_page_table[i] & 0b10000000000000000000000000000000) != 0b10000000000000000000000000000000) {...}
        // invalid
        else {...}
    }
}
```

If valid, we just update the page counter and read the result; if invalid, we should load the page from secondary memory to physical memory and reset the valid and dirty bit.

```c
// load the page from secondary memory to physical memory
for (int j = 0; j < vm->PAGESIZE; j++){
    vm->buffer[vm->PAGESIZE * i + j] = vm->storage[vm->PAGESIZE * page_num + j];
}
// udpate the valid and dirty bit
set_valid(vm, i, 0);
set_dirty(vm, i, 0);
```

If we not find the page_num matched, we should replace page. According to the LRU algorithm, we find the most unused page and swap it with the page we need.

```c
  // find the lru page num
  int lru_page_num = LRU(vm);
  // use the page num to find the frame num
  int lru_frame_num = vm->invert_page_table[lru_page_num + vm->PAGE_ENTRIES];
  // if the dirty bit is 1, remove the LRU page out
  if ((vm->invert_page_table[lru_page_num] & 0b00000000000000000000000000000001) == 0b00000000000000000000000000000001) {
    for (int i = 0; i < vm->PAGESIZE; i++) {
      vm->storage[lru_frame_num * vm->PAGESIZE + i] = vm->buffer[lru_page_num * vm->PAGESIZE + i];
    }
  }
  // load the page that we need into the buffer
  for (int i = 0; i < vm->PAGESIZE; i++) {
    vm->buffer[lru_page_num * vm->PAGESIZE + i] = vm->storage[page_num * vm->PAGESIZE + i];
  }
  // swap out the old page num
  vm->invert_page_table[lru_page_num + vm->PAGE_ENTRIES] = page_num;
```

Then, we just reset the valid and dirty bit and update the counter. And last, we can directly read the result.

For **vm_write** function, the overall process is almost consistent with vm_read function, and we just replace reading operation with writing operation. One thing we need to be careful is that we need set dirty bit whatever it is valid or invalid, because we write something and it has changed.

```c
set_dirty(vm, i, 1);
```

For **vm_snapshot** function, we just using vm_read to load all data.

```c
for (int i = 0; i < input_size; i++) {
    results[i] = vm_read(vm, i+offset);
}
```

### Page Fault Number

The page fault number in my program consists of 2 fault numbers.

1. The memory that needs to be accessed is not in virtual address space but is in physical address, so we only need to swap them according to the mapping between physical memory and virtual address space.
2. The valid bit of the corresponding page entry is invalid.

First, the program writes the data of address from 0 to 131071 continually. Every time when writing the page that is multiple of 32, the function will generate a page fault. Then, the next 31 data will not generate page fault, because they are in the same page of that data of address in the multiple of 32. Thus, there will be 131072/32 = 4096 page faults.

Second, the program would read 32769 bytes of data from address 131071 continually in the decreasing order. There will not generate page fault for the last 32768 bytes, since the 1024 pages with each page 32 bytes. But for the 32769th data, its page does not in the table, so there will be a page fault.

Third, the vm_snapshot function will transfer all the 131072 bytes data from address 0 consecutively in the increasing order. Since it is begin with address 0, the previous 1024 pages in the page table will not get any page hit since the pages are set for last 1024 pages of data. Therefore, there will be 131072/32 = 4096 page faults.

Totally, there are 4096+1+4096=8193 page faults.

### Problems

The first problem I met is how to mark a page table entry to indicate that it needs replacement. After googling, I found that I could use the LSB of the page table entry as the dirty bit. According to the dirty bit, I could implement the page replacing easily. 

The second problem I met is how to find the least recently used page. Firstly, I want use the double linked list to implement the LRU algorithm, which is the same as the Internet. Then, it suddenly occurred to me that I could add a page counter parameter in the vm structure to record the unused times of every page. This would be more easier than the was using double linked list, and it would not waste the memory to store the double linked list.

### Execution Steps

1. Login in **the cluster**.

2. Using the script named "slurm.sh" to execute.

   ```bash
   bash slurm.sh
   ```

### Output

#### Test 1

For test 1, the user program is

```c
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  for (int i = 0; i < input_size; i++)
    vm_write(vm, i, input[i]);

  for (int i = input_size - 1; i >= input_size - 32769; i--)
    int value = vm_read(vm, i);

  vm_snapshot(vm, results, 0, input_size);
}
```

The input size and pagefault number will be printed in the terminal.

![image-20221102192941853](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102192941853.png)

One file named "snapshot.bin" will be generated.

![image-20221102193251786](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193251786.png)

To verify the correctness of the program, use "cmp" to compare "data.bin" with "snapshot.bin".

![image-20221102193430128](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193430128.png)

No result means that it is correct.

#### Test 2

For test 2, the user program is

```c
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
  int input_size) {
// write the data.bin to the VM starting from address 32*1024
for (int i = 0; i < input_size; i++)
  vm_write(vm, 32*1024+i, input[i]);
// printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
// write (32KB-32B) data  to the VM starting from 0
for (int i = 0; i < 32*1023; i++)
  vm_write(vm, i, input[i+32*1024]);
// printf("pagefault number is %d\n", *vm->pagefault_num_ptr-4096);
// readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
vm_snapshot(vm, results, 32*1024, input_size);
// printf("pagefault number is %d\n", *vm->pagefault_num_ptr-4096-1023);
}
// expected page fault num: 9215
```

The input size and pagefault number will be printed in the terminal.

![image-20221107143807034](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221107143807034.png)

One file named "snapshot.bin" will be generated.

![image-20221102193251786](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193251786.png)

To verify the correctness of the program, use "cmp" to compare "data.bin" with "snapshot.bin".

![image-20221102193430128](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193430128.png)

No result means that it is correct.

### Learning

I learnt how to write a simple cuda program. Also, I had a much clearer understanding of how a page table work. Meanwhile, I understood the LRU algorithm and how to use it to implement page replacement. I also learnt more about the basic structure and mechanism of the virtual memory. In addition, I knew how to use  the cluster to compile and run my own programs.

### Bonus

#### Design

To launch 4 threads, I modify the kernel part of the main function in the "main.cu".

```c
mykernel<<<1, 4, INVERT_PAGE_TABLE_SIZE>>>(input_size);
```

To implement the non-preemptive priority schedule of threads, I just let each thread to read or write data of the corresponding address, i.e., only allow thread 0, 1, 2, 3 to read or write data of address that satisfy addr%4 = 0, 1, 2, 3 correspondingly. Because that only one thread will read or write, there are no threads racing for the same resource of memory.

Thus, I added the following codes.

```c
// __syncthreads() is a thread barrier, any thread reaching the barrier waits until all of the other threads in that block also reach it.
// it is used for avoiding race condition when loading shared memory

// in vm_read function
__syncthreads();
// if it is not the corresponding address, skip
if (addr % 4 != ((int)threadIdx.x))
    return;
printf("[thread %d]: vm read %d\n", threadIdx.x, addr);	// print the thread info

// in vm_write function
__syncthreads();
// if it is not the corresponding address, skip
if (addr % 4 != ((int)threadIdx.x))
    return;
printf("[thread %d]: vm write %d\n", threadIdx.x, addr); // print the thread info
```

Since we used 4 threads to execute, the iteration in vm_snapshot could be reduced. I also assigned different threads to different operations.

```c
for (int i = offset; i < input_size / 4; i++)
{
  results[i * 4 + (int)threadIdx.x] = vm_read(vm, i * 4 + (int)threadIdx.x);
}
```

#### Execution Steps

The executing way is the same as the assignment 3.
1. Login in **the cluster**.

2. Using the script named "slurm.sh" to execute.

   ```bash
   bash slurm.sh
   ```

#### Output

##### Test 1

The thread id and what it does will be printed, and finally the pagefault number will be printed.

![image-20221102193933905](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193933905.png)

One file named "snapshot.bin" will be generated.

![image-20221102193251786](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193251786.png)

To verify the correctness of the program, use "cmp" to compare "data.bin" with "snapshot.bin".

![image-20221102193430128](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193430128.png)

No result means that it is correct.

##### Test 2

The thread id and what it does will be printed, and finally the pagefault number will be printed.

![image-20221107144339564](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221107144339564.png)

One file named "snapshot.bin" will be generated.

![image-20221102193251786](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193251786.png)

To verify the correctness of the program, use "cmp" to compare "data.bin" with "snapshot.bin".

![image-20221102193430128](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 3\res\image-20221102193430128.png)

No result means that it is correct.