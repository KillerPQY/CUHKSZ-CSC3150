## Assignment 4 Report

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

#### FCB

First of all, I need to design a reasonable FCB. I designed it as follows.

1. 0~19 bytes are used to record the file name.
2. 20~21 bytes are used to record the block address of the file.
3. 22~23 bytes are used to record the size of the file.
4. 24~25 bytes are used to record the created time of the file.
5. 26~27 bytes are used to record the modified time of the file.
6. 28 byte is used to record whether the corresponding file is valid or not.

Thus, I need to initialize the FCBs at the beginning.

```c
for (int i = 0; i < FCB_ENTRIES; i++) {
  // valid: 0x00; invalid: 0xff;
  *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28) = 0xff;
}
// other bytes are 0s, so no need to modify.
```

#### fs_open

For `fs_open()` function, it will return the file pointer of the file with the given name. And if the file is not found and the op is `G_WRITE`, it will create a new file.

First, I need a function to compare the file name. Since cuda can not use the `string.h` library, I write the `strcmp()` function by myself.

```c
__device__ bool strcmp(char *a, char *b)
```

Then, I need to judge whether it find the file and whether the FCBs is full. 

```c
// if find the file, fp=file pointer, tmpt=-1
// if don't find the file and isn't full, fp=1025, tmpt=the next empty file pointer
// if don't find the file and is full, fp=1025, tmpt=-1
__device__ void find_file(FileSystem *fs, char *file_name, u32 &fp, int &tmpt)
```

According to the values of `fp` and `tmpt`, I can judge the situation and do the corresponding operation.

- fp != 1025: return fp.
- fp == 1025, op == G_READ: no such file exists.
- fp == 1025, op == G_WRITE, tmpt == -1: file number has arrived the maximun.
- fp == 1025, op == G_WRITE, tmpt != -1: create the new file.

 #### fs_read

For `fs_open()` function, it will read the data from the file to the given output buffer.

First of all, I need to judge whether the given `fp` is valid.

```c
__device__ bool check_fp_valid(FileSystem *fs, u32 fp)
```

Then, I need the compare the read size and the file size. Only when read size <= file size, fs_read can work.

Last, we just get the address of data and read them into output buffer.

```c
int addr_data = (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21)) * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS;
for (int i = 0; i < size; i++) {
	output[i] = *(fs->volume + addr_data + i);
}
```

#### fs_write

For `fs_write()` function, it will write data to the file given by the file pointer.

First of all, I need to check the fp and the write size.

Then, I need a function to update the superblock and a function to move all the memory blocks after the file indicated by the file pointer properly according to the new size of the file.

```c
// free Space Management: manage the superblocks by using the bit map
__device__ void change_sb(FileSystem *fs, int step, int init, int set)
```

```c
// record the address of the next memory block after the last used block
__device__ __managed__ u32 next_block = 0;
// compact all the memory blocks when the file has a new size
// solve the memory fragmentation problem
// return the new start addres of the file
__device__ u32 compact_blocks(FileSystem *fs, int size, u32 fp)
```

For the `compact_blocks` function, I need a global variable `next_block` to record the address of the next memory block after the last used block. And the `compact_blocks` function implemented by:
1. First, it get the original file size and compare it with the new file size.
2. If the original file size is 0, we can just return the address of the first free memory block.
3. If the original file size == new file size, we can just return the original data storage address.
4. If the original file size > new file size, we need to move the memory blocks by (original file size-new file size) steps, and then return the address of the first free memory block.
5. If the original file size < new file size, we need to move the memory blocks by (original file size) steps, and then return the original data storage address.
6. When moving blocks, we need to change the FCBs and superblocks of the files which are stored behind the writing file.
7. After moving blocks, we need to change the FCB and superblock of the writing file.

By using `compact_blocks`, I can get the start of the writing address. Then, I can begin to write data:

1. writing data.
2. update the global variable next_block.
3. update the superblock of the writing file.
4. reset the superblock of the files which has been empty if original_size > size.
5. update the fcb of the files which have been influenced.
6. update the fcb of the writing file.

#### fs_gsys (sort)

For `fs_gsys (sort)` function, it will list information about files according to the opcode.

For `LS_D` opcode, we can traverse all the valid FCBs, get the modified times and file names to list them.

```c
// print files' names after sorting by modified time
__device__ void sort_modified(FileSystem *fs)
```

For `LS_S` opcode, it need to list all files name and size in the directory and order by size and if there are several files with the same size, then first create first print. Thus, I need a function to compare file size and created time.

```c
// compare file size, if they are the same, compare created time
__device__ bool compare_file(FileSystem *fs, u32 a, u32 b)
```

Then, I need a function to find the file with the the max file size so that it can be the start condition of the traversal.

```c
// find the index of the file with the max file size
__device__ int find_max_size_file(FileSystem *fs)
```

Then, we can traverse all the FCBs to print the file names ordered by file size.

```c
// print files' names after sorting by file size
__device__ void sort_size(FileSystem *fs, int max_file_pt)
```

#### fs_gsys (remove)

For `fs_gsys (remove)` function, it will delete a file given by the file name and release the file space.

First, we can use `find_file()` function to find the file pointer.

Then, if we actually find the file, we can start removing the file.

1. Using `compact_blocks()` function to compact blocks so that clean the data.
2. Using `change_sb()` function to reset the superblock.
3. Update the FCB of the removing file, update the current file number `gtime`.
4. Update the created time and modified time of other files which were influenced.

For the 4th step, I need a function to traverse all the FCBs to find the files which were influenced and update their created time and modified time.

```c
// update the created time and modified time of other files after removing the file whose index is fp
__device__ void update_time(FileSystem *fs, u32 fp)
```

### Problems

The problems I met when doing this assignment were how to design the FCB and how to compact the blocks. After having the tutorials, I got that I could just record the information that I need to design the FCB. For compacting problem, I let it to find a new start address according to the new file size. What I always ignore is that the influence for other files. I debugged many times to figure out the influence for other files after compacting the block memory.

### Execution Steps

1. Login in **the cluster**.

2. Using the script named "slurm.sh" to execute.

   ```bash
   bash slurm.sh
   ```

### Output

#### Test 1

The output for test case 1 was shown in the following.

![image-20221125140753872](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125140753872.png)

One file named "snapshot.bin" was generated.

![image-20221125141002724](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141002724.png)

#### Test 2

The output for test case 2 was shown in the following.

![image-20221125141102664](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141102664.png)

One file named "snapshot.bin" was generated.

![image-20221125141002724](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141002724.png)

#### Test 3

The output for test case 3 was shown in the following. (The screenshot was incomplete).

![image-20221125141252277](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141252277.png)

One file named "snapshot.bin" was generated. (The screenshots were incomplete).

![image-20221125141446247](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141446247.png)

![image-20221125141458713](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141458713.png)

#### Test 4

The output for test case 4 was shown in the following. (The screenshot was incomplete).

![image-20221125141620666](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141620666.png)

One file named "snapshot.bin" was generated. (The screenshot was incomplete).

![image-20221125141730564](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 4\res\image-20221125141730564.png)

### Learning

I learnt a deeper understanding of the file system. And I learnt how to use superblocks to manage free block space. I also knew the mechanism of compacting block memory. In addition, I learnt the FCBs are very import, and we can get many useful information about files according to the FCBs.