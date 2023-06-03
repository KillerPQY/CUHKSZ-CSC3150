#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

const int COEF = 256; // 1 byte = 8 bit, max number of 8 bit is 256
__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // init FCB
  // 0~19: file_name; 20~21 block_address; 22~23 file_size; 24~25 created_time; 26~27 modified_time; 28 valid_bit;
  for (int i = 0; i < FCB_ENTRIES; i++)
  {
    // valid: 0; invalid: 0xff;
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28) = 0xff;
  }
}

// verify that two file name are the same
__device__ bool strcmp(char *a, char *b)
{
  while (*a == *b)
  {
    if (*a == '\0' && *b == '\0')
    {
      return true;
    }
    a = a + 1;
    b = b + 1;
  }
  return false;
}

// according to the file name, find the file index
// if not exist, find the next file number if it has not arrived 1024
// if full, return 1025 (most 1024 files)
__device__ void find_file(FileSystem *fs, char *file_name, u32 &fp, int &tmpt)
{
  // int fp = 1025; // most 1024 files
  // int tmpt = -1;   // record the next file number if it has not arrived 1024
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    // valid file
    if (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28) != 0xff)
    {
      // find the file
      if (strcmp((char *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i], file_name))
      {
        fp = i;
        break;
      }
    }
    // check that whether it has arrived 1024 files
    else
    {
      if (tmpt == -1)
      {
        tmpt = i;
      }
    }
  }
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  /* Implement open operation here */
  u32 fp = 1025; // most 1024 files
  int tmpt = -1; // record the next file number if it has not arrived 1024
  find_file(fs, s, fp, tmpt);

  // find the file
  if (fp != 1025)
  {
    return fp;
  }
  else
  {
    if (op == G_READ)
    {
      printf("No such file exists.\n");
      return fp;
    }
    else if (op == G_WRITE)
    {
      if (tmpt == -1)
      {
        printf("File number has arrived the maximun: 1024.\n");
        return fp;
      }
      else
      {
        int addr_fcb = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * tmpt;

        // file name
        int name_len = 0;
        uchar t = *s;
        char *s_pt = s;
        while (t != '\0')
        {
          *(fs->volume + addr_fcb + name_len) = t;
          name_len = name_len + 1;
          if (name_len == fs->MAX_FILENAME_SIZE)
          {
            printf("File name's length is too long.\n");
            return fp;
          }
          s_pt = s_pt + 1;
          t = *s_pt;
        }
        *(fs->volume + addr_fcb + name_len) = '\0';

        fp = tmpt;
        // file size
        *(fs->volume + addr_fcb + 22) = 0;
        *(fs->volume + addr_fcb + 23) = 0;
        // file create time
        *(fs->volume + addr_fcb + 24) = gtime / COEF;
        *(fs->volume + addr_fcb + 25) = gtime % COEF;
        // file modified time
        *(fs->volume + addr_fcb + 26) = gtime / COEF;
        *(fs->volume + addr_fcb + 27) = gtime % COEF;
        gtime = gtime + 1;
        // file valid
        *(fs->volume + addr_fcb + 28) = 0;
        return fp;
      }
    }
    else
    {
      printf("The input op is not supported.\n");
      return fp;
    }
  }
}

// check the file pointer and the valid bit, invalid-false, valid-true
__device__ bool check_fp_valid(FileSystem *fs, u32 fp)
{
  // max file number is 1024, i.e. pointer range is [0~1023]
  // valid bit, invalid-0xff, valid-0x00
  if (fp >= 0 && fp < 1024 && *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 28) == 0x00)
  {
    return true;
  }
  else
  {
    return false;
  }
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  /* Implement read operation here */
  // check the file pointer and the valid bit
  if (check_fp_valid(fs, fp))
  {
    // reading size too large
    int original_size = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23);
    if (original_size < size)
    {
      printf("Reading size is larger than exist file size.\n");
      return;
    }
    else
    {
      // get the storage address and read the data in the file
      int addr_data = (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21)) * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS;
      for (int i = 0; i < size; i++)
      {
        output[i] = *(fs->volume + addr_data + i);
      }
      return;
    }
  }
  else
  {
    return;
  }
}

// record the address of the next memory block after the last used block
__device__ __managed__ u32 next_block = 0;

// free Space Management: manage the superblocks by using the bit map
__device__ void change_sb(FileSystem *fs, int step, int init, int set)
{
  for (int i = 0; i < step; i++)
  {
    int addr = init + i;
    // invalid superblock address
    if (addr >= 0 && addr <= 1024)
    {
      int r = addr / 8;
      int c = addr % 8;

      // use mask to change the bit map in superblock
      uchar m = (1 << c);
      // set: 1-set the bit map; 0-clean
      switch (set)
      {
      case 1:
        *(fs->volume + r) |= m;
        break;
      case 0:
        m = ~m;
        *(fs->volume + r) &= m;
        break;
      default:
        printf("Error set parameter!");
        break;
      }
    }
  }
}

// compact all the memory blocks when the file has a new size
// solve the memory fragmentation problem
// return the new start addres of the file
__device__ u32 compact_blocks(FileSystem *fs, int size, u32 fp)
{
  // get the original file size
  int original_size = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23);

  // original_size == 0, fp is the address of the first free memory block
  if (original_size == 0)
  {
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20) = (next_block / fs->STORAGE_BLOCK_SIZE) / COEF;
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21) = (next_block / fs->STORAGE_BLOCK_SIZE) % COEF;
    return next_block + fs->FILE_BASE_ADDRESS;
  }
  else
  {
    // original_size != 0
    int addr_data = (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21)) *
                        fs->STORAGE_BLOCK_SIZE +
                    fs->FILE_BASE_ADDRESS;
    // original_size == size, fp is the original storage address
    if (original_size == size)
    {
      return addr_data;
    }
    else
    {
      // original_size > size, compact step is (original_size-size)
      // original_size < size, compact step is original_size
      int step;
      if (original_size < size)
      {
        step = original_size;
      }
      else if (original_size > size)
      {
        step = original_size - size;
      }
      // calculate the block moving step
      int block_step = step / fs->STORAGE_BLOCK_SIZE;
      if (original_size % fs->STORAGE_BLOCK_SIZE != 0)
      {
        block_step = block_step + 1;
      }
      // update the fcb and superblock of the files whcih behind the operating file
      for (int i = 0; i < fs->FCB_ENTRIES; i++)
      {
        if (i != fp && *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28) != 0xff)
        {
          int addr_block_start = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 20) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 21);
          if (addr_block_start * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS > addr_data)
          {
            addr_block_start = addr_block_start - block_step;
            // update the fcb
            *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 20) = addr_block_start / COEF;
            *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 21) = addr_block_start % COEF;
            // update the superblock
            change_sb(fs, block_step, addr_block_start, 1);
          }
        }
      }
      // update the fcb and superblock of the operating file
      int cur_block_step;
      if (original_size > size)
      {
        cur_block_step = addr_data + size;
      }
      else if (original_size < size)
      {
        cur_block_step = addr_data;
      }
      for (int i = 0; i < block_step; i++)
      {
        int block_init = fs->STORAGE_BLOCK_SIZE * i + cur_block_step;
        for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++)
        {
          *(fs->volume + block_init + j) = *(fs->volume + block_init + j + fs->STORAGE_BLOCK_SIZE * block_step);
        }
      }
      // the global variable next_block change
      next_block = next_block - fs->STORAGE_BLOCK_SIZE * block_step;
      int retval;
      // original_size > size, block address no change
      if (original_size > size)
      {
        retval = addr_data;
      }
      // original_size < size, block address change
      else if (original_size < size)
      {
        *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20) = (next_block / fs->STORAGE_BLOCK_SIZE) / COEF;
        *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21) = (next_block / fs->STORAGE_BLOCK_SIZE) % COEF;
        retval = next_block + fs->FILE_BASE_ADDRESS;
      }
      return retval;
    }
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{
  /* Implement write operation here */
  // check the file pointer, the valid bit and the writing size
  if (check_fp_valid(fs, fp) && size < fs->MAX_FILE_SIZE)
  {
    // get the start of the writing address
    int addr_data_init = compact_blocks(fs, size, fp);

    // 1. writing data
    for (int i = 0; i < size; i++)
    {
      *(fs->volume + addr_data_init + i) = input[i];
    }

    int original_s = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23);
    // compact block number
    int block_step = size / fs->STORAGE_BLOCK_SIZE;
    if (size % fs->STORAGE_BLOCK_SIZE != 0)
    {
      block_step = block_step + 1;
    }

    // 2. update the global variable next_block
    next_block = next_block + fs->STORAGE_BLOCK_SIZE * block_step;

    // 3. update the superblock of the writing file
    int block_init = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20] * COEF \ 
    + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21];
    change_sb(fs, block_step, block_init, 1);

    // 4. reset the superblock of the files which has been empty if original_size > size
    block_init = next_block / fs->STORAGE_BLOCK_SIZE;
    if (original_s > size)
    {
      int tmp_step = (original_s - size) / fs->STORAGE_BLOCK_SIZE;
      if (tmp_step % fs->STORAGE_BLOCK_SIZE != 0)
      {
        tmp_step = tmp_step + 1;
      }
      change_sb(fs, tmp_step, block_init, 0);
    }

    // 5. update the fcb of the files which have been influenced
    // original modified time
    int omt = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 26) \ 
    * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 27);
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      if (i != fp && *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28) != 0xff)
      {
        // temp modified time
        int tmt = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26) * COEF \ 
        + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27);
        if (tmt > omt)
        {
          tmt = tmt - 1;
          *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26) = tmt / COEF;
          *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27) = tmt % COEF;
        }
      }
    }

    // 6. update the fcb of the writing file
    // file size
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22) = size / COEF;
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23) = size % COEF;
    // modified time
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 26) = (gtime - 1) / COEF;
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 27) = (gtime - 1) % COEF;
    // valid
    *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 28) = 0x00;
    return 0;
  }
  else
  {
    return 1025; // 1025 out of range, see it as an error
  }
}

// compare file size, if they are the same, compare created time
__device__ bool compare_file(FileSystem *fs, u32 a, u32 b)
{
  // compare file size
  int s1 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * a + 22] * COEF + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * a + 23];
  int s2 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * b + 22] * COEF + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * b + 23];
  if (s1 > s2)
  {
    return true;
  }
  else if (s1 < s2)
  {
    return false;
  }
  else
  {
    // compare created time
    int t1 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * a + 24] * COEF + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * a + 25];
    int t2 = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * b + 24] * COEF + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * b + 25];
    if (t1 < t2)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
}

// find the index of the file with the max file size
__device__ int find_max_size_file(FileSystem *fs)
{
  int m = -1;
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    if (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28) != 0xff)
    {
      if (m == -1)
      {
        m = i;
      }
      else
      {
        if (compare_file(fs, i, m))
        {
          m = i;
        }
      }
    }
  }
  return m;
}

// print files' names after sorting by modified time
__device__ void sort_modified(FileSystem *fs)
{
  int last_file_pt = gtime - 1; // point to the last modified time file
  for (int i = 0; i < gtime; i++)
  {
    for (int j = 0; j < fs->FCB_ENTRIES; j++)
    {
      if (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 28) != 0xff)
      {
        // modified time
        int mt = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 26) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 27);
        if (mt == last_file_pt)
        {
          // file name
          uchar *fn = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j];
          while (*fn != '\0')
          {
            printf("%c", (char)*fn);
            fn = fn + 1;
          }
          printf("\n");
          last_file_pt = last_file_pt - 1;
          break;
        }
      }
    }
  }
}

// print files' names after sorting by file size
__device__ void sort_size(FileSystem *fs, int max_file_pt)
{
  for (int i = 0; i < gtime - 1; i++)
  {
    int tmp_max_file = -1;
    // find the second max size file index
    for (int j = 0; j < fs->FCB_ENTRIES; j++)
    {
      if (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 28) != 0xff)
      {
        if (tmp_max_file == -1 && compare_file(fs, max_file_pt, j))
        {
          tmp_max_file = j;
        }
        else
        {
          if (compare_file(fs, j, tmp_max_file) && compare_file(fs, max_file_pt, j))
          {
            tmp_max_file = j;
          }
        }
      }
    }
    uchar *fn = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * tmp_max_file];
    int max_size = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * tmp_max_file + 22) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * tmp_max_file + 23);
    while (*fn != '\0')
    {
      printf("%c", (char)*fn);
      fn = fn + 1;
    }
    printf(" %d\n", max_size);

    // update the max file index so that we find the next one
    max_file_pt = tmp_max_file;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
  /* Implement LS_D and LS_S operation here */
  // list all files name in the directory and order by modified time of files
  if (op == LS_D)
  {
    printf("===sort by modified time===\n");
    sort_modified(fs);
  }
  // list all files name and size in the directory and order by size
  else if (op == LS_S)
  {
    printf("===sort by file size===\n");

    // find the max size file and print it
    int max_size_file = find_max_size_file(fs);
    int max_size = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * max_size_file + 22) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * max_size_file + 23);
    uchar *fn = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * max_size_file];
    // print the max file
    while (*fn != '\0')
    {
      printf("%c", (char)*fn);
      fn = fn + 1;
    }
    printf(" %d\n", max_size);
    // print continue max files
    sort_size(fs, max_size_file);
  }
  // invalid op input
  else
  {
    printf("The input op is not supported.\n");
  }
}

// update the created time and modified time of other files after removing the file whose index is fp
__device__ void update_time(FileSystem *fs, u32 fp)
{
  // original created time
  int oct = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 24) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 25);
  // originnal modified time
  int omt = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 26) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 27);
  int cnt = 0; // file count
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    if (*(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28) != 0xff)
    {
      // temp created time and temp modified time
      int tct = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 24) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25);
      int tmt = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27);
      // update created time and modified time of other files
      if (tct > oct)
      {
        tct = tct - 1;
        *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 24) = tct / COEF;
        *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25) = tct % COEF;
      }
      if (tmt > omt)
      {
        tmt = tmt - 1;
        *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26) = tmt / COEF;
        *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27) = tmt % COEF;
      }
      cnt = cnt + 1;
    }
    if (gtime == cnt)
    {
      break;
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  /* Implement rm operation here */
  if (op == RM)
  {
    u32 fp = 1025; // most 1024 files
    int tmpt = -1; // record the next file number if it has not arrived 1024
    find_file(fs, s, fp, tmpt);
    if (fp != 1025)
    {
      int file_size = *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22) * COEF + *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23);
      // block number
      int blocks = file_size / fs->STORAGE_BLOCK_SIZE;
      if (file_size % fs->STORAGE_BLOCK_SIZE != 0)
      {
        blocks = blocks + 1;
      }
      // 1. compact blocks so that clean the data
      compact_blocks(fs, 0, fp);

      // 2. reset the superblock
      u32 block_begin = next_block / fs->STORAGE_BLOCK_SIZE;
      change_sb(fs, blocks, block_begin, 0);

      // 3. update the fcb
      // invalid the file
      *(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 28) = 0xff;
      // update current file number
      gtime = gtime - 1;
      // update the created time and modified time of other files which were influenced
      update_time(fs, fp);
    }
    else
    {
      printf("No file named %c.\n", *s);
    }
  }
  else
  {
    printf("The input op is not supported.\n");
  }
}
