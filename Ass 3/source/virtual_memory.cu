#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// update the page counter after accessing
__device__ void update_counter(VirtualMemory *vm, u32 page_num)
{
  // increase the times of unused page
  for (int i = 0; i < vm->PAGE_ENTRIES; i++)
  {
    vm->page_counter[i] += 1;
  }
  // reset the time of the used page
  vm->page_counter[page_num] = 0;
}

// set the valid bit: invalid-1 valid-0
__device__ void set_valid(VirtualMemory *vm, u32 page_num, int val)
{
  // invalid
  if (val == 1)
  {
    vm->invert_page_table[page_num] |= 0b10000000000000000000000000000000;
  }
  // valid
  else
  {
    vm->invert_page_table[page_num] &= 0b01111111111111111111111111111111;
  }
}

// set the dirty bit: dirty-1 clean-0
__device__ void set_dirty(VirtualMemory *vm, u32 page_num, int val)
{
  // dirty
  if (val == 1)
  {
    vm->invert_page_table[page_num] |= 0b00000000000000000000000000000001;
  }
  // clean
  else
  {
    vm->invert_page_table[page_num] &= 0b11111111111111111111111111111110;
  }
}

// use LRU algorithm to find the least recent used page, return the index of the page
__device__ int LRU(VirtualMemory *vm)
{
  int result = 0;
  u32 max_times = vm->page_counter[0];

  // traverse the page counter to find the page index of the most unused times of the page
  for (int i = 0; i < 1024; i++)
  {
    if (vm->page_counter[i] > max_times)
    {
      max_times = vm->page_counter[i];
      result = i;
    }
  }
  return result;
}

__device__ void init_invert_page_table(VirtualMemory *vm)
{

  for (int i = 0; i < vm->PAGE_ENTRIES; i++)
  {
    // valid bit is used for determining the page
    // dirty bit is used for changing page
    // use MSB as valid bit; use LSB as dirty bit
    // invalid := MSB is 1; dirty := LSB is 1
    vm->invert_page_table[i] = 0x80000000;
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr, u32 *page_counter,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES)
{
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // record the unused times of each page entry
  vm->page_counter = page_counter;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr)
{
  /* Complate vm_read function to read single element from data buffer */
  u32 page_num = addr / 32;
  u32 offset = addr % 32;
  uchar result;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++)
  {
    // find the page entry
    if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_num)
    {
      // valid
      if ((vm->invert_page_table[i] & 0b10000000000000000000000000000000) != 0b10000000000000000000000000000000)
      {
        // find the data, need to do nothing
      }
      // invalid or page table has not been full
      else
      {
        // page fault number plus 1
        (*vm->pagefault_num_ptr)++;
        // load the page from secondary memory to physical memory
        for (int j = 0; j < vm->PAGESIZE; j++)
        {
          vm->buffer[vm->PAGESIZE * i + j] = vm->storage[vm->PAGESIZE * page_num + j];
        }
        // udpate the page table
        set_valid(vm, i, 0);
        set_dirty(vm, i, 0);
      }
      // update the page counter
      update_counter(vm, i);
      // get result
      result = vm->buffer[i * vm->PAGESIZE + offset];
      return result;
    }
  }
  // did not find the page_num matched and the page table is full, need to replace page
  (*vm->pagefault_num_ptr)++; // page fault number plus 1

  /* page replacement*/
  // find the lru page num
  int lru_page_num = LRU(vm);
  // use the page num to find the frame num
  int lru_frame_num = vm->invert_page_table[lru_page_num + vm->PAGE_ENTRIES];

  // if the dirty bit is 1, remove the LRU page out
  if ((vm->invert_page_table[lru_page_num] & 0b00000000000000000000000000000001) == 0b00000000000000000000000000000001)
  {
    for (int i = 0; i < vm->PAGESIZE; i++)
    {
      vm->storage[lru_frame_num * vm->PAGESIZE + i] = vm->buffer[lru_page_num * vm->PAGESIZE + i];
    }
  }

  // load the page that we need into the buffer
  for (int i = 0; i < vm->PAGESIZE; i++)
  {
    vm->buffer[lru_page_num * vm->PAGESIZE + i] = vm->storage[page_num * vm->PAGESIZE + i];
  }

  // swap out the old page num
  vm->invert_page_table[lru_page_num + vm->PAGE_ENTRIES] = page_num;

  // update the page table
  set_valid(vm, lru_page_num, 0); // reset the valid bit 0
  set_dirty(vm, lru_page_num, 0); // reset the dirty bit 0

  // update the page counter
  update_counter(vm, lru_page_num);

  // get result
  result = vm->buffer[lru_page_num * vm->PAGESIZE + offset];
  return result;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value)
{
  /* Complete vm_write function to write value into data buffer */
  u32 page_num = addr / 32;
  u32 offset = addr % 32;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++)
  {
    // find the page entry
    if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_num)
    {
      // valid
      if ((vm->invert_page_table[i] & 0b10000000000000000000000000000000) != 0b10000000000000000000000000000000)
      {
        // find the writing position, nothing need to do
      }
      // invalid or page table has not been full
      else
      {
        // page fault number plus 1
        (*vm->pagefault_num_ptr)++;
        // load the page from secondary memory to physical memory
        for (int j = 0; j < vm->PAGESIZE; j++)
        {
          vm->buffer[vm->PAGESIZE * i + j] = vm->storage[vm->PAGESIZE * page_num + j];
        }
        // update page table, valid bit 0
        set_valid(vm, i, 0);
      }
      // update the page table, dirty bit 1
      set_dirty(vm, i, 1);
      // update the page counter
      update_counter(vm, i);
      // write the value into buffer
      vm->buffer[i * vm->PAGESIZE + offset] = value;
      return;
    }
  }
  // did not find the page_num matched and the page table is full, need to replace page
  (*vm->pagefault_num_ptr)++; // page fault number plus 1

  /* page replacement*/
  // find the lru page num
  int lru_page_num = LRU(vm);
  // use the page num to find the frame num
  int lru_frame_num = vm->invert_page_table[lru_page_num + vm->PAGE_ENTRIES];

  // if the dirty bit is 1, remove the LRU page out
  if ((vm->invert_page_table[lru_page_num] & 0b00000000000000000000000000000001) == 0b00000000000000000000000000000001)
  {
    for (int i = 0; i < vm->PAGESIZE; i++)
    {
      vm->storage[lru_frame_num * vm->PAGESIZE + i] = vm->buffer[lru_page_num * vm->PAGESIZE + i];
    }
  }

  // load the page that we need into the buffer
  for (int i = 0; i < vm->PAGESIZE; i++)
  {
    vm->buffer[lru_page_num * vm->PAGESIZE + i] = vm->storage[page_num * vm->PAGESIZE + i];
  }

  // swap out the old page num
  vm->invert_page_table[lru_page_num + vm->PAGE_ENTRIES] = page_num;
  // update the page table
  set_valid(vm, lru_page_num, 0);
  set_dirty(vm, lru_page_num, 0);

  // update the page counter
  update_counter(vm, lru_page_num);

  // write the value into buffer
  vm->buffer[vm->PAGESIZE * lru_page_num + offset] = value;
  return;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size)
{
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = 0; i < input_size; i++)
  {
    results[i] = vm_read(vm, i + offset);
  }
}
