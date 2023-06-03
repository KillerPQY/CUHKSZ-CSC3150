#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

struct FileSystem
{
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
						int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
						int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
						int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ bool strcmp(char *a, char *b);													// verify that two file name are the same
__device__ void find_file(FileSystem *fs, char *file_name, u32 &FCB_index, int &tmp_index); // find the file
__device__ u32 fs_open(FileSystem *fs, char *s, int op);

__device__ bool check_fp_valid(FileSystem *fs, u32 fp); // check the file pointer and the valid bit, invalid-false, valid-true
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);

__device__ void update_super_block(FileSystem *fs, int shift, int start, int set); // Free Space Management: update super block, bit map
__device__ u32 move_block(FileSystem *fs, int size, u32 fp);					   // move all the memory blocks when the file has a new size
__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp);

__device__ bool compare_file(FileSystem *fs, u32 a, u32 b); // compare file size, if they are the same, compare created time
__device__ int find_max_size_file(FileSystem *fs);			// find the index of the file with the max file size
__device__ void sort_modified(FileSystem *fs);				// print files' names after sorting by modified time
__device__ void sort_size(FileSystem *fs, int max_file_pt); // print files' names after sorting by file size
__device__ void fs_gsys(FileSystem *fs, int op);

__device__ void update_time(FileSystem *fs, u32 FCB_index); // update the created time and modified time of other files after removing the file whose index is fp
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

#endif