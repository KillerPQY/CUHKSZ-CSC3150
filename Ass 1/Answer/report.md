### Report

### Environment

version of OS: Ubuntu 16.04.7 LTS
version of kernel: 5.10.146
version of gcc: 5.4.0

### Task 1

#### Design

In this program, we should fork a child process to execute the test program in user mode. So, we can divide it into 3 steps:

1. Fork a child process to execute a test program.
2. If fork() return 0, we fork a child process to execute a test program and wait for its returning signal. If fork() return the pid, it is the parent process and it should wait for the signal of the child process and display the information of ths signal. If fork() return -1, the creation of a child process was unsuccessful, there are some errors.
3. When the parent process receives the signal, it will analyze which kind of signal it received (normal, fail, stop).

#### Steps for implementation

1. Use **fork()** function to fork a child process, check its return.

   ```c
   pid = fork();
   if (pid == -1){...}	// the creation of a child process was unsuccessful
   else{
       if (pid == 0){...}	// child process created successful
       else {...}	// parent process
   }
   ```

2. According to the value of the pid, do different operations, for parent process, using **waitpid** function to wait.

   ```c
   // pid == -1, error
   perror("fork");
   exit(1);
   
   // pid == 0, child process created successful, it start to execute test program
   int i;
   char *arg[argc];
   for (i = 0; i < argc - 1; i++)
   {
   	arg[i] = argv[i + 1];
   }
   arg[argc - 1] = NULL;
   execve(arg[0], arg, NULL);
   
   // pid == other, parent process wait for the signal of the child process and display the information
   waitpid(pid, &status, WUNTRACED);
   // Analyze the value of status to print different information
   ```

3. Using **WIFEXITED, WIFSIGNALED, WTERMSIG, WIFSTOPPED, WSTOPSIG** functions to analyze the value of status.

   ```c
   if (WIFEXITED(status)){...}	// normal termination
   else if (WIFSIGNALED(status)) {	// child execution terminated
       int sign = WTERMSIG(status);
       switch (sign){...}	// different sign display different information, details can be seen in program1.c file
   }
   else if (WIFSTOPPED(status)) {...}	// child process stopped
   else{...} // continued
   exit(0);
   // the details displaying information can be seen in program1.c file
   ```


#### Execute

```shell
# In the 'program1' directory
sudo su	# switch to the root user
make	# compile this program
./program1 xxx	# execute this program, xxx is the file for test
make clean	# clean
```

#### Output

abort

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008081635109.png" alt="image-20221008081635109" style="zoom:80%;" />

alarm

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008081722927.png" alt="image-20221008081722927" style="zoom:80%;" />

bus

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008081835845.png" alt="image-20221008081835845" style="zoom:80%;" />

floating

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008081904437.png" alt="image-20221008081904437" style="zoom:80%;" />

hangup

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008081942614.png" alt="image-20221008081942614" style="zoom:80%;" />

illegal_instr

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082026507.png" alt="image-20221008082026507" style="zoom:80%;" />

interrupt

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082057948.png" alt="image-20221008082057948" style="zoom:80%;" />

kill

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082125920.png" alt="image-20221008082125920" style="zoom:80%;" />

normal

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082200966.png" alt="image-20221008082200966" style="zoom:80%;" />

pipe

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082238404.png" alt="image-20221008082238404" style="zoom:80%;" />



quit

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082309538.png" alt="image-20221008082309538" style="zoom:80%;" />

segment_fault

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082339596.png" alt="image-20221008082339596" style="zoom:80%;" />

stop

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082411250.png" alt="image-20221008082411250" style="zoom:80%;" />

terminate

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082440332.png" alt="image-20221008082440332" style="zoom:80%;" />

trap

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008082510157.png" alt="image-20221008082510157" style="zoom:80%;" />

#### Learning

In this program, I learnt how to create a child process and check it (fork()). Also, I learnt how to use child processes to do some tasks (execve()). Meanwhile, I knew what the different signal values mean and the basic knowledge of the relationship between child and parent process.

### Task 2

#### Design

In this program, we need to create a kernel thread. Thus, it needs some functions in the linux kernel and we also need to do some modification so that these functions can be used by our program. This total program can be divide into 4 steps:

1. modify the kernel files, recompile the kernel and reboot so that we can use some functions in the kernel files.
2. create functions `my_exec`, `my_wait` and structure `wait_opts` by using `do_execve`, `do_wait`, `getname_kernel`functions in the kernel file.
3. initialize the module.
4. fork a process using `kernel_clone` and do some operations similar with program1.

#### Step for implementation

1. find functions `kernel_clone` in /kernel/fork.c, `do_execve` in /fs/exec.c, `getname_kernel` in /fs/namei.c, `do_wait` in /kernel/exit.c, add `EXPORT_SYMBOL(func)` (replace func with the function name) behind these functions, then recompile the kernel and reboot.

   ```c
   EXPORT_SYMBOL(kernel_clone);
   EXPORT_SYMBOL(do_execve);
   EXPORT_SYMBOL(do_wait);
   EXPORT_SYMBOL(getname_kernel);
   ```

2. In order to make `do_execve`, `do_wait` useful in our program, creates `my_exec`, `my_wait` functions.

   ```c
   // execute the test file
   int my_exec(void)
   {
   	int retval;
   
   	// the path of the test file
   	const char path[] = "/tmp/test"; // path to the test file
   	struct filename *my_filename = getname_kernel(path);
   
   	printk("[program2] : child process");
   	retval = do_execve(my_filename, NULL, NULL);
   
   	// success
   	if (!retval)
   		return 0;
   	do_exit(retval); // fail
   }
   
   // transfer parameters to do_wait
   long my_wait(pid_t pid)
   {
   	int status;
   	int retval;
   	struct wait_opts wo;
   	enum pid_type type;
   	type = PIDTYPE_PID;
   	wo.wo_type = type;
   	wo.wo_flags = WEXITED | WUNTRACED;
   	struct pid *wo_pid = NULL;
   	wo_pid = find_get_pid(pid);
   	wo.wo_pid = wo_pid;
   	wo.wo_info = NULL;
   	wo.wo_stat = status;
   	wo.wo_rusage = NULL;
   	do_wait(&wo);
   	retval = wo.wo_stat;
   	put_pid(wo_pid);
   	return retval;
   }
   ```

3. initialize the program2 module by using `kthread_create` and `wake_up_process`  functions.

   ```c
   static int __init program2_init(void)
   {
   	printk("[program2] : module_init Peng Qiaoyu 120090175\n");
   	// using kthread_create() to run my_fork() function
   	printk("[program2] : module_init create kthread start\n");
   	task = kthread_create(&my_fork, NULL, "MyThread");
   	if (!IS_ERR(task))
   	{
   		printk("[program2] : module_init kthread start\n");
   		wake_up_process(task);
   	}
   	return 0;
   }
   ```

4. In `my_fork()` function, using `kernel_clone` to fork a process. Wait until child process terminates, do corresponding operations according to the return status.

   ```c
   /* fork a process using kernel_clone or kernel_thread */
   struct kernel_clone_args args = {
   	.flags = ((SIGCHLD | CLONE_VM | CLONE_UNTRACED) & ~CSIGNAL),
   	.pidfd = NULL,
   	.parent_tid = NULL,
   	.child_tid = NULL,
   	.exit_signal = (SIGCHLD & CSIGNAL),
   	.stack = (unsigned long)&my_exec,
   	.stack_size = 0,
   	.tls = 0,
   };
   pid = kernel_clone(&args);
   
   /* wait until child process terminates */
   status = my_wait(pid);
   
   int sign = status & 0x7f; // WTERMSIG
   int stop = status & 0xff; // WIFSTOPPED
   if (sign == 0){...}	// normal termination
   else if (stop == 0x7f) {...} // WIFSTOPPED, child process stopped
   
   /* WIFSIGNALED, child process terminated, 
   display different information according to the signal */
   else if (((signed char)((sign + 1) >> 1)) > 0) {...}
   else{...} // continued
   do_exit(0);
   
   // the detail displaying information can be seen in program2.c
   ```

#### Execute

First of all, we should update the Linux source code.

1. Find the kernel files named "fork.c", add `EXPORT_SYMBOL(kernel_clone)` behind the function `kernel_clone`.

2. Find the kernel files named "namei.c", add `EXPORT_SYMBOL(getname_kernel)` behind the function `getname_kernel`.

3. Find the kernel files named "exit.c", add `EXPORT_SYMBOL(do_wait)` behind the function `do_wait`.

4. Find the kernel files named "exec.c", add `EXPORT_SYMBOL(do_execve)` behind the function `do_execve`.

5. Rebuild the module using the following codes:

   ```bash
   sudo su
   cd /home/seed/work/KERNEL_FILE
   make -j$(nproc)
   make modules_install
   make install
   reboot
   ```

After reboot, go back the directory of program2, type the following to execute:

```shell
sudo su	# switch to the root user
gcc -o test test.c	# compile to test program
make	# compile the program2 program
insmod program2.ko	# insert the program2 module
rmmod program2.ko	# remove the program2 module
dmesg	# display the output message
make clean	# clean
```

#### Output

Using `dmesg` can see the result.

test.c

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008090104228.png" alt="image-20221008090104228"  />

normal

![image-20221008090550459](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008090550459.png)

stop

![image-20221008090708438](C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 1\Answer\res\image-20221008090708438.png)

#### Learning

In this program, I learnt how to update and recompile the linux kernel. Also, I knew how to modify the kernel file for using these function in  other programs. Meanwhild, I learnt how to create a new process by using kernel_clone, and have a basic understanding of the working principle of do_wait and do_execve. In this program, I also looked up a lot of information, so it may aslo improve my Google ability.
