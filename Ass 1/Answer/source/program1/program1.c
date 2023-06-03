#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[])
{
	/* fork a child process */
	pid_t pid;
	int status;

	printf("Process start to fork\n");
	pid = fork();

	if (pid == -1) // the creation of a child process was unsuccessful
	{
		perror("fork");
		exit(1);
	} else {
		if (pid == 0) // child process
		{
			printf("I'm the Child Process, my pid = %d\n",
			       getpid());

			// get the arguments to execute
			int i;
			char *arg[argc];
			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;

			/* execute test program */
			printf("Child process start to execute test program:\n");
			execve(arg[0], arg, NULL);

			// check if the child process is replaced by new program
			printf("Continue to run the original process.\n");
			perror("execve");
			exit(EXIT_FAILURE);
		} else // parent process
		{
			printf("I'm the Parent Process, my pid = %d\n",
			       getpid());

			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receiving the SIGCHLD signal\n");

			/* check child process'  termination status */
			if (WIFEXITED(status)) {
				printf("Normal termination with EXIT STATUS = %d\n",
				       WEXITSTATUS(status));
			} else if (WIFSIGNALED(
					   status)) // child execution terminated
			{
				int sign = WTERMSIG(status);

				switch (sign) {
				case 6:
					printf("child process get SIGABRT signal\n");
					printf("child process is terminated by a abort call\n");
					break;
				case 14:
					printf("child process get SIGALRM signal\n");
					printf("child process is terminated by a alarm call\n");
					break;
				case 7:
					printf("child process get SIGBUS signal\n");
					printf("child process is terminated by a bus call\n");
					break;
				case 8:
					printf("child process get SIGFPE signal\n");
					printf("child process is terminated by a floating call\n");
					break;
				case 1:
					printf("child process get SIGHUP signal\n");
					printf("child process is terminated by a hangup call\n");
					break;
				case 4:
					printf("child process get SIGILL signal\n");
					printf("child process is terminated by a illegal_instr call\n");
					break;
				case 2:
					printf("child process get SIGINT signal\n");
					printf("child process is terminated by a interrupt call\n");
					break;
				case 9:
					printf("child process get SIGKILL signal\n");
					printf("child process is terminated by a kill call\n");
					break;
				case 13:
					printf("child process get SIGPIPE signal\n");
					printf("child process is terminated by a pipe call\n");
					break;
				case 3:
					printf("child process get SIGQUIT signal\n");
					printf("child process is terminated by a quit call\n");
					break;
				case 11:
					printf("child process get SIGSEGV signal\n");
					printf("child process is terminated by a segment_fault call\n");
					break;
				case 15:
					printf("child process get SIGTERM signal\n");
					printf("child process is terminated by a terminate call\n");
					break;
				case 5:
					printf("child process get SIGTRAP signal\n");
					printf("child process is terminated by a trap call\n");
					break;
				default:
					printf("child process get a signal which is not supported\n");
					break;
				}
				printf("CHILD EXECUTION FAILED\n");
			} else if (WIFSTOPPED(status)) // child process stopped
			{
				int stop_sign = WSTOPSIG(status);
				if (stop_sign == SIGSTOP) {
					printf("child process get SIGSTOP signal\n");
				} else {
					printf("child process raised a stop signal which is not supported\n");
				}
				printf("CHILD PROCESS STOPPED\n");
			} else // continued
			{
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);
		}
	}
}