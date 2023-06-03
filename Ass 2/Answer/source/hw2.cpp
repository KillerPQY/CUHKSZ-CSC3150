#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50

const int LOGLENGTH = 15; // length of logs
const int FPS = 100000;	  // control the refresh speed, 0.1 second

// mutex
pthread_mutex_t mutexStatus; // protecting access to game status
pthread_mutex_t mutexFrog;	 // protecting access to frog node
pthread_mutex_t mutexMap;	 // protecting access to map

struct Node
{
	int x, y;
	Node(int _x, int _y) : x(_x), y(_y){};
	Node(){};
} frog;

// game status: 0-playing, 1-win, 2-lose, 3-exit
int STATUS = 0;					  // record game status
int *LOGSTART = new int[ROW - 1]; // record the starting postion of logs

char map[ROW + 10][COLUMN];

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void)
{
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if (ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

// refresh and display the map
void refreshMap()
{
	// get x, y of frog
	pthread_mutex_lock(&mutexFrog);
	int x = frog.x;
	int y = frog.y;
	pthread_mutex_unlock(&mutexFrog);

	// update the first and the last row of the map
	pthread_mutex_lock(&mutexMap);
	for (int j = 0; j < COLUMN - 1; ++j)
		map[ROW][j] = map[0][j] = '|';
	for (int j = 0; j < COLUMN - 1; ++j)
		map[0][j] = map[0][j] = '|';
	// update the frog
	map[x][y] = '0';

	printf("\033[H\033[2J"); // clean the terminal
	// display
	for (int i = 0; i < ROW + 1; ++i)
	{
		puts(map[i]);
	}
	pthread_mutex_unlock(&mutexMap);
}

// judge status whether the game can continue
bool judgeStatus()
{
	pthread_mutex_lock(&mutexStatus);
	int f = STATUS;
	pthread_mutex_unlock(&mutexStatus);
	return (f == 0);
}

// refresh the game status after the frog moved
void refreshStatus()
{
	pthread_mutex_lock(&mutexFrog);
	int x = frog.x;
	int y = frog.y;
	pthread_mutex_unlock(&mutexFrog);

	// whether forg is not on the log
	pthread_mutex_lock(&mutexMap);
	bool flag = (map[x][y] == ' ');
	pthread_mutex_unlock(&mutexMap);

	// frog out of bound or does not on the log, lose
	if (x < 0 || x > ROW || y < 0 || y > COLUMN - 2 || flag)
	{
		pthread_mutex_lock(&mutexStatus);
		STATUS = 2;
		pthread_mutex_unlock(&mutexStatus);
	}

	// frog successfully arrived at the other side, win
	if (x == 0)
	{
		pthread_mutex_lock(&mutexStatus);
		STATUS = 1;
		pthread_mutex_unlock(&mutexStatus);
	}

	// frog is on a log, continue, status no change
}

// refresh the log in the map
void refreshLog(long logId, int startIndex, int x, int y)
{
	int endIndex = (startIndex + LOGLENGTH) % (COLUMN - 1);

	// update the map
	pthread_mutex_lock(&mutexMap);
	// update the frog, up: -1,0; down: 1,0; left: 0,-1; right: 0,1
	if (x == logId + 1)
	{
		if (y >= startIndex && y < endIndex)
		{
			if (logId % 2)
			{
				pthread_mutex_lock(&mutexFrog);
				frog.y += 1;
				pthread_mutex_unlock(&mutexFrog);
			}
			else
			{
				pthread_mutex_lock(&mutexFrog);
				frog.y += -1;
				pthread_mutex_unlock(&mutexFrog);
			}
		}
	}
	// update the log
	if (startIndex <= endIndex)
	{
		for (int i = 0; i < startIndex; ++i)
		{
			map[logId + 1][i] = ' ';
		}
		for (int i = startIndex; i < endIndex; ++i)
		{
			map[logId + 1][i] = '=';
		}
		for (int i = endIndex; i < COLUMN - 1; ++i)
		{
			map[logId + 1][i] = ' ';
		}
	}
	else
	{
		for (int i = 0; i < endIndex; ++i)
		{
			map[logId + 1][i] = '=';
		}
		for (int i = endIndex; i < startIndex; ++i)
		{
			map[logId + 1][i] = ' ';
		}
		for (int i = startIndex; i < COLUMN - 1; ++i)
		{
			map[logId + 1][i] = '=';
		}
	}
	pthread_mutex_unlock(&mutexMap);
}

// thread for changing the map
void *map_change(void *t)
{
	while (judgeStatus())
	{
		// update
		refreshMap();
		usleep(FPS);
	}

	pthread_exit(NULL); // exit thread
}

// thread for moving the frog
void *frog_move(void *t)
{
	while (judgeStatus())
	{
		if (kbhit())
		{
			char move;
			std::cin >> move;
			// up: -1,0; down: 1,0; left: 0,-1; right: 0,1
			switch (move)
			{
			case 'w':
			case 'W':
				pthread_mutex_lock(&mutexFrog);
				frog.x += -1;
				pthread_mutex_unlock(&mutexFrog);
				break;

			case 's':
			case 'S':
				pthread_mutex_lock(&mutexFrog);
				frog.x += 1;
				pthread_mutex_unlock(&mutexFrog);
				break;
			case 'a':
			case 'A':
				pthread_mutex_lock(&mutexFrog);
				frog.y += -1;
				pthread_mutex_unlock(&mutexFrog);
				break;
			case 'd':
			case 'D':
				pthread_mutex_lock(&mutexFrog);
				frog.y += 1;
				pthread_mutex_unlock(&mutexFrog);
				break;
			case 'q':
			case 'Q':
				pthread_mutex_lock(&mutexStatus);
				STATUS = 3;
				pthread_mutex_unlock(&mutexStatus);
				break;
			default:
				break;
			}
		}
		// refresh game status after the frog moved
		refreshStatus();
	}

	pthread_exit(NULL); // exit thread
}

// thread for moving the log
void *logs_move(void *t)
{
	long logId = (long)t;
	int startIndex = LOGSTART[logId];

	while (judgeStatus())
	{
		// log move right
		if (logId % 2)
		{
			startIndex = (startIndex + 1 + COLUMN - 1) % (COLUMN - 1);
		}
		// log move left
		else
		{
			startIndex = (startIndex - 1 + COLUMN - 1) % (COLUMN - 1);
		}

		// frog move
		pthread_mutex_lock(&mutexFrog);
		int x = frog.x;
		int y = frog.y;
		pthread_mutex_unlock(&mutexFrog);
		// update
		refreshLog(logId, startIndex, x, y);
		usleep(FPS);
	}
	pthread_exit(NULL); // exit thread
}

int main(int argc, char *argv[])
{
	// Initialize the river map and frog's starting position
	memset(map, 0, sizeof(map));
	int i, j;
	for (i = 1; i < ROW; ++i)
	{
		for (j = 0; j < COLUMN - 1; ++j)
			map[i][j] = ' ';
	}
	// initialize the frog
	frog = Node(ROW, (COLUMN - 1) / 2);

	// initialize the game status, already initialize
	// STATUS = 0;

	// initialize the startIndex of logs
	srand(time(NULL));
	for (int i = 0; i < ROW - 1; ++i)
	{
		LOGSTART[i] = rand() % (COLUMN - 1);
		int startIndex = LOGSTART[i];
		for (int j = 0; j < LOGLENGTH; ++j)
		{
			map[i + 1][(startIndex + j + (COLUMN - 1)) % (COLUMN - 1)] = '=';
		}
	}

	// initialize mutex
	pthread_mutex_init(&mutexStatus, NULL);
	pthread_mutex_init(&mutexFrog, NULL);
	pthread_mutex_init(&mutexMap, NULL);

	// display the map into screen first
	refreshMap();

	/*  Create pthreads for wood move and frog control.  */
	// logs, frog, status
	pthread_t threads[ROW + 1];

	// log thread, each log corresponding to each thread
	long logId;
	for (logId = 0; logId < ROW - 1; ++logId)
	{
		int ret;
		ret = pthread_create(&threads[logId], NULL, logs_move, (void *)logId);
		if (ret != 0)
		{
			printf("ERROR: return code from log pthread_create() is %d", ret);
			exit(1);
		}
	}

	// frog thread
	int ret;
	ret = pthread_create(&threads[ROW - 1], NULL, frog_move, NULL);
	if (ret != 0)
	{
		printf("ERROR: return code from frog pthread_create() is %d", ret);
		exit(1);
	}

	// status, map change thread
	ret = pthread_create(&threads[ROW], NULL, map_change, NULL);
	if (ret != 0)
	{
		printf("ERROR: return code from status pthread_create() is %d", ret);
		exit(1);
	}

	// pthread_join for sychronizing the program; logs, frog and status.
	for (int i = 0; i <= ROW; ++i)
	{
		ret = pthread_join(threads[i], NULL);
		if (ret != 0)
		{
			printf("ERROR: return code from status pthread_join() is %d", ret);
			exit(1);
		}
	}

	/*  Display the output for user: win, lose or quit.  */
	printf("\033[?25h\033[H\033[2J"); // clean the terminal
	switch (STATUS)
	{
	case 1:
		printf("You win the game!!\n");
		break;
	case 2:
		printf("You lose the game!!\n");
		break;
	case 3:
		printf("You exit the game.\n");
		break;
	default:
		break;
	}

	// free mutex
	pthread_mutex_destroy(&mutexMap);
	pthread_mutex_destroy(&mutexFrog);
	pthread_mutex_destroy(&mutexStatus);

	// delete LOGSATART
	delete[] LOGSTART;

	// exit the main thread
	pthread_exit(NULL);

	return 0;
}
