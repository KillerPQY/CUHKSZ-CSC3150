## Assignment 2 Report

### Environment

version of OS: Ubuntu 16.04.7 LTS
version of kernel: 5.10.146
version of gcc: 5.4.0
version of g++: 5.4.0

### Design

In this program, I divided the whole implementation into 3 parts.

1. Create threads
   - Create a thread for **each log**, totally 9 threads.
   - Create a thread for **updating and displaying the map**.
   - Create a thread for **moving the frog**.
2. Create mutex
   - Use a mutex to protect access to the **game status**.
   - Use a mutex to protect access to the **frog node**.
   - Use a mutex to protect access to the **map**.
3. According to the game status, print corresponding information or continue.

#### Create Threads

##### Logs

For threads of logs, each log need a thread to move. Thus, I should create 9 threads:

```c++
long logId;
for (logId = 0; logId < ROW - 1; ++logId)
{
	ret = pthread_create(&threads[logId], NULL, logs_move, (void *)logId);
}
```

For logs moving, we can set the start position to control the moving direction of logs:

```c++
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
```

After moving the log, we should modify the value in the map. When the frog is on the log, the frog will move with the log:

```c
// update the log
if (startIndex <= endIndex) {
	for (int i = 0; i < startIndex; ++i) {
		map[logId + 1][i] = ' ';
	}
	for (int i = startIndex; i < endIndex; ++i) {
			map[logId + 1][i] = '=';
	}
	for (int i = endIndex; i < COLUMN - 1; ++i) {
			map[logId + 1][i] = ' ';
	}
} else {
	for (int i = 0; i < endIndex; ++i) {
			map[logId + 1][i] = '=';
	}
	for (int i = endIndex; i < startIndex; ++i) {
			map[logId + 1][i] = ' ';
	}
	for (int i = startIndex; i < COLUMN - 1; ++i) {
			map[logId + 1][i] = '=';
	}
}

// update the frog
if (x == logId + 1) {
	if (y >= startIndex && y < endIndex) {
		if (logId % 2) {
			refreshFrog(0, 1);
		}
		else {
			refreshFrog(0, -1);
        }
	}
}
```

##### Map

This thread is used for changing value in the map and display the game screen:

```c++
ret = pthread_create(&threads[ROW], NULL, map_change, NULL);
```

For displaying, use `usleep()` function to control the update rate, and use the following function to change the map and render the game screen:

```c++
void refreshMap()
```

##### Frog

This thread is used for player to control the frog:

```c++
ret = pthread_create(&threads[ROW - 1], NULL, frog_move, NULL);
```

According to the player's input char, update the attributes of frog:

```c
char move = getchar();
switch (move) {
    case w:...;
    case s:...;
    case a:...;
    case d:...;
    case q:...;
}
```

To control the frog, just modify the attributes of frog node:

```c++
// up: -1,0; down: 1,0; left: 0,-1; right: 0,1
pthread_mutex_lock(&mutexFrog);
frog.x += x;
frog.y += y;
pthread_mutex_unlock(&mutexFrog);
```

#### Create Mutex

3 mutex were used:

```c++
pthread_mutex_t mutexStatus; // protecting access to game status
pthread_mutex_t mutexFrog;	 // protecting access to frog node
pthread_mutex_t mutexMap;	 // protecting access to map
```

Each time we access status, frog, map, we should lock and unlock to protect these data:

```c++
pthread_mutex_lock(&mutex);
...;
pthread_mutex_unlock(&mutex);
```

Noted that before using mutex we should initialize them. After using mutex, we should destroy them:

```c++
pthread_mutex_init(&mutexStatus, NULL);
pthread_mutex_init(&mutexFrog, NULL);
pthread_mutex_init(&mutexMap, NULL);
...;
pthread_mutex_destroy(&mutexMap);
pthread_mutex_destroy(&mutexFrog);
pthread_mutex_destroy(&mutexStatus);
```

#### Judge Status

According to the global variable **STATUS**, we can judge the game situation and do the corresponding operations:

```c++
// 0-playing, 1-win, 2-lose, 3-exit
switch (STATUS){
    case 1:
    case 2:
    case 3:
}
```

As long as the game does not stop, all threads will keep active, thus, we need a function to judge the status of the game:

```c++
bool judgeStatus()
{
    // STATUS is shared data, need mutex to access
	pthread_mutex_lock(&mutexStatus);
	int f = STATUS;
	pthread_mutex_unlock(&mutexStatus);
	return (f == 0);
}
```

### Steps to Execute

#### Start

```bash
# In the 'source' directory
g++ hw2.cpp -lpthread	# link pthread library and compile this game
./a.out	# start this game
```

#### Operation

- **W/w**: let the frog move up.
- **S/s**: let the frog move down.
- **A/a**: let the frog move left.
- **D/d**: let the frog move right.
- **Q/q**: exit this game.

### Output

After executing this game, the game would be displayed.

<img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 2\Answer\res\image-20221016154830190.png" alt="image-20221016154830190"  />

When players finish the game, the game will print information according to the situation of the game.

- win

  <img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 2\Answer\res\image-20221016155041824.png" alt="image-20221016155041824"  />

- lose

  <img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 2\Answer\res\image-20221016155128570.png" alt="image-20221016155128570"  />

- quit

  <img src="C:\Users\Dake\Desktop\CSC3150\CSC3150 Ass\Ass 2\Answer\res\image-20221016155211160.png" alt="image-20221016155211160"  />

### Learning

In this assignment, I learnt how to use `pthread` library for multithread programming, thus, I have  known more about threads when I was searching in the Google. Meanwhile, I learnt how important the protection of shared data source is when I use  `mutex` to lock them. In this assignment, I should write all contents only in one file, this makes  too many functions in a file, which make these functions inconvenient to use. Thus, organizing code structure is also very important to programmers.

