# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import multiprocessing as mp
import threading as td
import time
from multiprocessing import Process, Queue
import os

def worker(start, end, queue):
    print("start:", start)
    print("end:", end)
    print("프로세스 ID: {0} (부모 프로세스 ID: {1})".format(os.getpid(), os.getppid()))
    sum = 0
    for i in range(start,end):
        sum += i
    queue.put(sum)
    return

if __name__ == '__main__':
    START = 0
    END = 2000
    queue = Queue()
    pr1 = Process(target=worker, args=(START, int(END/2) , queue))
    pr2 = Process(target=worker, args=(int(END/2), END, queue))

    # 쓰레드가 지 맘대로 동작
    pr1.start()
    pr2.start()
    pr1.join()
    pr2.join()

    # 쓰레드가 순서대로 동작
    # pr1.start()
    # pr1.join()
    # pr2.start()
    # pr2.join()

    queue.put('STOP')
    sum = 0
    while 1:
        tmp = queue.get()
        if tmp == 'STOP' : break
        else: sum += tmp
    print("Result: ", sum)
