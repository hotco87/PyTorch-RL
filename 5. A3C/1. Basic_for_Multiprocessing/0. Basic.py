# https://kimdoky.github.io/python/2017/11/27/library-book-chap13-1.html

from multiprocessing import Process
import os

# python에서는 기본적으로 병렬처리 라이브러리(multiprocessing)를 지원하고 있음.
# multiprocessing 은 프로세스의 생성과 병렬처리를 지원
# Process 객체를 만들어 start() 메서드를 호출하면 간단하게 자식 프로세스를 생성할 수 있음.
# Process 객체를 생성할 때,

# 1. Process 객체 생성
# 2. 자식 Process 를 생성하는 대상(target)과 인수(args)를 지정
# 3. start() : 자식 프로세서 생성
# 4. join() : 자식 Process 종료를 기다림.
# join() 메서드를 생략하면 병렬처리가 완료되기 전 다음 처리가 시작될 수 있음.


def f(x):
    print("{0} - 프로세스 ID: {1} (부모 프로세스 ID: {2})".format(x, os.getpid(), os.getppid()))

def main():
    for i in range(3): # 3개의 자식 프로세스 생성
        p = Process(target=f, args=(i, )) # p라는 프로세스 객체 생성
        p.start() # 자식프로세스 생성
    p.join()

if __name__ == "__main__":
    main()