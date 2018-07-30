from multiprocessing import Process, Queue

# 프로세스 간 통신을 위해 큐(Queue)와 파이프(Pipe) 두가지가 있음
# 0. Queue : 먼저들어간 놈이 먼저나옴, (First In First Out : FIFO)
# 1. Queue : 어떤 객체에 접근하는 프로세스를 한 번에 하나로 제한할 수 있음(process-safe)

def save(q, n):
    # 세번의 q.put(save)가 일어남
    q.put('{0}회째의 Hello World'.format(n))

def main():
    que = Queue()
    for i in range(3):
        p = Process(target=save, args=(que, i))
        p.start()

    # 저장된 큐를 입력순(FIFO 방식)으로 빼내옴
    print(que.get())
    print(que.get())
    print(que.get())
    p.join()

if __name__ == "__main__":
    main()
