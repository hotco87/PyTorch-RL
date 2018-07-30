from multiprocessing import Process, Lock, Value

val = Value('i', 0)  # integer val = 0
lock = Lock()

def adder(num, val):
    while 1:
        if val.value >= 10000:
            print("process number {} closing and val is {}".format(num, val.value))
            break
        lock.acquire()
        val.value += 1
        lock.release()
        print("process number {} on progress and var is {}".format(num, val.value))

# def printers(num, val):
#     if val.value >= 10000:
#         print("process number {} closing and val is {}".format(num, val.value))
#     lock.acquire()
#     val.value += 1
#     lock.release()
#     print("process number {} on progress and var is {}".format(num, val.value))


procs = [Process(target=adder, args=(i, val)) for i in range(10)]

for p in procs:
    p.start()

for p in procs:
   p.join()
