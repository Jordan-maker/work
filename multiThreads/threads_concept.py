import threading
import time


nThreads = 10
my_list = []


def worker(inicio, fin):
    for i in range(inicio, fin):
        my_list.append(i)
        time.sleep(0.01)
    return my_list

p = 100//nThreads

inicios = [i*p for i in range(nThreads)]
fines = [(i+1)*p for i in range(nThreads)]


t0 = time.time()
threads = []

for i in range(len(inicios)):
    t = threading.Thread(target=worker, args=(inicios[i], fines[i]))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

tf = time.time()

print(f"total time {tf-t0}. nThreads is {nThreads}")

#if __name__ == '__main__':

    executor = ThreadPoolExecutor(max_workers=5)
    executor.submit(super_task, 1, 2)
    executor.submit(super_task, 3, 4)
    executor.submit(super_task, 5, 6)
    #executor.submit(super_task, 7, 8)
    #executor.submit(super_task, 9, 10)

    super_task(1 ,2)
    super_task(3, 4)
    super_task(5, 6)
    #super_task(7, 8)
    #super_task(9, 10)


