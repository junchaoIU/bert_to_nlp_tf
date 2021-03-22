import time
# 功能：装饰器 之 计数器
def timer(func):
    '''
        功能：装饰器 之 计数器
        操作：在 需要计算的函数 前面 加上 @timer
    '''
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print('函数：{}() 共耗时约 {:.5f} 秒'.format(func.__name__,time.time() - start))
        return res
    return wrapper