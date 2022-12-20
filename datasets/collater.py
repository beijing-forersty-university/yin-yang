class Collater():

    def __call__(self, data):
        '''在这里重写collate_fn函数'''
        return tuple(zip(*data))