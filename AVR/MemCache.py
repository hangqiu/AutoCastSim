import numpy as np


class MemCache(object):
    def __init__(self, max_cache_length=5, gap_frames=2):
        self._cache_list = []
        self._max_cache_length = max_cache_length
        self._gap_frames = gap_frames

    def __len__(self):

        return int( (len(self._cache_list)+1) / self._gap_frames)

    def append(self, data):
        self._cache_list.append(data)
        while self.__len__() > self._max_cache_length:
            self._cache_list.pop(0)

        # print("cache list")
        # print(self._cache_list)

    def get_numpy_array(self):
        idx_list = [-1 - (i * self._gap_frames) for i in range(self.__len__())]
        # print("idx list")
        # print(self.__len__())
        # print(idx_list)
        return np.asarray(self._cache_list)[idx_list]

    def get_padded_numpy_array(self):
        print("in padded numpy array")
        if len(self._cache_list) == 0:
            return None

        cache_array = self.get_numpy_array()
        newshape = []
        for i in range(len(cache_array.shape)-1):
            if i == 0:
                continue

            newshape.append(cache_array.shape[i])
        newshape.append(-1)

        # print(cache_array.shape)
        cache_array_reshape = np.reshape(cache_array, newshape)
        # print(cache_array.shape)

        diff = self._max_cache_length - self.__len__()

        if diff == 0:
            return cache_array_reshape

        padding = np.zeros(cache_array[0].shape)
        # print(padding.shape)
        # newshape = (1,) + padding.shape
        # padding = np.reshape(padding, newshape)
        # print(cache_array)
        # print(padding.shape)
        for i in range(diff):
            cache_array_reshape = np.concatenate([padding, cache_array_reshape], axis=len(cache_array_reshape.shape)-1)
            # print(cache_array_reshape.shape)

        # print("cache array")
        return cache_array_reshape


