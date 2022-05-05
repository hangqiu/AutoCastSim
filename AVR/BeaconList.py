import threading


class BeaconList(object):
    """
    A centralized beacon list/cache at RSU
    Can also be used as distributed local cache
    """
    def __init__(self):
        ### for ensuring beacon delivery
        self.BeaconListLock = threading.Lock()
        self.CachedBeaconList = []
        self.BeaconList = []

    def append_beacon_list(self, b):
        self.BeaconListLock.acquire()
        self.BeaconList.append(b)
        self.BeaconListLock.release()

    def get_beacon_list(self):
        self.BeaconListLock.acquire()
        ret = self.CachedBeaconList
        self.BeaconListLock.release()
        return ret

    def clear_beacon_list(self):
        self.BeaconListLock.acquire()
        del self.BeaconList
        self.BeaconList = []
        self.BeaconListLock.release()

    def tick(self):
        self.BeaconListLock.acquire()
        del self.CachedBeaconList
        self.CachedBeaconList = self.BeaconList
        self.BeaconList = []
        self.BeaconListLock.release()
