from .uav20l import UAV20Dataset
from .uavdt import UAVDTDataset
from .v4r import V4RDataset


datapath = {
            'UAV123_20L':'/UAV123_20L',
            'UAVDT':'/UAVDT',
            'UAVTrack112':'/UAVTrack112',
            }

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):


        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAV123_20L' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'UAVDT' in name:
            dataset = UAVDTDataset(**kwargs)
        elif 'UAVTrack112' in name:
            dataset = V4RDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset
