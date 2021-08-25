import os
# from data.datasets.lacmus import *
from yolox.data import get_yolox_datadir

if __name__ == '__main__':
    # target_transform=AnnotationTransform()
    # print(target_transform)

    from yolox.data import (
        LacmusDetection,
        # VOCDetection,
        TrainTransform,
        YoloBatchSampler,
        DataLoader,
        InfiniteSampler,
        MosaicDetection,
        worker_init_reset_seed,
    )
    from yolox.utils import (
        wait_for_the_master,
        get_local_rank,
    )

    local_rank = get_local_rank()
    dataset = LacmusDetection(
        data_dir=os.path.join(get_yolox_datadir(), "lacmus-ods"),
        image_sets=[('2021', 'trainval')],
        img_size=(512, 512),
        preproc=TrainTransform(max_labels=50),
        cache=True,
    )


    print('Dataset length: ', len(dataset))
    print(len(dataset.ids))
    # print(dataset.annotations())
