_base_ = './faster-rcnn_r50_fpn_amp-1x_pragas.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=2,  # your two classes
        )
    )
)

data_root = 'data/pragas'
metainfo = {
    'classes': ('Chysodeixis', 'Spodoptera'),
    'palette': [
        (220, 20, 60), 
        (0, 0, 142),
    ]
}

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + '/annotations/instances_val.json')
test_evaluator = dict(ann_file=data_root + '/annotations/instances_test.json')

load_from = "https://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth"

vis_backends = []
visualizer = dict()


default_hooks = dict(checkpoint=dict(type='CheckpointHook', 
                                     interval=-1, 
                                     by_epoch=False, 
                                     save_best='auto',
                                     save_last=True
                                    )
                    )