# training schedule for 1x
_base_ = [
    'configs/textrecog/maerec/_base_marec_vit_s.py',
    'configs/textrecog/_base_/datasets/mjsynth.py',
    'configs/textrecog/_base_/datasets/synthtext.py',
    'configs/textrecog/_base_/datasets/cute80.py',
    'configs/textrecog/_base_/datasets/iiit5k.py',
    'configs/textrecog/_base_/datasets/svt.py',
    'configs/textrecog/_base_/datasets/svtp.py',
    'configs/textrecog/_base_/datasets/icdar2013.py',
    'configs/textrecog/_base_/datasets/icdar2015.py',
    'configs/textrecog/_base_/default_runtime.py',
    'configs/textrecog/_base_/schedules/schedule_adamw_cos_10e.py',
]

model = dict(
    backbone=dict(
        type='VisionTransformer',
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        pretrained=''),       # change this
    decoder=dict(
        type='MAERecDecoder',
        n_layers=6,
        d_embedding=768,
        n_head=8,
        d_model=768,
        d_inner=3072,
        d_k=96,
        d_v=96))

# dataset settings
train_list = [
    _base_.synthtext_textrecog_train, _base_.mjsynth_textrecog_train
]

val_list = [
    _base_.cute80_textrecog_test, _base_.iiit5k_textrecog_test,
    _base_.svt_textrecog_test, _base_.svtp_textrecog_test,
    _base_.icdar2013_textrecog_test, _base_.icdar2015_textrecog_test
]

test_list = [
    _base_.cute80_textrecog_test, _base_.iiit5k_textrecog_test,
    _base_.svt_textrecog_test, _base_.svtp_textrecog_test,
    _base_.icdar2013_textrecog_test, _base_.icdar2015_textrecog_test
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

auto_scale_lr = dict(base_batch_size=512)

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)
val_dataset = dict(
    type='ConcatDataset', datasets=val_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=64,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

val_evaluator = dict(dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])


test_evaluator = dict(dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
