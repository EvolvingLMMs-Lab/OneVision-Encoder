# import os

# import numpy as np

# from dataset.registry import DATASET_REGISTRY

# from .properties import Property

# # 获取分布式训练环境变量 (Get distributed training environment variables)
# rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
# local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
# world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


# list_prefix = []


# def init_prefix():
#     list_prefix = []

#     path = "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/list_videos_frames64_kinetics_ssv2_new"
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             lines = f.readlines()
#             lines = [x.strip() for x in lines]
#     list_label_path = [
#         # "/data_0/feilong/label_reshape_16_merged.npy",
#         # "/data_0/feilong/label_reshape_08_merged.npy",
#         # "/data_0/feilong/label_reshape_04_merged.npy",
#         # "/data_0/feilong/label_reshape_02_merged.npy",
#         # "/data_0/feilong/label_reshape_01_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_16_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_08_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_04_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_02_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_01_merged.npy",
#         # "/data_3/label_reshape_16_merged.npy",
#         # "/data_3/label_reshape_08_merged.npy",
#         # "/data_3/label_reshape_04_merged.npy",
#         # "/data_3/label_reshape_02_merged.npy",
#         # "/data_3/label_reshape_01_merged.npy",
#         ]

#     label_frames_16 = np.load(list_label_path[0])
#     num_label = label_frames_16.shape[0]
#     # label_frames_08 = np.load(list_label_path[1]).reshape(num_label, -1)
#     # label_frames_04 = np.load(list_label_path[2]).reshape(num_label, -1)
#     # label_frames_02 = np.load(list_label_path[3]).reshape(num_label, -1)
#     # label_frames_01 = np.load(list_label_path[4]).reshape(num_label, -1)


#     label = np.concatenate([
#         label_frames_16,
#         # label_frames_08,
#         # label_frames_04,
#         # label_frames_02,
#         # label_frames_01,
#     ], axis=1)

#     print(label.shape)

#     for i in range(label.shape[0]):
#         list_prefix.append(
#             (lines[i], label[i])
#         )
#     return list_prefix


# def init_prefix_dense():
#     list_prefix = []

#     path = "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/list_videos_frames64_kinetics_ssv2_new"
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             lines = f.readlines()
#             lines = [x.strip() for x in lines]
#     list_label_path = [
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_16_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_08_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_04_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_02_merged.npy",
#         "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/label_reshape_01_merged.npy",
#         # "/data_3/label_reshape_16_merged.npy",
#         # "/data_3/label_reshape_08_merged.npy",
#         # "/data_3/label_reshape_04_merged.npy",
#         # "/data_3/label_reshape_02_merged.npy",
#         # "/data_3/label_reshape_01_merged.npy",
#         ]

#     label_frames_16 = np.load(list_label_path[0])
#     num_label = label_frames_16.shape[0]
#     label_frames_08 = np.load(list_label_path[1]).reshape(num_label, -1)
#     label_frames_04 = np.load(list_label_path[2]).reshape(num_label, -1)
#     label_frames_02 = np.load(list_label_path[3]).reshape(num_label, -1)
#     label_frames_01 = np.load(list_label_path[4]).reshape(num_label, -1)


#     label = np.concatenate([
#         label_frames_16,
#         label_frames_08,
#         label_frames_04,
#         label_frames_02,
#         label_frames_01,
#     ], axis=1)

#     # print(label.shape)
   

#     for i in range(label.shape[0]):
#         list_prefix.append(
#             (lines[i], label[i])
#         )
#     # print(list_prefix)
#     return list_prefix


# # _mlcd_video_v5_dense = Property(
# #     num_class=[100_000] * 4,
# #     num_example=0,
# #     name="mlcd_video_v5_dense",
# #     prefix=list_prefix,
# #     num_shards=8,
# #     shard_id=local_rank,
# #     label_select=[
# #             [0, 10],
# #             [10, 30],
# #             [30, 70],
# #             [70, 150],
# #         ],
# #     frame_scales=[16, 8, 4, 2],
# #     label_start=0,
# #     dali_type="decord",
# #     random_diff=10,
# #     pfc_type="partial_fc",
# # )

# # _mlcd_video_v5_dense_frames16 = Property(
# #     num_class=[100_000],
# #     num_example=0,
# #     name="mlcd_video_v5_dense",
# #     prefix=list_prefix,
# #     num_shards=8,
# #     shard_id=local_rank,
# #     label_select=[
# #             [0, 10],
# #         ],
# #     frame_scales=[16],
# #     label_start=0,
# #     dali_type="decord",
# #     random_diff=10,
# #     pfc_type="partial_fc",
# # )


# @DATASET_REGISTRY.register()
# def unvit_video_v5():
#     list_prefix = init_prefix()
#     _unvit_video_v5 = Property(
#         num_class=100_000,
#         num_example=0,
#         name="mlcd_video_v5",
#         prefix=list_prefix,
#         num_shards=8,
#         shard_id=local_rank,
#         label_select=0,
#         label_start=0,
#         dali_type="decord",
#         random_diff=10,
#         pfc_type="partial_fc",
#     )
#     return _unvit_video_v5

# @DATASET_REGISTRY.register()
# def unvit_video_v5_dense():
#     list_prefix = init_prefix_dense()
#     _unvit_video_v5_dense = Property(
#         num_class=[100_000] * 1,
#         num_example=0,
#         name="mlcd_video_v5_dense",
#         prefix=list_prefix,
#         num_shards=8,
#         shard_id=local_rank,
#         label_select=[
#                 [0, 10],
#                 # [10, 30],
#                 # [30, 70],
#                 # [70, 150],
#             ],
#         frame_scales=[16],
#         label_start=0,
#         dali_type="decord",
#         random_diff=10,
#         pfc_type="partial_fc",
#     )
#     return _unvit_video_v5_dense

# # _mlcd_video_v5_dense = Property(
# #     num_class=[100_000] * 4,
# #     num_example=0,
# #     name="mlcd_video_v5_dense",
# #     prefix=list_prefix,
# #     num_shards=8,
# #     shard_id=local_rank,
# #     label_select=[
# #             [0, 10],
# #             [10, 30],
# #             [30, 70],
# #             [70, 150],
# #         ],
# #     frame_scales=[16, 8, 4, 2],
# #     label_start=0,
# #     dali_type="decord",
# #     random_diff=10,
# #     pfc_type="partial_fc",
# # )


# # @DATASET_REGISTRY.register()
# # def mlcd_video_v5_dense():
# #     return _mlcd_video_v5_dense


# # @DATASET_REGISTRY.register()
# # def mlcd_video_v5_dense_frames16():
# #     return _mlcd_video_v5_dense_frames16
