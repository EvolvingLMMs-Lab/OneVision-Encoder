# # import os

# # from .properties import Property
# # from dataset.registry import DATASET_REGISTRY

# # rank = int(os.getenv("RANK", "0"))
# # local_rank = int(os.getenv("LOCAL_RANK", "0"))
# # world_size = int(os.getenv("WORLD_SIZE", "1"))


# # # 建议使用的 Dataloader 为 dataloader/data_v2_parallel_rec.py

# # hostname = os.environ["HOSTNAME"]


# # def init_list_prefix():
# #     if hostname in [
# #             # 前8台服务器的每台机器都是不同的分片
# #             "VM-2-20-tencentos",
# #             "VM-2-39-tencentos",
# #             "VM-2-85-tencentos"
# #         ]:
# #         list_prefix = [
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin13_994x994",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin00_1008x1008",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin01_1008x672",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin02_1008x560",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin03_1008x756",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin06_798x532",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin07_798x798",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin09_672x672",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin04_588x588",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin05_490x490",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin08_630x476",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin10_630x350",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin11_294x294",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin12_798x588",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin14_392x392",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin15_588x392",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin16_476x350",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin17_448x252",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin18_490x322",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin19_392x294",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin20_448x336",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin21_238x238",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin22_308x238",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin23_448x448",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin24_476x476",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin25_294x224",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin26_308x308",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin27_518x518",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin28_224x224",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin29_294x196",
# #             "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin30_196x196",
# #             "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin31_210x210",
# #         ]
# #     # elif hostname == "VM-2-85-tencentos": # must be 32
# #     #     list_prefix = [
# #     #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin15_588x392",
# #     #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin24_476x476",
# #     #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin26_308x308",
# #     #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin30_196x196",
# #     #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin10_630x350",
# #     #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin25_294x224",
# #     #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin10_630x350",
# #     #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin25_294x224",
# #     #     ]

# #     list_prefix = list_prefix[local_rank::8]
# #     return list_prefix



# # @DATASET_REGISTRY.register()
# # def LAION225_COYO400_IN14_HOI31():
# #     list_prefix = init_list_prefix()    

# #     # 这里的8是根据实际的GPU数量进行分片
# #     # 每张卡8个GPU，分片数为8

# #     if os.environ["HOSTNAME"] in \
# #         [
# #             # 前8台服务器的每台机器都是不同的分片
# #             "VM-2-20-tencentos",
# #             "VM-2-21-tencentos",
# #             "VM-2-23-tencentos",
# #             "VM-2-28-tencentos",
# #             "VM-2-34-tencentos",
# #             "VM-2-58-tencentos",
# #             "VM-2-62-tencentos",
# #             "VM-2-85-tencentos",

# #             # VM-2-39-tencentos 这台机器是 VM-2-20-tencentos 这台机器的复制
# #             "VM-2-39-tencentos",
# #         ]:

# #         _LAION225_COYO400_IN14_HOI31 = Property(
# #             num_class=2000000,
# #             num_example=0,
# #             prefix=list_prefix,
# #             name="LAION225_COYO400_IN14_HOI31",
# #             label_select=0,
# #             label_start=0,
# #             num_shards=1,
# #             shard_id=0,
# #             dali_type="parallel_rec",
# #             random_diff=10,
# #         )
# #     else:
# #         # 非指定主机设置为None (Set to None for non-specified hosts)
# #         _LAION225_COYO400_IN14_HOI31 = None
# #     return _LAION225_COYO400_IN14_HOI31
# import os

# from .properties import Property
# from dataset.registry import DATASET_REGISTRY

# rank = int(os.getenv("RANK", "0"))
# local_rank = int(os.getenv("LOCAL_RANK", "0"))
# world_size = int(os.getenv("WORLD_SIZE", "1"))


# # 建议使用的 Dataloader 为 dataloader/data_v2_parallel_rec.py

# hostname = os.environ["HOSTNAME"]

# # if hostname in [
# #         # 前8台服务器的每台机器都是不同的分片
# #         "VM-2-20-tencentos",
# #         "VM-2-21-tencentos",
# #         "VM-2-23-tencentos",
# #         "VM-2-28-tencentos",
# #         "VM-2-34-tencentos",
# #         "VM-2-58-tencentos",
# #         "VM-2-62-tencentos",
# #         # "VM-2-85-tencentos" fix, this is bug

# #         # VM-2-39-tencentos 这台机器是 VM-2-20-tencentos 这台机器的复制
# #         "VM-2-39-tencentos",
# #         "VM-2-79-tencentos",
# #     ]:
#     # list_prefix = [
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin13_994x994",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin00_1008x1008",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin01_1008x672",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin02_1008x560",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin03_1008x756",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin06_798x532",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin07_798x798",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin09_672x672",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin04_588x588",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin05_490x490",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin08_630x476",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin10_630x350",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin11_294x294",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin12_798x588",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin14_392x392",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin15_588x392",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin16_476x350",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin17_448x252",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin18_490x322",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin19_392x294",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin20_448x336",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin21_238x238",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin22_308x238",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin23_448x448",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin24_476x476",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin25_294x224",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin26_308x308",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin27_518x518",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin28_224x224",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin29_294x196",
#     # "/data_3/training_mlcd_laion_coyo_hoi/multi_rec_bin30_196x196",
#     # "/data_4/training_mlcd_laion_coyo_hoi/multi_rec_bin31_210x210"
#     # ]
#     #  /vlm/data/coyo400m_resized448
# list_prefix = [
#     "/vlm/data/coyo400m_resized448/coyo700m_00_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_01_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_02_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_03_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_04_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_05_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_06_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_07_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_08_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_09_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_10_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_11_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_12_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_13_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_14_resize448_resize448",
#     "/vlm/data/coyo400m_resized448/coyo700m_15_resize448_resize448",
# ]
#     # "/vlm/data/coyo400m_resized448/coyo700m_16_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_17_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_18_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_19_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_20_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_21_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_22_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_24_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_23_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_25_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_26_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_27_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_28_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_29_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_30_resize448_resize448",
#     # "/vlm/data/coyo400m_resized448/coyo700m_31_resize448_resize448",
# # elif hostname == "VM-2-85-tencentos": # must be 32
# #     list_prefix = [
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin13_994x994",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin00_1008x1008",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin01_1008x672",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin02_1008x560",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin03_1008x756",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin06_798x532",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin07_798x798",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin09_672x672",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin04_588x588",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin05_490x490",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin08_630x476",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin10_630x350",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin11_294x294",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin12_798x588",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin14_392x392",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin15_588x392",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin16_476x350",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin17_448x252",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin18_490x322",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin19_392x294",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin20_448x336",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin21_238x238",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin22_308x238",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin23_448x448",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin24_476x476",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin25_294x224",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin26_308x308",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin27_518x518",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin28_224x224",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin29_294x196",
# #         "/data_0/training_mlcd_laion_coyo_hoi/multi_rec_bin30_196x196",
# #         "/data_1/training_mlcd_laion_coyo_hoi/multi_rec_bin31_210x210",
# #     ]

# list_prefix = list_prefix[local_rank::8]
# # 这里的8是根据实际的GPU数量进行分片
# # 每张卡8个GPU，分片数为8

# # if os.environ["HOSTNAME"] in \
# #     [
# #         # 前8台服务器的每台机器都是不同的分片
# #         "VM-2-20-tencentos",
# #         "VM-2-21-tencentos",
# #         "VM-2-23-tencentos",
# #         "VM-2-28-tencentos",
# #         "VM-2-34-tencentos",
# #         "VM-2-58-tencentos",
# #         "VM-2-62-tencentos",
# #         "VM-2-85-tencentos",

# #         # VM-2-39-tencentos 这台机器是 VM-2-20-tencentos 这台机器的复制
# #         "VM-2-39-tencentos",
# #         "VM-2-79-tencentos",
# #     ]:

# _LAION225_COYO400_IN14_HOI31 = Property(
#     num_class=2000000,
#     num_example=0,
#     prefix=list_prefix,
#     name="LAION225_COYO400_IN14_HOI31",
#     label_select=0,
#     label_start=0,
#     num_shards=1,
#     shard_id=0,
#     dali_type="parallel_rec",
#     random_diff=10,
# )
# # else:
# #     # 非指定主机设置为None (Set to None for non-specified hosts)
# #     _LAION225_COYO400_IN14_HOI31 = None


# @DATASET_REGISTRY.register()
# def LAION225_COYO400_IN14_HOI31():
#     return _LAION225_COYO400_IN14_HOI31
