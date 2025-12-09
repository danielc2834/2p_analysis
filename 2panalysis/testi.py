# import pickle
# with open("C:/phd/02_twophoton/251023_tdc2_cschr_pan/3_DATA/_Chirp_OFF_LOP_Y.pkl", 'rb') as fp:
#                 dict_con = pickle.load(fp)
# indent = 0
# def dict_structure_to_text(dict_con, indent):
#     result = []
#     spaces = "  " * indent
#     tseries_counter, roi_counter = 0,0
#     for tseries, value in dict_con.items():
#         tseries_counter=+1
#         for roi in dict_con[tseries]['final_rois']:
#             for key in roi.__dict__.keys():
#                  value = roi.key
#                  if istance(value, dict):
#                       continue
#                  else:
#                     tem_type = type(value[0]).__name__
#                     result.append(f"{spaces}{key}: [list of {item_type}]")
#         #     roi_counter=+1
#         #     roi.experiment_info
#         #     roi.imaging_info
#         #     roi.mask
#         #     roi.unique_id
#         #     roi.category
#         #     roi.source_image
#         #     roi.raw_trace
#         #     roi.baseline_method
#         #     roi.df_trace
#         # print(roi)


#         # for key, value in d.items():
#         #     if isinstance(value, dict):
#         #         if key != 'videos' or key != 'condition_metrics':
#         #             video_counter += 1
#         #         if key == 'condition_metrics':
#         #             video_counter = 0
#         #         if video_counter < 2:
#         #             result.append(f"{spaces}{key}/ (dict)")
#         #             result.append(self.dict_structure_to_text(value, indent + 1))
#         #     elif isinstance(value, list):
#         #         if value:  # if list has items, show type of first item
#         #             item_type = type(value[0]).__name__
#         #             result.append(f"{spaces}{key}: [list of {item_type}]")
#         #         else:
#         #             result.append(f"{spaces}{key}: [empty list]")
#         #     else:
#         #         result.append(f"{spaces}{key}: {type(value).__name__}")
        
#         # return "\n".join(result)


# #             result.append(f"{spaces}{tseries}/ (dict)")
# #             result.append(dict_structure_to_text(value, indent + 1))


# #         elif isinstance(value, list):
# #             if value:  # if list has items, show type of first item
# #                 item_type = type(value[0]).__name__
# #                 result.append(f"{spaces}{key}: [list of {item_type}]")
# #             else:
# #                 result.append(f"{spaces}{key}: [empty list]")
# #         else:
# #             result.append(f"{spaces}{key}: {type(value).__name__}")
# #     return result
# resilts=dict_structure_to_text(dict_con, 0)
# print(resilts)



import numpy as np
import matplotlib.pyplot as plt
value = [] 
duraiton =  []

v1 = [0,0,1,0,0.5]
d1=[4,2,3,3,2]

v2 = [0.5,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
d2 = [0.243,0.449,0.406,0.372,0.343,0.317,0.296,0.277,0.26,0.246,0.232,0.221,0.21,0.2,0.192,0.183,0.176,0.169,0.163,0.157,0.151,0.146,0.142,0.137,0.133,0.129,0.125,0.122,0.118,0.115,0.112,0.11,0.107,0.104,0.102,0.099,0.097,0.095,0.093,0.091,0.09,0.087,0.086,0.084,0.082,0.081,0.079,0.078,0.077,0.075,0.074,0.073,0.072,0.07,0.069,0.068,0.068,0.066,0.065,0.064,0.063,0.062,0.062,0.06,0.06,0.059,0.058,0.058,0.056,0.056,0.055,0.055,0.054,0.053,0.052,0.052,0.052,0.05]

v3 = [0.5,0.53125,0.4375,0.59375,0.375,0.65625,0.3125,0.71875,0.25,0.78125,0.1875,0.84375,0.125,0.90625,0.0625,0.96875,0]
d3 = [2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

v4 = [0.5,0.6,0.7,0.8,0.9,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,0.1,0.2,0.3,0.4,0.5,0]
d4 = [2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 2]
# print(len(value))
# print(len(duraiton))
# protocoll=[]
# fps=mean_fps
# for val, dur in zip(value, duraiton):
#     frames= fps*dur
#     list =[val]*round(frames)
#     # values = np.array([])
#     protocoll.extend(list)
# time=np.arange(0,len(protocoll), 1)
# plt.plot(time, protocoll)
# plt.show()
a=1