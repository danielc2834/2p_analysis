import os
import sys
code_path = r'D:\progresources\Calcium-imaging-analysis\general_lab_code'
sys.path.insert(0, code_path) 
import ROI_mod

home_path='E:\\PhD\\Experiments\\2p\\t4t5_freqtunning_Mi1_glucl_overexpression\\processed\\TF\\pickles'
#'E:\\PhD\\Experiments\\2p\\t4t5_freqtunning_Mi1_glucl_overexpression\\processed\\TF\\pickles'
os.chdir(home_path)


datasets_to_load=os.listdir(home_path)
#pickleDir='E:\\PhD\\Experiments\\2p\\t4t5_freqtunning_Mi1_glucl_overexpression\\processed\\TF\\pickles\\20210325_jv_fly2-TSeries-fly2-003-cycle_2.pickle'
savepath='E:\\PhD\\Experiments\\2p\\t4t5_freqtunning_Mi1_glucl_overexpression\\summary\\traces_edges'
# 'E:\\PhD\\Experiments\\2p\\t4t5_freqtunning_Mi1_glucl_overexpression\\summary\\traces_edges'
#picleDir= 'E:\\PhD\\Experiments\\2p\\t4t5_freqtunning_Mi1_glucl_overexpression\\processed\\edges\\pickles\\20210325_jv_fly2-TSeries-fly2-003-cycle_1.pickle'

for idx,dataset in enumerate(datasets_to_load):
    #first run analisys script
    ROI_mod.plot_traces_freq_epochs(idx,dataset,savepath,plot_prev=True)