
import os
from NeutrosophicSet import *
from Visualization import *

folder_path = "./Results/"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)


h = 3
sz = 250
x = cv.imread("./samples/c_man.png", 0)
tru = Neutrosophic_set(x, h, sz).truth_mem()
ind = Neutrosophic_set(x, h, sz).indeter_mem()
fal = Neutrosophic_set(x, h, sz).false_mem()

NS_plots(x, h, sz).tru_viz(folder_path+"tru.png")
NS_plots(x, h, sz).ind_viz(folder_path+"ind.png")
NS_plots(x, h, sz).fal_viz(folder_path+"fal.png")

NS_plots(x, h ,sz).tru_intensity(folder_path+"tru_int.png")
NS_plots(x, h ,sz).ind_intensity(folder_path+"ind_int.png")
NS_plots(x, h ,sz).fal_intensity(folder_path+"fal_int.png")

NS_plots(x, h ,sz).ns_kde(folder_path+"ns_kde.png")
