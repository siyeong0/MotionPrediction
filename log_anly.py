import matplotlib.pyplot as plt
import numpy as np

from utils.log import parse_log_file

if __name__ == "__main__":
    data = parse_log_file("logs/r0/log.txt")
    
    x_label = "Total timesteps"
    y_label = "Mean step reward"
    x_list = []
    y_list = []
    for d in data:
        x_list.append(d[x_label])
        y_list.append(d[y_label])
            
    plt.plot(np.array(x_list), np.array(y_list))
    plt.show()