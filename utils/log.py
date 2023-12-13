
def parse_log_file(path):
    log_file = open(path)
    
    data_list = []
    
    total_step = 0
    curr_step = -1
    while True:
        line = log_file.readline()
        if not line:
            break
        tokens = line.split()
        
        if "Learning iteration" in line:
            step_str = tokens[3]
            curr_step, total_step = step_str.split("/")
            data = {}
        elif "Computation" in line:
            data["Computation"] = float(tokens[1])
            data["Collection time"] = float(tokens[4][:-2])
            data["Learning time"] = float(tokens[6][:-2])
        elif "Value function loss" in line:
            data["Value function loss"] = float(tokens[3])
        elif "Surrogate loss" in line:
            data["Surrogate loss"] = float(tokens[2])
        elif "Mean action noise std" in line:
            data["Mean action std"] = float(tokens[4])
        elif "Mean reward" in line:
            data["Mean reward"] = float(tokens[2])
        elif "Mean episode length" in line:
            data["Mean episode length"] = float(tokens[3])
        elif "Total timesteps" in line:
            data["Total timesteps"] = float(tokens[2])
        elif "Iteration time" in line:
            data["Iteration time"] = float(tokens[2][:-1])
        elif "Total time" in line:
            data["Total time"] = float(tokens[2][:-1])
        elif "ETA" in line:
            data["ETA"] = float(tokens[1][:-1])
            
            data["Mean step reward"] = data["Mean reward"] / data["Mean episode length"]
            data_list.append(data)
            
    return data_list
        
        
