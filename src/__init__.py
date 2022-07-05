import torch
if torch.cuda.is_available():
    print('GPU is available on this device')
    print(str(torch.cuda.device_count())+' GPUs are available')
    device = torch.device("cuda:0")
else:
    print("GPU isn't available on this device")
    device = torch.device("cpu")
    
    

