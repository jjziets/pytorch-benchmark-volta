"""Compare speed of different models with batch size 12"""
import torch
import torchvision.models as models
import platform
import psutil
import torch.nn as nn
import datetime
import time
import os
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
import json
import inspect 
torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware. 
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.



# Function to list all callable model constructors in a given module
def list_models_from_module(module):
    model_constructors = {}
    for model_name in module.__all__:
        model_constructor = getattr(module, model_name)
        if callable(model_constructor):
            # Check if the constructor needs additional arguments
            args = inspect.signature(model_constructor).parameters
            model_constructors[model_name] = {
                "constructor": model_constructor,
                "requires_args": len(args) > 1  # excluding 'self'
            }
    return model_constructors

# Automatically populating MODEL_LIST
MODEL_LIST = {
    'mnasnet': list_models_from_module(models.mnasnet),
    'resnet': list_models_from_module(models.resnet),
    'densenet': list_models_from_module(models.densenet),
    'squeezenet': list_models_from_module(models.squeezenet),
    'vgg': list_models_from_module(models.vgg),
    'mobilenet': list_models_from_module(models.mobilenet),
    'shufflenetv2': list_models_from_module(models.shufflenetv2)
}

print('Model List:', MODEL_LIST)

precisions=["float","half",'double']
# For post-voltaic architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.
device_name=str(torch.cuda.get_device_name(0))
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--WARM_UP','-w', type=int,default=5, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=50,required=False, help="Num of Test")
parser.add_argument('--BATCH_SIZE','-b', type=int, default=12, required=False, help='Num of batch size')
parser.add_argument('--NUM_CLASSES','-c', type=int, default=1000, required=False, help='Num of class')
parser.add_argument('--NUM_GPU','-g', type=int, default=1, required=False, help='Num of gpus')
parser.add_argument('--folder','-f', type=str, default='result', required=False, help='folder to save results')
args = parser.parse_args()
args.BATCH_SIZE*=args.NUM_GPU
class RandomDataset(Dataset):

    def __init__(self,  length):
        self.len = length
        self.data = torch.randn( 3, 224, 224,length)

    def __getitem__(self, index):
        return self.data[:,:,:,index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset( args.BATCH_SIZE*(args.WARM_UP + args.NUM_TEST)),
                         batch_size=args.BATCH_SIZE, shuffle=False,num_workers=8)
def train(precision='single'):
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    try:
        benchmark = {}
        for model_type in MODEL_LIST.keys():
            for model_constructor in MODEL_LIST[model_type]:
                model_args = {}
                # Check if 'alpha' is a required parameter and set it to 1 if needed
                if 'alpha' in inspect.signature(model_constructor).parameters:
                    model_args['alpha'] = 1
                    # Instantiate the model
                if 'pretrained' in inspect.signature(model_constructor).parameters:
                    model = model_constructor(pretrained=False, **model_args)
                else:
                    model = model_constructor(**model_args)

            # Convert model to the correct precision
                if precision == 'half':
                    model = model.half()
                elif precision == 'double':
                    model = model.double()
                else:  # default to 'float' / 'single'
                    model = model.float()

                if args.NUM_GPU > 1:
                    model = nn.DataParallel(model, device_ids=range(args.NUM_GPU))
                model = model.to('cuda')

                durations = []
                model_name = model_constructor.__name__
                print(f'Benchmarking Training {precision} precision type {model_name} ')

                for step, img in enumerate(rand_loader):
                    # Convert image to the correct precision
                    if precision == 'half':
                        img = img.half()
                    elif precision == 'double':
                        img = img.double()
                    else:  # default to 'float' / 'single'
                        img = img.float()

                    img = img.to('cuda')
                    torch.cuda.synchronize()
                    start = time.time()

                    model.zero_grad()
                    prediction = model(img)
                    loss = criterion(prediction, target)
                    loss.backward()

                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start) * 1000)

                print(f'{model_name} model average train time : {sum(durations)/len(durations)}ms')
                del model
                benchmark[model_name] = durations

    except Exception as e:
        print(f'An error occurred in {precision} precision: {e}')
    return benchmark


def inference(precision='float'):
    benchmark = {}
    with torch.no_grad():
        try:
            for model_type in MODEL_LIST.keys():
                for model_constructor in MODEL_LIST[model_type]:
                    # Instantiate the model
                    if 'pretrained' in inspect.signature(model_constructor).parameters:
                        model = model_constructor(pretrained=False)
                    else:
                        model = model_constructor()

                # Convert model to the correct precision
                    if precision == 'half':
                        model = model.half()
                    elif precision == 'double':
                        model = model.double()
                    else:  # default to 'float' / 'single'
                        model = model.float()

                    if args.NUM_GPU > 1:
                        model = nn.DataParallel(model, device_ids=range(args.NUM_GPU))
                    model = model.to('cuda')
                    model.eval()

                    durations = []
                    model_name = model_constructor.__name__
                    print(f'Benchmarking Inference {precision} precision type {model_name} ')

                    for step, img in enumerate(rand_loader):
                        # Convert image to the correct precision
                        if precision == 'half':
                            img = img.half()
                        elif precision == 'double':
                            img = img.double()
                        else:  # default to 'float' / 'single'
                            img = img.float()

                        img = img.to('cuda')
                        torch.cuda.synchronize()
                        start = time.time()

                        model(img)

                        torch.cuda.synchronize()
                        end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start) * 1000)

                print(f'{model_name} model average inference time : {sum(durations)/len(durations)}ms')
                del model
                benchmark[model_name] = durations

        except Exception as e:
            print(f'An error occurred in {precision} precision: {e}') 
        return benchmark


f"{platform.uname()}\n{psutil.cpu_freq()}\ncpu_count: {psutil.cpu_count()}\nmemory_available: {psutil.virtual_memory().available}"


if __name__ == '__main__':
    folder_name=args.folder
    
    device_name=f"{device_name}_{args.NUM_GPU}_gpus_"
    system_configs=f"{platform.uname()}\n\
                     {psutil.cpu_freq()}\n\
                    cpu_count: {psutil.cpu_count()}\n\
                    memory_available: {psutil.virtual_memory().available}"
    gpu_configs=[torch.cuda.device_count(),torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0)]
    gpu_configs=list(map(str,gpu_configs))
    temp=['Number of GPUs on current device : ','CUDA Version : ','Cudnn Version : ','Device Name : ']

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    now = datetime.datetime.now()
    
    start_time=now.strftime('%Y/%m/%d %H:%M:%S')
    
    print(f'benchmark start : {start_time}')

    for idx,value in enumerate(zip(temp,gpu_configs)):
        gpu_configs[idx]=''.join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name,"system_info.txt"), "w") as f:
        f.writelines(f'benchmark start : {start_time}\n')
        f.writelines('system_configs\n\n')
        f.writelines(system_configs)
        f.writelines('\ngpu_configs\n\n')
        f.writelines(s + '\n' for s in gpu_configs )

    
    for precision in precisions:
        train_result=train(precision)
        train_result_df = pd.DataFrame(train_result)
        path=f'{folder_name}/{device_name}_{precision}_model_train_benchmark.csv'
        train_result_df.to_csv(path, index=False)

        inference_result=inference(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path=f'{folder_name}/{device_name}_{precision}_model_inference_benchmark.csv'
        inference_result_df.to_csv(path, index=False)

    now = datetime.datetime.now()

    end_time=now.strftime('%Y/%m/%d %H:%M:%S')
    print(f'benchmark end : {end_time}')
    with open(os.path.join(folder_name,"system_info.txt"), "a") as f:
        f.writelines(f'benchmark end : {end_time}\n')
        
