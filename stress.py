import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.resnet import resnet50
import torch
import time
import numpy as np
import os

torch.backends.cudnn.benchmark = True

# Set environment variable to handle memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

NUM_GPUS = 1  # Set to 1 to use only one GPU
BATCH_SIZE = 1024  # Adjust batch size as needed
WARM_UP = 5
NUM_STEP = 100
ACCUMULATION_STEPS = 10  # Number of steps to accumulate gradients (adjustable)

def main():
    benchmark = []
    print('Benchmarking ResNet50 on batch size %i with %i GPU' % (BATCH_SIZE, NUM_GPUS))
    model = resnet50()
    model.cuda()
    model.eval()

    img = Variable(torch.randn(BATCH_SIZE // ACCUMULATION_STEPS, 3, 224, 224)).cuda()

    # Enable mixed precision with the updated method
    scaler = torch.amp.GradScaler('cuda')

    durations = []
    for step in range(NUM_STEP + WARM_UP):
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(ACCUMULATION_STEPS):
            with torch.amp.autocast('cuda'):
                model(img)

        torch.cuda.synchronize()
        end = time.time()

        if step >= WARM_UP:
            duration = (end - start) * 1000
            durations.append(duration)

        torch.cuda.empty_cache()

    benchmark.append(durations)
    del model
    return benchmark

if __name__ == '__main__':
    benchmark = main()
    for duration in benchmark:
        print('Batch size %i, time cost %.4fms' % (BATCH_SIZE, np.mean(duration)))
