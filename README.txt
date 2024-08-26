

We need to use a  NVIDIA A100-SXM4-80GB for the LPIPS to run properly.

We launch using:

'''
accelerate launch train_lpips_text_to_image.py
'''

condor_submit_bid 1000 -i -append request_memory=281920 -append request_cpus=10 -append request_disk=100G -append request_gpus=2 -append 'requirements = CUDADeviceName == "NVIDIA A100-SXM4-80GB"'