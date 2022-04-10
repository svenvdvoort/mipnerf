#!/usr/bin/env python3

from colorama import Fore, Style
import numpy as np
import os
import subprocess

####### Script configuration #######
data_dir = "data/nerf_synthetic_data/lego_small_raw"
results_dir = "nerf_results/debug/lego_small_raw_results_09042022"
raw_format = True
tmp_working_dir = "tmp_working_dir"
####################################

def printg(message):
    print(Fore.GREEN + message + Style.RESET_ALL)

printg(f"Creating temporary working directory {tmp_working_dir}")
os.system(f"mkdir -p {tmp_working_dir}")

psnr_file = open("psnr_scores.txt", "a")
psnr_file.write("checkpoint, psnr_score_average, psnr_score_variance\n")
ssim_file = open("ssim_scores.txt", "a")
ssim_file.write("checkpoint, ssim_score_average, ssim_score_variance\n")

for i in range(10000, 500000, 10000):
    try:
        printg(f"Running eval on checkpoint_{i}")
        os.system(f"cp {results_dir}/checkpoint_{i} {tmp_working_dir}/")
        eval_command = f"python3 -m eval --gin_param=Config.raw_format={'True' if raw_format else 'False'} --data_dir={data_dir} --train_dir={tmp_working_dir} --chunk=3076 --gin_file=configs/blender.gin --logtostderr".split(" ")
        eval_output = subprocess.run(eval_command)
        eval_output.check_returncode()

        with open(f"{tmp_working_dir}/test_preds/psnrs_{i}.txt", "r") as f:
            psnr_scores = list(map(float, f.read().split(" ")))
            psnr_score_average = np.average(psnr_scores)
            psnr_score_variance = np.var(psnr_scores)
            psnr_file.write(f"{i}, {psnr_score_average}, {psnr_score_variance}\n")
            psnr_file.flush()
        with open(f"{tmp_working_dir}/test_preds/ssims_{i}.txt", "r") as f:
            ssim_scores = list(map(float, f.read().split(" ")))
            ssim_score_average = np.average(ssim_scores)
            ssim_score_variance = np.var(ssim_scores)
            ssim_file.write(f"{i}, {ssim_score_average}, {ssim_score_variance}\n")
            ssim_file.flush()

        printg(f"Eval on checkpoint_{i} is done. Average psnr score: {psnr_score_average} (variance: {psnr_score_variance}), average ssim score: {ssim_score_average} (variance: {ssim_score_variance}).")
        os.system(f"rm {tmp_working_dir}/checkpoint_{i}")
    except Exception as e:
        printg(f"Failed to eval checkpoint {i}, error: {e}")
        printg("Continue? (yes/no)")
        user_input = input()
        if user_input != "yes":
            break

psnr_file.close()
ssim_file.close()

printg("Completed checkpoints eval.")
printg("CSV files with average scores are available at psnr_scores.txt and ssim_scores.txt")
printg(f"Text files with individual scores are available at {tmp_working_dir}/test_preds/")
