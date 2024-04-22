# /data2/pkw/MotionGPT/rlhf/seed_1234
# /data2/pkw/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=/data2/pkw/MotionGPT/rlhf/labeling/0_M004660 --mode=video

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["MUJOCO_GL"] = "egl"

import imageio
import random
import torch
import time
import cv2
import os
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import moviepy.editor as mp
from pathlib import Path
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
from scipy.spatial.transform import Rotation as RRR
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mGPT.render.pyrender.smpl_render import SMPLRender
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import time
import argparse

# data_dir = "/data2/pkw/MotionGPT/rlhf/seed_1234"

# import time
# starttime = time.time()
# for dir in sorted(os.listdir(data_dir))[:10]:

#     dir_path = os.path.join(data_dir, dir)
#     command1 = f"python -m fit --dir {dir_path} --save_folder {dir_path} --cuda 0; wait; /data2/pkw/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir={dir_path} --mode=video"
#     # command2 = f"/data2/pkw/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir={dir_path} --mode=video"
#     print(command1)
#     os.system(command1)
#     # os.system(command2)
#     print("Done with ", dir)

# print("Total time for 10: ", time.time() - starttime)


def render_motion(data, output_mp4_path, method='fast'):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'

    if method == 'slow':
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        render = SMPLRender("./deps/smpl_models/smpl")

        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        vid = []
        aroot = data[[0], 0]
        aroot[:, 1] = -aroot[:, 1]
        params = dict(pred_shape=np.zeros([1, 10]),
                      pred_root=aroot,
                      pred_pose=pose)
        render.init_renderer([shape[0], shape[1], 3], params)
        for i in range(data.shape[0]):
            renderImg = render.render(i)
            # images already have the artifact
            img = Image.fromarray(renderImg)
            #img.save(f"/data2/pkw/MotionGPT/frame_obama_{i}.png")
            #import pdb; pdb.set_trace()
            # renderImg = Image.fromarray(renderImg).convert("RGB")
            vid.append(renderImg)

        out = np.stack(vid, axis=0)
        output_gif_path = output_mp4_path[:-4] + '.gif'
        imageio.mimwrite(output_gif_path, out, loop=1, duration=50)
        out_video = mp.VideoFileClip(output_gif_path).loop(n=1)
        out_video.write_videofile(output_mp4_path)
        del out, render

    elif method == 'fast': # stick figure
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
        out_video = mp.VideoFileClip(output_gif_path).loop(n=10)
        out_video.write_videofile(output_mp4_path)
        del pose_vis

    return output_mp4_path, video_fname

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_dir", type=str, default="./outputs/generated_npys")
    parser.add_argument("--prompts", type=str, default="./preference_data/selected_prompts_test.txt")
    parser.add_argument("--video_dir", type=str, default="./outputs/generated_videos")
    parser.add_argument("--model_name", type=str, default="motiongpt")
    return parser.parse_args()


def main():
    args = arguments()
    seed = args.seed
    data_dir = args.data_dir
    prompts = args.prompts
    video_dir = args.video_dir
    model_name = args.model_name

    starttime = time.time()
    
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    with open(prompts, 'r') as f:
        for i, line in enumerate(f.readlines()):
            dir = line.split("#")[0].strip()
            dir_path = os.path.join(data_dir, dir)
            save_path = os.path.join(video_dir, dir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            joints = np.load(os.path.join(dir_path, f"{model_name}_{seed}_{dir}_out.npy"))
            output_mp4_path = os.path.join(save_path, "{}_{}_{}_out.mp4".format(model_name, seed, dir))
            output_mp4_path, video_fname = render_motion(
                joints, output_mp4_path, "slow")

            text_path = os.path.join(dir_path, "text.txt")
            with open(text_path, 'r') as f:
                prompt = f.read().strip()   
                new_text_path = os.path.join(save_path, "text.txt")
                with open(new_text_path, 'w') as f:
                    f.write(prompt)
            
            print("Done with video ", i, " ", dir, " ", output_mp4_path)


if __name__ == "__main__":
    main()