# generate gif and video from images
# Given a results folder with annoted images, generate a video and gif
# and save them to the specified output file
import glob 
import cv2 # for video
import imageio.v2 as imageio # for gif
import os
from pygifsicle import optimize

def generate_video(images, out_file, fps=10):
    """
    Generate a video from images
    Args:
        images (list): list of images
        out_file (str): output file name
        fps (int): frames per second
    """
    img = cv2.imread(images[0])
    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
    
    for img in images:
        img = cv2.imread(img)
        video.write(img)
    
    cv2.destroyAllWindows()
    video.release()
    
    
def generate_videos(image_dir, out_file, fps=10):
    # load all images with png or jpg extension
    pngs = glob.glob(os.path.join(image_dir, '*.png')) + glob.glob(os.path.join(image_dir, '*.jpg'))
    # may contain cam and lidar images
    cam_pngs = [png for png in pngs if 'CAM' in png]
    lidar_pngs = [png for png in pngs if 'LIDAR' in png]
    
    # cam video
    cam_pngs = sorted(cam_pngs)
    generate_video(cam_pngs, out_file+'_cam.mp4', fps)
    
    # lidar video
    lidar_pngs = sorted(lidar_pngs)
    generate_video(lidar_pngs, out_file+'_lidar.mp4', fps)

def generate_gif(images, out_file, fps=10):
    
    img = cv2.imread(images[0])
    h, w, _ = img.shape
    with imageio.get_writer(out_file, mode='I', fps=fps) as writer:
        for image in images:
            img = imageio.imread(image)
            img = cv2.resize(img, (w//2, h//2))
            writer.append_data(img)
            
def generate_gifs(image_dir, out_file, fps=10):
    pngs = glob.glob(os.path.join(image_dir, '*.png')) + glob.glob(os.path.join(image_dir, '*.jpg'))
    # may contain cam and lidar images
    cam_pngs = [png for png in pngs if 'CAM' in png]
    lidar_pngs = [png for png in pngs if 'LIDAR' in png]
    
    # cam video
    cam_pngs = sorted(cam_pngs)
    generate_gif(cam_pngs, out_file+'_cam.gif', fps)
    optimize(out_file+'_cam.gif')
    
    # lidar video
    lidar_pngs = sorted(lidar_pngs)
    generate_gif(lidar_pngs, out_file+'_lidar.gif', fps)
    optimize(out_file+'_lidar.gif')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate video and gif from images')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--out-file', type=str, default='video', help='Output file name')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_dir
    out_file = args.out_file
    fps = args.fps

    # Generate video
    generate_videos(image_dir, out_file, fps)

    # Generate gif
    generate_gifs(image_dir, out_file, fps)
