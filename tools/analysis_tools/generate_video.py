# generate gif and video from images
import glob 
import cv2 # for video
import imageio.v2 as imageio # for gif
import os
from pygifsicle import optimize

def generate_video(image_dir, out_file, fps=10):
    images = glob.glob(os.path.join(image_dir, '*.png'))
    images = sorted(images)

    img = cv2.imread(images[0])
    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
    
    for image in images:
        img = cv2.imread(image)
        video.write(img)
    
    cv2.destroyAllWindows()
    video.release()
    
    print(f'Video saved to {out_file}')


def generate_gif(image_dir, out_file, fps=10):
    images = glob.glob(os.path.join(image_dir, '*.png'))
    images = sorted(images)[:500]
    
    img = cv2.imread(images[0])
    h, w, _ = img.shape
    
    with imageio.get_writer(out_file, mode='I', fps=fps) as writer:
        for image in images:
            img = imageio.imread(image)
            img = cv2.resize(img, (w//2, h//2))
            writer.append_data(img)
            
    print(f'Gif saved to {out_file}')

# generate video
#generate_video('tmp', 'tmp.mp4', fps=12)
generate_gif('tmp', 'tmp.gif', fps=12)
optimize('tmp.gif') # For overwriting the original one
