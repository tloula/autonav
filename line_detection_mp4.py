from advanced_functions import vid_pipeline
from moviepy.editor import VideoFileClip
import sys

def main(args):

    right_curves, left_curves = [],[]

    myclip = VideoFileClip('project_video.mp4') #.subclip(40,43)
    clip = myclip.fl_image(vid_pipeline)
    output_vid = 'output.mp4'
    clip.write_videofile(output_vid, audio=False)

if __name__ == "__main__":
    main(sys.argv)
