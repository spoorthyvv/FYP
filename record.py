import sys, os, cv2, time, readchar, datetime
import soundfile as sf 
import sounddevice as sd 
import numpy as np
from multiprocessing import Process
import ray, json
import subprocess32 as sp
import pyautogui, shutil, zipfile 
from natsort import natsorted

curdir=os.getcwd()
try:
	os.mkdir('temp')
	os.chdir('temp')
except:
	shutil.rmtree('temp')
	os.mkdir('temp')
	os.chdir('temp')

# def zip(src, dst):
#     zf = zipfile.ZipFile("%s.zip"%(dst), "w", zipfile.ZIP_DEFLATED)
#     abs_src = os.path.abspath(src)
#     for dirname, subdirs, files in os.walk(src):
#         for filename in files:
#             absname = os.path.abspath(os.path.join(dirname, filename))
#             arcname = absname[len(abs_src) + 1:]
#             print('zipping %s as %s'%(os.path.join(dirname, filename),arcname))
#             zf.write(absname, arcname)
#     zf.close()

@ray.remote
def video_record(filename, duration):
	print('recording video (.AVI)')
	print('--> '+filename)

	t0 = time.time() # start time in seconds

	video=cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	frame_width = int(video.get(3))
	frame_height = int(video.get(4))
	out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

	a=0
	start=time.time()

	while True:
		a=a+1
		check, frame=video.read()
		#print(check)
		#print(frame)
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		out.write(frame)
		#cv2.imshow("frame",gray)
		end=time.time()
		if end-start>duration:
		    break 

	print(a)
	video.release()
	out.release() 
	cv2.destroyAllWindows()

@ray.remote 
def audio_record(filename, duration):
	print('recording audio (.WAV)')
	print('--> '+filename)
	time.sleep(0.50)
	fs=44100
	channels=1
	myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
	sd.wait()
	sf.write(filename, myrecording, fs)

def video_audio_record(videofile, duration):
	# record each in parallel 
	# runInParallel(audio_record(filename[0:-4]+'.wav', duration), video_record(filename,duration))
	audiofile=filename[0:-4]+'.wav'
	ray.get([video_record.remote(videofile,duration), audio_record.remote(audiofile, duration)])
	#os.system('ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s'%(videofile, audiofile, videofile))
	#os.remove(audiofile)
	# connect two files 

filename=sys.argv[1]
duration=int(sys.argv[2])

print(filename)
print(duration)

if filename.find('.avi') > 0:
	ray.init()
	video_audio_record(filename, duration)

# for testing !! (calculate the right framerate for duration)


# initialize names of stuff 
audiofile=filename[0:-4]+'.wav'
newfilename=filename[0:-4]+'_new.mp4'
newfilename2=filename[0:-4]+'_new2.mp4'

if vid_duration > duration or vid_duration < duration:
	# convert to be proper length
	print('converting to 20 seconds of video...')
	# following ffmpeg documentation https://trac.ffmpeg.org/wiki/How%20to%20speed%20up%20/%20slow%20down%20a%20video
	speedfactor=duration/vid_duration
	print(speedfactor)
	os.system('ffmpeg -i %s -filter:v "setpts=%s*PTS" %s'%(filename, str(speedfactor), newfilename))
	os.system('ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s'%(newfilename, audiofile, newfilename2))
	#os.remove(filename)
	#os.remove(newfilename)
	#os.rename(newfilename2, filename)
else:
	os.system('ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s'%(filename, audiofile, newfilename2))
	#os.remove(filename)
	#os.rename(newfilename2, filename)


# make everything into one video 

one=newfilename2[0:-4]+'_.mp4'
two=filename[0:-4]+'_screenshots_2.mp4'

#resize video 1 
os.system('ffmpeg -i %s -vf scale=640:360 %s -hide_banner -preset ultrafast -framerate 30 -vf mpdecimate -c:a copy -vsync vfr'%(newfilename2, one))
# resize video 2 
os.system('ffmpeg -i %s -vf scale=640:360 %s -hide_banner -preset ultrafast -framerate 30 -vf mpdecimate -c:a copy -vsync vfr'%(filename[0:-4]+'_screenshots.mp4', two))

# combine 
os.system('ffmpeg -i %s -i %s -filter_complex hstack output.mp4 -preset ultrafast -framerate 30 -vf mpdecimate -c:a copy -vsync vfr'%(one, two))
#os.system('open output.mp4')

# remove temp files and rename
os.remove(one)
os.remove(two)
os.remove(filename)
os.rename(newfilename, filename[0:-4]+'.mp4')
os.remove(filename[0:-4]+'.mp4')
os.rename(newfilename2, filename[0:-4]+'.mp4')
shutil.rmtree(filename[0:-4]+'_screenshots')

os.chdir(curdir)
zip("temp", filename[0:-4])
shutil.rmtree('temp')

