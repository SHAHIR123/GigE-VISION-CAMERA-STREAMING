#!/usr/bin/env python

"""
Code For STreaming YUV422_YUYV_PACKED Pixel Format from a GIGE Vision Camera
"""
__author__ = " Shahir K"

#import datetime
#from numpy import *
#import time
#from glob import glob
import cv2
# camera settings
from harvesters.core import Buffer, Harvester
#from harvesters.util.pfnc import mono_location_formats, \
#    rgb_formats, bgr_formats, \
#    rgba_formats, bgra_formats

# Set width, height and pixel format of frame if you know the details.
WIDTH = 720  # Image buffer width as per the camera output
HEIGHT = 576  # Image buffer height as per the camera output
PIXEL_FORMAT = "YUV422_YUYV_PACKED"  # Camera pixel format as per the camera output


h = Harvester()
h.add_file("C:\\Program Files\\MATRIX VISION\\mvIMPACT Acquire\\bin\\x64\\mvGenTLProducer.cti") # Path to mvGenTLProducer.cti
#h.files
h.update()
print(h.device_info_list[0])

io = h.create_image_acquirer(0)
io.remote_device.node_map.Width.value = WIDTH 
io.remote_device.node_map.Width.value = WIDTH 
io.remote_device.node_map.PixelFormat.value = PIXEL_FORMAT
#io.remote_device.node_map.AcquisitionFrameRate.value = fps # Set if required 
io.start_acquisition()
#print(len(h.device_info_list))

i = 0

# content_type = 'image/jpeg'
# headers = {'content-type': content_type}


output_filename = 'video.avi' # Save stream
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, 10.0, (720, 576))


while cv2.waitKey(1) != 27:
    Buffer = io.fetch_buffer(timeout=-1)    
    component = Buffer.payload.components[0]
    #print(component.width)
    if component.width == 720: # To make sure the correct size frames are passed for converting
        original = component.data.reshape(576, 720, 2)
        img = original.copy() # To prevent isues due to buffer queue
        #print(size(img))
        image = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)

        # Place your trained model here when running computer vision model on this stream.      
               
        cv2.imshow('img', image)
        out.write(cv2.resize(image, (720, 576)))
        Buffer.queue()
        #time.sleep(0.03)
        i +=1
    else:
        i +=1


out.release()    
io.stop_acquisition()
io.destroy()
h.reset()
cv2.destroyAllWindows()


