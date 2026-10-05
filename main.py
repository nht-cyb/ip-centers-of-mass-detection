import mainloop

imgdir="/home/pi/Desktop/Captures/"
imgprefix="CapF"

def ImageDetection():
        #Work on Image Detection (press ESC when done, Space to capture image)
        fullscreen=False
        #set detect XYZ to False when you want to use this loop to capture pictures (press spacebar)
        detectXYZ=True
        #set calculateXYZ to enable real world XYZ to be calculated
        calculateXYZ=True
        mainloop.capturefromPiCamera(imgdir,imgprefix,fullscreen,detectXYZ,calculateXYZ)

ImageDetection()