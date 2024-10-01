
import cv2
import time
import numpy as np
import math
from adafruit_servokit import ServoKit


print("cv2.__version__")

# servo set up
kit = ServoKit(channels=8)
angle_speed = 5
servo_angle = 90
scanRight = True

kit.servo[0].angle = servo_angle





# Cascade Set Up
face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier()

if face_cascade.load(cv2.samples.findFile(face_cascade_path)):
	print('Success Loaded Cascade!')
else:
	print('Error unable to load cascade!')
	exit(0)	
	

def GetCenterOfFace(faceCordinate):
	(x,y,w,h) = faceCordinate
	center = (x + w // 2, y + h // 2)
	return center


def DrawCircleOnFace(image, faceCoordinate):
	(x,y,w,h) = faceCoordinate
	center = GetCenterOfFace(faceCoordinate)
	face_image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (0,125,125), 4)
	
	return face_image
	
def ShootWater():
	start = time.time()
	duration = 1
	
	while(time.time() - start < duration):
		x = 3 + 4
		print("shooting water...")
	
	


def MoveMotorRight(moveRight = True):
	global servo_angle, angle_speed, scanRight
	
	if moveRight:
		servo_angle = servo_angle - angle_speed
		if servo_angle <= 5:
			servo_angle = 5
			scanRight = False
		kit.servo[0].angle = servo_angle
	else:
		servo_angle = servo_angle + angle_speed
		if servo_angle >= 175:
			servo_angle = 175
			scanRight = True
		kit.servo[0].angle = servo_angle

	print('angle:', servo_angle)


def MoveMotor(image, faceCoordinate):
	THRESHOLD_PERCENT = .1
	height, width, _ = image.shape

	center_x = width / 2
	center_y = height / 2	
	
	(face_x, face_y) = GetCenterOfFace(faceCoordinate)
	
	# diff < 0 means to the left or top
	diff_x = face_x - center_x 
	diff_y = face_y - center_y

	print(width, height, diff_x, diff_y, face_x, face_y, center_x, center_y, abs(diff_x / width), abs(diff_y / height))
	
	if abs(diff_x / width) > THRESHOLD_PERCENT:
		if diff_x > 0:
			print("you are to the right")
			MoveMotorRight()
		else:
			print("you are to the left")
			MoveMotorRight(False)

	else:
		print("You are ok in the x axis")
		ShootWater()
	
	if abs(diff_y / height) > THRESHOLD_PERCENT:
		if diff_y > 0:
			print("you are to the bottom")
		else:
			print("you are to the top")
	else:
		print("You are ok in the y axis")

def DetectLargestFace(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_image = cv2.equalizeHist(gray_image)
	
	face_image = image.copy()
		
	faces = face_cascade.detectMultiScale(gray_image, minNeighbors = 4)
	if len(faces) > 0:
	
		areas = [w*h for x,y,w,h in faces]
		i_biggest = np.argmax(areas)
		largestFace = faces[i_biggest]

		return largestFace

	return None



def DetectLargestAndDrawFace(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_image = cv2.equalizeHist(gray_image)
	
	face_image = image.copy()
		
	faces = face_cascade.detectMultiScale(gray_image, minNeighbors = 4)
	
	if len(faces) > 0:
	
		areas = [w*h for x,y,w,h in faces]
		i_biggest = np.argmax(areas)
		largeFace = faces[i_biggest]

		(x,y,w,h) = largeFace

		center = (x + w // 2, y + h // 2)

		face_image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (0,125,125), 4)
				
	cv2.imshow("face",face_image)

	
def DetectAndDrawFace(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_image = cv2.equalizeHist(gray_image)
	
	face_image = image.copy()
		
	faces = face_cascade.detectMultiScale(gray_image, minNeighbors = 4)
	for (x,y,w,h) in faces:
		center = (x + w // 2, y + h // 2)
		face_image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (0,125,125), 4)
			
	cv2.imshow("face",face_image)
	
	
# Camera Set Up
camera = cv2.VideoCapture(0)
if  camera.isOpened:
	print('Success Camera is Open!')
else:
	print('Error unable to open camera!')
	exit(0)



# Main Loop
while True:
	ret,image = camera.read()
	
	if image is None:
		print('Error unable to read frame!')
		break	
	
	image = cv2.flip(image, -1)
	
	face_coord = DetectLargestFace(image)
	if not face_coord is None:
		image = DrawCircleOnFace(image, face_coord)
		MoveMotor(image, face_coord)
	else:
		#scanmode
		MoveMotorRight(scanRight)
		
			
	cv2.imshow("face",image)
	
	if cv2.waitKey(42) & 0xFF == ord("q"):
		break


camera.release()
cv2.destroyAllWindows()
