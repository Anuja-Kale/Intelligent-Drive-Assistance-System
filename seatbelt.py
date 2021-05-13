from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import geocoder
import time
from playsound import playsound
import dlib
import cv2
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


import random
speed = [60, 65, 70, 75 , 80, 85, 90, 95, 100]
seat = False
yawn = False
import os



def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear

def Slope(a,b,c,d):
	return (d - b)/(c - a)

def final_ear(shape):
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]

	leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)

	ear = (leftEAR + rightEAR) / 2.0
	return (ear, leftEye, rightEye)

def lip_distance(shape):
	top_lip = shape[50:53]
	top_lip = np.concatenate((top_lip, shape[61:64]))

	low_lip = shape[56:59]
	low_lip = np.concatenate((low_lip, shape[65:68]))

	top_mean = np.mean(top_lip, axis=0)
	low_mean = np.mean(low_lip, axis=0)

	distance = abs(top_mean[1] - low_mean[1])
	return distance



def Send_Mail_Alert():
	fromaddr="abc@gmail.com"  #sender gmail address
	toaddr="abcd@gmail.com"   #reciver gmail address
	msg=MIMEMultipart()
	msg['From']=fromaddr
	msg['To']=toaddr
	msg['Subject']="found"
	body="sent from xyz"
	msg.attach(MIMEText(body,'plain'))
	filename="alert.png"
	attachment=open("alert.png","rb") #image folder

	p=MIMEBase('application','octet-stream')
	p.set_payload((attachment).read())

	encoders.encode_base64(p)

	p.add_header('Content-Disposition',"attachment; filename=%s"%filename)
	msg.attach(p)

	s=smtplib.SMTP('smtp.gmail.com',587)

	s.starttls()

	s.login(fromaddr,"8652415809n") #enter sender gmail password here

	text=msg.as_string()

	s.sendmail(fromaddr,toaddr,text)

	s.quit()


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
				help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()

time.sleep(1.0)

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=850)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if not(seat):
		# No Belt Detected Yet
		belt = False

		# Bluring The Image For Smoothness
		blur = cv2.blur(gray, (1, 1))

		# Converting Image To Edges
		edges = cv2.Canny(blur, 50, 400)


		# Previous Line Slope
		ps = 0

		# Previous Line Co-ordinates
		px1, py1, px2, py2 = 0, 0, 0, 0

		# Extracting Lines
		lines = cv2.HoughLinesP(edges, 1, np.pi/270, 30, maxLineGap = 20, minLineLength = 170)

		# If "lines" Is Not Empty
		if lines is not None:

			
			for line in lines:

				# Co-ordinates Of Current Line
				x1, y1, x2, y2 = line[0]

				# Slope Of Current Line
				s = Slope(x1,y1,x2,y2)
				
				# If Current Line's Slope Is Greater Than 0.7 And Less Than 2
				if ((abs(s) > 0.7) and (abs (s) < 2)):

					# And Previous Line's Slope Is Within 0.7 To 2
					if((abs(ps) > 0.7) and (abs(ps) < 2)):

						# And Both The Lines Are Not Too Far From Each Other
						if(((abs(x1 - px1) > 5) and (abs(x2 - px2) > 5)) or ((abs(y1 - py1) > 5) and (abs(y2 - py2) > 5))):

							# Plot The Lines On "frame"
							cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
							cv2.line(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

							# Belt Is Detected
							print ("Belt Detected")
							belt = True

				# Otherwise Current Slope Becomes Previous Slope (ps) And Current Line Becomes Previous Line (px1, py1, px2, py2)            
				ps = s
				px1, py1, px2, py2 = line[0]
			
					
		if belt == False:
			cv2.putText(frame, "No seat belt Detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
		else:
			cv2.putText(frame, "Seat Belt Confirmed", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)



	if not(yawn):

		rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
			minNeighbors=5, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)

		if(len(rects)==0):
			cv2.putText(frame, "Face Not Found!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


		for (x, y, w, h) in rects:
			rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
			
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			eye = final_ear(shape)
			ear = eye[0]
			leftEye = eye [1]
			rightEye = eye[2]

			distance = lip_distance(shape)

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			lip = shape[48:60]
			cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

			if ear < EYE_AR_THRESH:
				COUNTER += 1

				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					# if alarm_status == False:
					# 	alarm_status = True
					# 	t = Thread(target=alarm, args=('wake up sir',))
					# 	t.deamon = True
					# 	t.start()

					cv2.putText(frame, "DROWSINESS ALERT!", (400, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					cv2.imwrite("alert.png", frame)
					playsound('beep.mp3')
					g = geocoder.ip('me')

					a = g.latlng

					lat=str(a[0])
					lon=str(a[1])

					print(lat)
					print(lon)
					url1 = ("http://my-demo.in/Driverdrowsiness/request.aspx?l1="+lat+"&l2="+lon+"&userid=2004")
					import urllib.request
					contents = urllib.request.urlopen(url1).read()
					Send_Mail_Alert()
					time.sleep(1.0)
			else:
				COUNTER = 0
				alarm_status = False

			if (distance > YAWN_THRESH):
					cv2.putText(frame, "Yawn Alert", (400, 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					# if alarm_status2 == False and saying == False:
					# 	alarm_status2 = True
					# 	t = Thread(target=alarm, args=('take some fresh air sir',))
					# 	t.deamon = True
					# 	t.start()
			else:
				alarm_status2 = False

			cv2.putText(frame, "EAR: {:.2f}".format(ear), (350, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "YAWN: {:.2f}".format(distance), (500, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
