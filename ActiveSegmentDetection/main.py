import numpy as np
from ActiveSegmentDetection.ShortEnergyFun import Active_Segment_Detection
from ActiveSegmentDetection.DetectionFun import Save_and_Filtering_Active_segment
from ActiveSegmentDetection.DetectionAndShortEnergyShow import Show_active_segment
import os

'''
Signal preprocessing master file
'''

th1 = [0.00000691301041466369,0.000013172796743014,0.00000891301041466369,0.00000816301041466369,0.00000909479674301402,0.0000209994796743014,0.0000259994796743014,
	   0.000008463464834161,0.0000219975851229286,0.000006463464834161,0.0000199048610609335,0.0000110768431103039,0.0000259910768431103,0.0000259099107684311,
	   0.0000139130104146636,0.00000793172796743014,0.00000691301041466369,0.00000816301041466369,0.0000209994796743014,0.0000109994796743014,0.0000259994796743014,
	   0.000006463464834161,0.00000299758512292866,0.000006463464834161,0.00000890486106093358,0.0000110768431103039,0.0000259910768431103,0.0000109099107684311,
	   0.00000691301041466369,0.00000493172796743014,0.00000691301041466369,0.00000516301041466369,0.0000209994796743014,0.0000109994796743014,0.0000209994796743014,
	   0.000006463464834161,0.000007463464834161,0.000006463464834161,0.00000890486106093358,0.0000110768431103039,0.0000259910768431103,0.0000159099107684311,
	   0.0000259913010414663,0.00000369130104146636,0.00000691301041466369,0.00000506301041466369,0.0000209994796743014,0.0000109994796743014,0.0000109994796743014,
	   0.000006463464834161,0.00000199863464834161,0.000006463464834161,0.00000490486106093358,0.00000991076843110303,0.0000259910768431103,0.0000109099107684311,
	   0.0000189130104146636,0.00000691301041466369,0.00000691301041466369,0.00000506301041466369,0.0000209994796743014,0.0000109994796743014,0.0000109994796743014,
	   0.000006463464834161,0.000006463464834161,0.000006463464834161,0.00000490486106093358,0.00000891076843110303,0.00000909910768431103,0.000009,
	   0.0000189130104146636,0.00000691301041466369,0.00000691301041466369,0.00000607301041466369,0.0000209994796743014,0.0000109994796743014,0.00000509994796743014,
	   0.000006463464834161,0.000006463464834161,0.000006463464834161,0.00000609048610609335,0.00000891076843110303,0.00000909910768431103,0.00000400000910768431,
	   0.00000691301041466369,0.00000391301041466369,0.00000691301041466369,0.00000507301041466369,0.0000209994796743014,0.0000109994796743014,0.00000509994796743014,
	   0.0000179130104146636,0.000004463464834161,0.000006463464834161,0.00000500904861060933,0.00000991076843110303,0.00000909910768431103,0.00000400000910768431,
	   0.00000691301041466369,0.00000999,0.00000691301041466369,0.0000299973010414663,0.0000209994796743014,0.0000259994796743014,0.00000509994796743014,
	   0.0000179130104146636,0.0000459154070574911,0.0000299946346483416,0.0000299909048610609,0.0000259910768431103,0.0000259099107684311,0.00000800000910768431
	   ]
th2 = [0.000026815563718975,0.000010786364389788,0.00000547738128005903,0.00000491228448541769,0.00000294686207066732,0.00000280468620706673,0.00000674686207066732,
	   0.00000767684218755959,0.00000518303946068445,0.00000489803912110641,0.00000435798340628471,0.000010991204414869,0.0000050991204414869,0.0000169991204414869,
	   0.000002415563718975,0.000003586364389788,0.0000016315563718975,0.00000221228448541769,0.00000204686207066732,0.00000280468620706673,0.00000274686207066732,
	   0.00000127684218755959,0.0000009,0.0000014665468755959,0.00000225798340628471,0.0000040991204414869,0.0000030991204414869,0.0000010091204414869,
	   0.00000242663128005903,0.000002086364389788,0.0000016315563718975,0.00000281228448541769,0.00000204686207066732,0.00000280468620706673,0.00000204686207066732,
	   0.00000127684218755959,0.00000367684218755959,0.0000014665468755959,0.00000225798340628471,0.0000012991204414869,0.0000030991204414869,0.0000010091204414869,
	   0.000019926631280059,0.00000100681556371897,0.0000016315563718975,0.00000251228448541769,0.00000204686207066732,0.00000280468620706673,0.00000204686207066732,
	   0.00000022684218755959,0.000000120966546875595,0.0000013665468755959,0.00000105798340628471,0.0000010991204414869,0.0000030991204414869,0.0000010001204414869,
	   0.000014426631280059,0.0000036815563718975,0.0000026315563718975,0.00000261228448541769,0.00000304686207066732,0.00000150468620706673,0.00000584686207066732,
	   0.00000267684218755959,0.00000129665468755959,0.0000023665468755959,0.00000205798340628471,0.00000250991204414869,0.00000220991204414869,0.000002,
	   0.000015426631280059,0.0000022815563718975,0.0000026315563718975,0.00000301122844854176,0.00000254686207066732,0.00000250468620706673,0.00000204686207066732,
	   0.00000607684218755959,0.00000229665468755959,0.0000023665468755959,0.00000150579834062847,0.00000250991204414869,0.0000020991204414869,0.00000149001204414869,
	   0.00000242663128005903,0.00000126815563718975,0.0000026315563718975,0.00000150112284485417,0.00000250468620706673,0.00000200468620706673,0.00000204686207066732,
	   0.00000102663128005903,0.00000229665468755959,0.0000023665468755959,0.00000100057983406284,0.00000200991204414869,0.0000020991204414869,0.00000149001204414869,
	   0.00000242663128005903,0.000005,0.0000026315563718975,0.00000400112284485417,0.00000200468620706673,0.00000274686207066732,0.00000350468620706673,
	   0.00000522663128005903,0.0000074,0.0000025665468755959,0.000004,0.0000060991204414869,0.0000027991204414869,0.00000249001204414869
	   ]


path = 'D:\\All_Datasets\\Datasets_Norm\\'
index = 0
for root, dirs, files in os.walk(path):
	for i in range(len(files)):
		if (len(files[i])==16):
			file_path = root + '\\' + files[i]
			print('file_path:', file_path)
			emg = np.loadtxt(file_path,delimiter=',',skiprows=0)
			rest = np.loadtxt(file_path[:-5]+'rest.txt',delimiter=',',skiprows=0)
			E, x, y, raw_x, raw_y = Active_Segment_Detection(data=emg,
															 window_size=64,
															 step_size=32,
															 th1=th1[index],
															 th2=th2[index],
															 channel=0
															 )
			# Show_active_segment(file_path, th1=th1[index], th2=th2[index])
			index +=1
			Save_and_Filtering_Active_segment(list_x=raw_x,list_y=y,path=file_path,data=emg,rest=rest)



