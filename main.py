from multiprocessing import Process
from multiprocessing import Pipe
from Drowsiness_detection.test2_yawn import drowzy
from lane_detection.laneDetection import lane
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('Alert.wav')
sound1 = mixer.Sound('Warning.wav')


if __name__ == '__main__':
    p_con, ch_con = Pipe()
    p1_con, ch1_con = Pipe()
    p = Process(target=drowzy, args=(ch1_con,))
    p1 = Process(target=lane,args=(ch_con,))
    p1.start()
    p.start()
   
    while(1):
        steer = p_con.recv()
        fatigue = p1_con.recv()
        s1 = 0
        f1 = 0
        S_flag = 0
        s2 = ''
        Close_flag = 0
        yawn_flag = 0
        bp_flag = 0

        #Conditions for steer
        if(steer[1] == 'right' or steer[1] == 'left'):
            if( s2 == 'left' or s2 == 'right'):
                S_flag = 1
        
        if((steer[0] - s1) > 2.0 or (steer[0] - s1) < -2.0):
            S_flag = 1
        
        #Conditions for drowzy
        if(fatigue[0] > 15):
            Close_flag = 1


        if(fatigue[1] > 5):
            yawn_flag = 1
        
        if((fatigue[2] - f1) > 50 or (fatigue[2] - f1) < 0 ):
            bp_flag = 1
        elif(fatigue[2] < 50 or fatigue[2] > 100):
            bp_flag = 1

        #Alerting the driver

        if(S_flag == 1 and bp_flag == 1):
            sound.play()
            print("Health problem")
        elif(S_flag == 1 and Close_flag == 1):
            sound.play()
            print("Driver has fainted")
        elif(Close_flag == 1 or yawn_flag == 1):
            sound1.play()
            print("Driver is tired")
            time.sleep(10)
        elif(Close_flag and bp_flag):
            sound.play()
            print("Health issue")
            time.sleep(3)


        s1 = steer[0]
        s2 = steer[1]
        f1 = fatigue[2]
        print(steer)
        print(fatigue)
    p.join()
    p1.join()
