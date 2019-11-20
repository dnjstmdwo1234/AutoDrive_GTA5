
def control_car(num):
    if(num==1):
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        print("직진")
        
    elif(num==2):
        PressKey(A)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(A)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        print("좌회전")
        

    elif(num==3):
        PressKey(D)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(D)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        print("우회전")
     
    elif(num==5):
        ReleaseKey(D)
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        time.sleep(1)
        print("충돌위험으로 제동")
        
    else:
        PressKey(S)
        time.sleep(3)
        ReleaseKey(S)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        
        print("후진")