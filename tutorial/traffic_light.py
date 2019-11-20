def traffic_light_red(screen):
    num = 0
    b, g, r    = screen[:, :, 0], screen[:, :, 1], screen[:, :, 2]
    shape = b.shape
    shape1 = shape[0]*shape[1]
    b = b.reshape(shape1)
    shape = r.shape
    shape1 = shape[0]*shape[1]
    r = r.reshape(shape1)
    shape = g.shape
    shape1 = shape[0]*shape[1]
    g = g.reshape(shape1)
    
    for i in range(0,100):
        if(b[i] > 180 & g[i] < 40 & r[i] < 40):
            num += 1
    
    
    if num >= 10:
        print("빨간불 입니다.")
        control_car(5)