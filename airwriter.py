#rishika airwriter.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import random
import os

w = 640
h = 480

PALETTE = {'Blue':(255,0,0),'Green':(0,255,0),'Red':(0,0,255),'Yellow':(0,255,255),'Eraser':(255,255,255)}

cur_col = PALETTE['Blue']
brush_th = 5
eraser_th = 50

strokes = []
redo_buf = []
pts = deque(maxlen=1024)

buttons = [
 {'name':'CLEAR','x':20,'y':10,'w':60,'h':50,'color':(0,0,0)},
 {'name':'Blue','x':90,'y':10,'w':50,'h':50,'color':PALETTE['Blue']},
 {'name':'Green','x':150,'y':10,'w':50,'h':50,'color':PALETTE['Green']},
 {'name':'Red','x':210,'y':10,'w':50,'h':50,'color':PALETTE['Red']},
 {'name':'Yellow','x':270,'y':10,'w':50,'h':50,'color':PALETTE['Yellow']},
 {'name':'Eraser','x':330,'y':10,'w':70,'h':50,'color':(100,100,100)},
 {'name':'Undo','x':410,'y':10,'w':60,'h':50,'color':(50,50,50)},
 {'name':'Redo','x':480,'y':10,'w':60,'h':50,'color':(50,50,50)},
 {'name':'SaveImage','x':550,'y':10,'w':60,'h':50,'color':(0,100,255)}
]

last_click = 0
cool = 0.5

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

cap = cv2.VideoCapture(0)
canvas = np.ones((h,w,3), dtype=np.uint8)*255

def draw_ui(img):
    cv2.rectangle(img,(0,0),(w,70),(220,220,220),-1)
    for b in buttons:
        x=b['x']; y=b['y']; ww=b['w']; hh=b['h']
        cv2.rectangle(img,(x,y),(x+ww,y+hh),b['color'],-1)
        cv2.rectangle(img,(x,y),(x+ww,y+hh),(255,255,255),2)
        cv2.putText(img,b['name'],(x+5,y+30),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)

def save_canvas(c):
    fn = "drawing_"+str(int(time.time()))+".png"
    path = fn
    cv2.imwrite(path,c)
    print("Saved to", path)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    feed = frame.copy()

    canvas[:] = 255

    for s in strokes:
        pl = s['points']
        col = s['color']
        th = eraser_th if col==PALETTE['Eraser'] else brush_th
        if len(pl)>1:
            for i in range(1,len(pl)):
                a=pl[i-1]; b=pl[i]
                if a is None or b is None: continue
                cv2.line(canvas, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), col, th)

    if len(pts)>=2:
        tlist = list(pts)
        thick = eraser_th if cur_col==PALETTE['Eraser'] else brush_th
        for i in range(1,len(tlist)):
            p1=tlist[i-1]; p2=tlist[i]
            if p1 is None or p2 is None: continue
            cv2.line(canvas, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), cur_col, thick)

    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        for lmset in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(feed, lmset, mp_hands.HAND_CONNECTIONS)

        L = res.multi_hand_landmarks[0].landmark
        fh,fw,_ = frame.shape
        ix = int(L[8].x * fw); iy = int(L[8].y * fh)
        mx = int(L[12].x * fw); my = int(L[12].y * fh)

        dist = ((ix-mx)**2 + (iy-my)**2)**0.5

        if dist < 40:
            if len(pts)>0:
                strokes.append({'color':cur_col,'points':list(pts)})
                pts.clear()
                redo_buf.clear()

            cv2.circle(frame,(ix,iy),20,(100,100,100),2)

            if iy < 70 and (time.time()-last_click) > cool:
                for btn in buttons:
                    if btn['x'] < ix < btn['x']+btn['w']:
                        last_click = time.time()
                        n = btn['name'].lower()
                        if n == 'clear':
                            strokes=[]; redo_buf=[]
                        elif n == 'undo':
                            if strokes:
                                redo_buf.append(strokes.pop())
                        elif n == 'redo':
                            if redo_buf:
                                strokes.append(redo_buf.pop())
                        elif n in ('saveimage','save'):
                            save_canvas(canvas)
                            cv2.putText(frame,"SAVED!",(300,240),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                            cv2.waitKey(200)
                        elif n == 'eraser':
                            cur_col = PALETTE['Eraser']
                        else:
                            for k in PALETTE:
                                if k.lower()==n:
                                    cur_col = PALETTE[k]
                                    break
        else:
            if iy > 70:
                p=(ix,iy)
                pts.append(p)
                cv2.circle(frame,(ix,iy),10,cur_col,-1)
    else:
        if len(pts)>0:
            strokes.append({'color':cur_col,'points':list(pts)})
            pts.clear()
            redo_buf.clear()

    draw_ui(frame)

    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _,inv = cv2.threshold(g,240,255,cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(inv))
    frame = cv2.bitwise_or(frame, cv2.bitwise_and(canvas, inv))

    cv2.imshow("User Feed (Me)", feed)
    cv2.imshow("Virtual Whiteboard (Board)", canvas)
    cv2.imshow("Control Panel (UI)", frame)

    try:
        cv2.moveWindow("User Feed (Me)",0,0)
        cv2.moveWindow("Control Panel (UI)",650,0)
        cv2.moveWindow("Virtual Whiteboard (Board)",300,500)
    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
