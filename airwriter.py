import cv2
import numpy as np
import mediapipe as mp
import os
import time
from collections import deque

widow_widt = 640
widow_hite = 480
colors = {
    'BLue': (255, 0, 0),
    'GreeN': (0, 255, 0),
    'ReD': (0, 0, 255),
    'Yellow': (0, 255, 255),
    'Erase': (255, 255, 255)
}
current_color = colors['BLue']
brush_thickness = 5
eraser_thickness = 50
strokes = [] 
undo_stack = [] 
current_stroke_points = deque(maxlen=1024) 

buttons = [
    {'name': 'CLEAR', 'x': 20, 'y': 10, 'w': 60, 'h': 50, 'color': (0, 0, 0)},
    {'name': 'Blue', 'x': 90, 'y': 10, 'w': 50, 'h': 50, 'color': colors['BLue']},
    {'name': 'Green', 'x': 150, 'y': 10, 'w': 50, 'h': 50, 'color': colors['GreeN']},
    {'name': 'Red', 'x': 210, 'y': 10, 'w': 50, 'h': 50, 'color': colors['ReD']},
    {'name': 'Yellow', 'x': 270, 'y': 10, 'w': 50, 'h': 50, 'color': colors['Yellow']},
    {'name': 'Eraser', 'x': 330, 'y': 10, 'w': 70, 'h': 50, 'color': (100, 100, 100)},
    {'name': 'Undo', 'x': 410, 'y': 10, 'w': 60, 'h': 50, 'color': (50, 50, 50)},
    {'name': 'Redo', 'x': 480, 'y': 10, 'w': 60, 'h': 50, 'color': (50, 50, 50)},
    {'name': 'SaveImage', 'x': 550, 'y': 10, 'w': 60, 'h': 50, 'color': (0, 100, 255)},
]

last_click_time = 0
click_cooldown = 0.5 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils 
'''the above is for me box'''
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 255

def draw_ui(img):
    cv2.rectangle(img, (0, 0), (640, 70), (220, 220, 220), -1)
    for btn in buttons:
        cv2.rectangle(img, (btn['x'], btn['y']), (btn['x']+btn['w'], btn['y']+btn['h']), btn['color'], -1)
        cv2.rectangle(img, (btn['x'], btn['y']), (btn['x']+btn['w'], btn['y']+btn['h']), (255, 255, 255), 2)
        font_scale = 0.4
        if btn['name'] in ['CLEAR', 'ERASER', 'UNDO', 'REDO', 'SAVE']:
            cv2.putText(img, btn['name'], (btn['x'] + 5, btn['y'] + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

def save_canvas(img):
    filename = f"drawing_{int(time.time())}.png"
    cv2.imwrite(filename, img)
    print(f"Saved to {filename}")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    user_feed = frame.copy()
    
    canvas[:] = 255 
    
    for stroke in strokes:
        points_list = stroke['points']
        col = stroke['color']
        thick = eraser_thickness if col == colors['ERASER'] else brush_thickness
        for i in range(1, len(points_list)):
            if points_list[i - 1] is None or points_list[i] is None: continue
            cv2.line(canvas, points_list[i - 1], points_list[i], col, thick)

    for i in range(1, len(current_stroke_points)):
        if current_stroke_points[i - 1] is None or current_stroke_points[i] is None: continue
        thick = eraser_thickness if current_color == colors['ERASER'] else brush_thickness
        cv2.line(canvas, current_stroke_points[i - 1], current_stroke_points[i], current_color, thick)

    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(user_feed, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        landmarks = result.multi_hand_landmarks[0].landmark
        h, w, c = frame.shape
        index_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))
        middle_tip = (int(landmarks[12].x * w), int(landmarks[12].y * h))
        
        distance = np.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])
        
        
        if distance < 40:
            if len(current_stroke_points) > 0:
                strokes.append({'color': current_color, 'points': list(current_stroke_points)})
                current_stroke_points.clear()
                undo_stack.clear()
            
            cv2.circle(frame, index_tip, 20, (100, 100, 100), 2) 
            
            if index_tip[1] < 70 and (time.time() - last_click_time) > click_cooldown:
                for btn in buttons:
                    if btn['x'] < index_tip[0] < btn['x'] + btn['w']:
                        last_click_time = time.time()
                        if btn['name'] == 'CLEAR':
                            strokes.clear()
                            undo_stack.clear()
                        elif btn['name'] == 'UNDO':
                            if len(strokes) > 0: undo_stack.append(strokes.pop())
                        elif btn['name'] == 'REDO':
                            if len(undo_stack) > 0: strokes.append(undo_stack.pop())
                        elif btn['name'] == 'SAVE':
                            save_canvas(canvas)
                            cv2.putText(frame, "SAVED!", (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                            cv2.waitKey(500)
                        elif btn['name'] == 'ERASER':
                            current_color = colors['ERASER']
                        elif btn['name'] in colors:
                            current_color = colors[btn['name']]
        
        
        else:
            if index_tip[1] > 70:
                current_stroke_points.append(index_tip)
                cv2.circle(frame, index_tip, 10, current_color, -1)
    else:
        if len(current_stroke_points) > 0:
            strokes.append({'color': current_color, 'points': list(current_stroke_points)})
            current_stroke_points.clear()
            undo_stack.clear()


    draw_ui(frame)
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(img_inv))
    frame = cv2.bitwise_or(frame, cv2.bitwise_and(canvas, img_inv))

    
    cv2.imshow("User Feed (Me)", user_feed)          
    cv2.imshow("Virtual Whiteboard (Board)", canvas)
    cv2.imshow("Control Panel (UI)", frame)         
    cv2.moveWindow("User Feed (Me)", 0, 0)
    cv2.moveWindow("Control Panel (UI)", 650, 0)
    cv2.moveWindow("Virtual Whiteboard (Board)", 300, 500)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()