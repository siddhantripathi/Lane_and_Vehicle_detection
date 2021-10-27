import numpy as np
import cv2


def object_detection(frame,labels,COLORS,net,outputLayer):
    writer = None
    confidenceThreshold = 0.3
    NMSThreshold = 0.2

    

    (W, H) = (None, None)
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    count = 0
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(outputLayer)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    count=0
    if(len(detectionNMS) > 0):
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            #if labels[classIDs[i]] =='person':
            cv2.putText(frame, labels[classIDs[i]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            count+=1

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter('chase_output.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    frame=cv2.putText(frame, 'Total Object count = '+str(count), (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
    return frame
