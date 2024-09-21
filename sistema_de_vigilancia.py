from ultralytics import YOLO
import cv2 as cv

# importando o modelo yolov8n.pt (nano)
modelo = YOLO('./classificadores/yolov8n.pt')
video = cv.VideoCapture('./videos/aeroporto.mp4')

# faz um looping nos frames do vídeo
while video.isOpened():
    rect, frame = video.read()
    if rect:
        # reduzindo o tamanho da janela
        frame_reduzido = cv.resize(frame, (900, 600), interpolation=cv.INTER_LINEAR)

        # realizando a deteção de pessoas
        previsao = modelo(frame_reduzido, verbose=False, classes=[0])
        caixas = previsao[0].boxes
        
        # desenhando os contornos
        for caixa in caixas:
            left, top, right, bottom = caixa.xyxy[0].int().tolist()
            cv.rectangle(frame_reduzido, (left, top), (right, bottom), (255, 0, 0), 2)

        # apresentando os frames
        cv.imshow('Sistema de Vigilancia', frame_reduzido)

        # pressione "q" para fechar
        if cv.waitKey(90) & 0xFF == ord('q'):
            break
    else:
        break

# fechando todas as janelas
video.release()
cv.destroyAllWindows()