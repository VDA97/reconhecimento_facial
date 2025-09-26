# registro/camera.py
import cv2
import os
from django.conf import settings
from registro.models import Funcionario, Treinamento

class VideoCamera(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Erro ao acessar a câmera.")
            self.video = None

        self.img_dir = "./tmp"
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # --- Lógica de reconhecimento ---
        self.recognizer = cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)
        self.model_loaded = False
        try:
            treinamento = Treinamento.objects.first()
            if treinamento:
                model_path = os.path.join(settings.MEDIA_ROOT, treinamento.modelo.name)
                if os.path.exists(model_path):
                    self.recognizer.read(model_path)
                    self.model_loaded = True
                    print("Modelo de reconhecimento facial carregado com sucesso.")
                else:
                    print("Caminho do modelo não encontrado.")
            else:
                print("Modelo de treinamento não encontrado no banco de dados.")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")

    def __del__(self):
        if self.video and self.video.isOpened():
            self.video.release()

    def restart(self):
        if self.video and self.video.isOpened():
            self.video.release()
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Erro ao reiniciar a câmera.")
            self.video = None

    def get_camera(self, retries=3):
        if not self.video or not self.video.isOpened():
            return False, None

        for _ in range(retries):
            ret, frame = self.video.read()
            if ret and frame is not None:
                return ret, frame
        return False, None

    def recognize_face(self):
        ret, frame = self.get_camera()
        if not ret:
            return None

        # PASSO 1: Redimensiona o frame para as mesmas dimensões do script de teste
        frame = cv2.resize(frame, (480, 360))

        # PASSO 2: Converte para tons de cinza para a detecção de faces
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # PASSO 3: Detecta as faces
        faces_detectadas = self.face_cascade.detectMultiScale(
            imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(400, 400))

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        conf_text = "Nenhum rosto detectado"
        text_color = (0, 0, 255)

        # Variáveis de posição que só serão usadas se uma face for detectada
        face_x, face_y, face_w, face_h = 0, 0, 0, 0

        for (x, y, l, a) in faces_detectadas:
            # Armazena as coordenadas da face para o texto
            face_x, face_y, face_w, face_h = x, y, l, a

            # Recorta a imagem do rosto
            imagemFace = imagemCinza[y:y + a, x:x + l]

            # Redimensiona para o tamanho de treinamento
            imagemFace = cv2.resize(imagemFace, (220, 220))

            # Aplica o pré-processamento igual ao do treinamento
            imagemFace = cv2.equalizeHist(imagemFace)
            imagemFace = cv2.normalize(imagemFace, None, 0, 255, cv2.NORM_MINMAX)

            # Desenha o retângulo no frame original (colorido)
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)

            if self.model_loaded:
                label, confidence = self.recognizer.predict(imagemFace)
                print(f"O valor de confiança do reconhecimento é: {confidence}, Label: {label}")

                if confidence < 8000:
                    try:
                        funcionario = Funcionario.objects.get(id=label)
                        nome = str(funcionario.nome).strip("(),'")
                        conf_text = f"{nome} ({int(confidence)})"
                        text_color = (0, 255, 0)
                    except Funcionario.DoesNotExist:
                        conf_text = "Desconhecido"
                        text_color = (0, 0, 255)
                else:
                    conf_text = "Baixa confiabilidade"
                    text_color = (0, 0, 255)
            else:
                conf_text = "Modelo nao carregado"
                text_color = (0, 0, 255)

        # PASSO 4: Inverte o frame (espelhamento)
        frame = cv2.flip(frame, 1)

        # PASSO 5: Desenha o texto no frame espelhado
        if len(faces_detectadas) > 0:
            # Posição do texto próxima ao rosto, após o espelhamento
            final_pos_x = frame.shape[1] - (face_x + face_w)
            final_pos_y = face_y + face_h + 30
            cv2.putText(frame, conf_text, (final_pos_x, final_pos_y), font, 1, text_color, 2)
        else:
            # Se não houver rosto, desenha no canto superior esquerdo
            cv2.putText(frame, conf_text, (10, 30), font, 1, text_color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def detect_face(self):
        ret, frame = self.get_camera()  # Leitura do frame
        if not ret:  # Verifica se a captura do frame foi bem-sucedida
            print("detect_face error: get_camera failed")
            return None

        # Defina a região de interesse (ROI) onde o rosto será detectado
        altura, largura, _ = frame.shape
        centro_x, centro_y = int(largura / 2), int(altura / 2)
        a, b = 140, 180
        x1, y1 = centro_x - a, centro_y - b
        x2, y2 = centro_x + a, centro_y + b
        roi = frame[y1:y2, x1:x2]

        # Converta a ROI em escala de cinza para a detecção de faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces no frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        cv2.ellipse(frame, (centro_x, centro_y), (a, b), 0, 0, 360, (0, 0, 255), 10)

        for (x, y, w, h) in faces:
            cv2.ellipse(frame, (centro_x, centro_y), (a, b), 0, 0, 360, (0, 255, 0), 10)

        # Retorna o frame com a face detectada como imagem JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()  # Converte o frame para formato JPEG em bytes

    def sample_faces(self):
        ret, frame = self.get_camera()
        if not ret:  # Verifica se a captura do frame foi bem-sucedida
            print("sample_faces error : get_camera failed")
            return None
        frame = cv2.flip(frame, 180)
        frame = cv2.resize(frame, (480, 360))

        # Detecta a face
        faces = self.face_cascade.detectMultiScale(
            frame, minNeighbors=20, minSize=(30, 30), maxSize=(400, 400))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cropped_face = frame[y:y + h, x:x + w]
            return cropped_face  # Retorna o rosto recortado