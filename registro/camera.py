# registro/camera.py
import cv2
import os
import threading  # Importa o módulo de threading para usar locks
from django.conf import settings
from registro.models import Funcionario, Treinamento


class VideoCamera(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.video = None  # Inicializa a câmera como None
        self.is_streaming = False  # Flag para controlar o estado do streaming
        self.camera_lock = threading.Lock()  # Lock para acesso seguro à câmera

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
        self.stop_camera()

    def start_camera(self):
        """Inicializa a câmera se ela ainda não estiver aberta."""
        with self.camera_lock:
            if not self.video or not self.video.isOpened():
                self.video = cv2.VideoCapture(0)
                if not self.video.isOpened():
                    print("Erro ao iniciar a câmera.")
                    self.video = None
                    return False
            self.is_streaming = True
            return True

    def stop_camera(self):
        """Libera o recurso da câmera de forma segura."""
        with self.camera_lock:
            if self.video and self.video.isOpened():
                self.video.release()
            self.video = None
            self.is_streaming = False

    def get_frame(self, retries=3):
        """
        Lê um frame da câmera de forma segura.
        Retorna o frame e um booleano indicando o sucesso da leitura.
        """
        with self.camera_lock:
            if not self.video or not self.video.isOpened():
                return None, False

            for _ in range(retries):
                ret, frame = self.video.read()
                if ret and frame is not None:
                    return frame, True
        return None, False

    def recognize_face(self):
        """
        Lê um frame, realiza a detecção e o reconhecimento facial.
        Retorna o frame em bytes para streaming e o ID do funcionário se
        houver sucesso, caso contrário, retorna (None, None).
        """
        frame, success = self.get_frame()
        if not success:
            return None, None

        frame = cv2.resize(frame, (480, 360))
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Copia o frame antes de desenhar para a lógica de reconhecimento
        frame_copy = frame.copy()

        faces_detectadas = self.face_cascade.detectMultiScale(
            imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(400, 400))

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        conf_text = "Nenhum rosto detectado"
        text_color = (0, 0, 255)
        funcionario_id = None  # Inicializa com None

        face_x, face_y, face_w, face_h = 0, 0, 0, 0

        for (x, y, l, a) in faces_detectadas:
            face_x, face_y, face_w, face_h = x, y, l, a

            imagemFace = imagemCinza[y:y + a, x:x + l]
            imagemFace = cv2.resize(imagemFace, (220, 220))
            imagemFace = cv2.equalizeHist(imagemFace)
            imagemFace = cv2.normalize(imagemFace, None, 0, 255, cv2.NORM_MINMAX)

            cv2.rectangle(frame_copy, (x, y), (x + l, y + a), (0, 255, 0), 2)

            if self.model_loaded:
                label, confidence = self.recognizer.predict(imagemFace)

                if confidence < 8000:
                    try:
                        funcionario = Funcionario.objects.get(id=label)
                        conf_text = f"{funcionario.nome} ({int(confidence)})"
                        text_color = (0, 255, 0)
                        funcionario_id = funcionario.id  # Retorna o ID
                    except Funcionario.DoesNotExist:
                        conf_text = "Desconhecido"
                        text_color = (0, 0, 255)
                else:
                    conf_text = "Baixa confiabilidade"
                    text_color = (0, 0, 255)
            else:
                conf_text = "Modelo nao carregado"
                text_color = (0, 0, 255)

        frame_copy = cv2.flip(frame_copy, 1)

        if len(faces_detectadas) > 0:
            final_pos_x = frame_copy.shape[1] - (face_x + face_w)
            final_pos_y = face_y + face_h + 30
            cv2.putText(frame_copy, conf_text, (final_pos_x, final_pos_y), font, 1, text_color, 2)
        else:
            cv2.putText(frame_copy, conf_text, (10, 30), font, 1, text_color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame_copy)
        return jpeg.tobytes(), funcionario_id

    def detect_face(self):
        """
        Retorna apenas o frame com a detecção visual da face, sem reconhecimento.
        """
        frame, success = self.get_frame()
        if not success:
            return None

        altura, largura, _ = frame.shape
        centro_x, centro_y = int(largura / 2), int(altura / 2)
        a, b = 140, 180
        x1, y1 = centro_x - a, centro_y - b
        x2, y2 = centro_x + a, centro_y + b

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        cv2.ellipse(frame, (centro_x, centro_y), (a, b), 0, 0, 360, (0, 0, 255), 10)

        for (x, y, w, h) in faces:
            cv2.ellipse(frame, (centro_x, centro_y), (a, b), 0, 0, 360, (0, 255, 0), 10)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def sample_faces(self):
        """
        Captura frames da face para coleta de dados.
        """
        frame, success = self.get_frame()
        if not success:
            return None

        frame = cv2.flip(frame, 180)
        frame = cv2.resize(frame, (480, 360))

        faces = self.face_cascade.detectMultiScale(
            frame, minNeighbors=20, minSize=(30, 30), maxSize=(400, 400))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cropped_face = frame[y:y + h, x:x + w]
            return cropped_face