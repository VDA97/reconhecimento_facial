# registro/views.py
import cv2
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.http import StreamingHttpResponse, JsonResponse
from registro.forms import FuncionarioForm, ColetaFacesForm
from registro.models import Funcionario, ColetaFaces
from registro.camera import VideoCamera
from datetime import datetime # Importa o módulo datetime
from django.utils import timezone

# Instância da classe VideoCamera
camera_detection = VideoCamera()


# --- Views para Detecção e Coleta de Faces ---

def gen_detect_face(camera_detection):
    """Gerador para o streaming de detecção facial."""
    # Inicia a câmera
    camera_detection.start_camera()
    while camera_detection.is_streaming:
        frame = camera_detection.detect_face()
        if frame is None:
            # Para o loop se o frame for None
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    # Para a câmera ao sair do loop
    camera_detection.stop_camera()


def face_detection(request):
    """View para exibir o streaming de detecção facial."""
    return StreamingHttpResponse(gen_detect_face(camera_detection),
                                 content_type='multipart/x-mixed-replace; \
                                     boundary=frame')


def criar_funcionario(request):
    """Cria um novo funcionário e o redireciona para a coleta de faces."""
    if request.method == 'POST':
        form = FuncionarioForm(request.POST, request.FILES)
        if form.is_valid():
            funcionario = form.save()
            return redirect(f'/criar_coleta_faces/{funcionario.id}?passo=1')
    else:
        form = FuncionarioForm()
    return render(request, 'criar_funcionario.html', {'form': form})


def extract(camera_detection, funcionario_slug):
    """
    Função para extrair e retornar o file_path das faces.
    Utiliza os novos métodos start_camera e stop_camera.
    """
    amostra = 0
    numeroAmostras = 30
    largura, altura = 220, 220
    file_paths = []

    camera_detection.start_camera()  # Inicia a câmera para a extração
    while amostra < numeroAmostras:
        crop = camera_detection.sample_faces()
        if crop is not None:
            amostra += 1
            face = cv2.resize(crop, (largura, altura))
            imagemCinza = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = f'./tmp/{funcionario_slug}_{amostra}.jpg'
            cv2.imwrite(file_name_path, imagemCinza)
            file_paths.append(file_name_path)
        else:
            print("Face não encontrada")

        if amostra >= numeroAmostras:
            break

    camera_detection.stop_camera()  # Para a câmera ao terminar
    return file_paths


def face_extract(context, funcionario):
    """
    Função para gerenciar o processo de extração e salvamento das faces.
    """
    num_coletas = ColetaFaces.objects.filter(funcionario__slug=funcionario.slug).count()
    print(num_coletas)

    if num_coletas >= 90:
        context['erro'] = 'Limite máximo de coletas atingido.'
    else:
        files_paths = extract(camera_detection, funcionario.slug)
        print(files_paths)

        for path in files_paths:
            coleta_face = ColetaFaces.objects.create(funcionario=funcionario)
            coleta_face.image.save(os.path.basename(path), open(path, 'rb'))
            os.remove(path)

        context['file_paths'] = ColetaFaces.objects.filter(
            funcionario__slug=funcionario.slug)
        context['extracao_ok'] = True

    return context


def criar_coleta_faces(request, funcionario_id):
    """View para o fluxo de coleta de fotos do funcionário."""
    passo = int(request.GET.get('passo', 1))
    extracao_ok = request.GET.get('extracao_ok', 'False') == 'True'
    mapa_imagens = {1: 'centro.png', 2: 'direita.png', 3: 'esquerda.png'}
    instrucao_imagem = mapa_imagens.get(passo, 'centro.png')

    try:
        funcionario = Funcionario.objects.get(id=funcionario_id)
    except Funcionario.DoesNotExist:
        return redirect('url_da_pagina_inicial')

    if request.method == 'GET' and request.GET.get('clicked') == 'True':
        print(f"Iniciando extração de faces no passo {passo}...")
        face_extract({}, funcionario)
        return redirect(f'/criar_coleta_faces/{funcionario.id}?extracao_ok=True&passo={passo}')

    context = {
        'funcionario': funcionario,
        'passo': passo,
        'extracao_ok': extracao_ok,
        'instrucao_imagem': instrucao_imagem,
    }

    if extracao_ok:
        context['file_paths'] = ColetaFaces.objects.filter(
            funcionario=funcionario
        ).order_by('-id')[:30]

    return render(request, 'criar_coleta_faces.html', context)


# --- Novas Views para Reconhecimento e Redirecionamento ---

def face_recognition(request):
    """
    View que apenas renderiza o template do reconhecimento.
    A lógica de streaming e checagem fica em outras views.
    """
    return render(request, 'face_recognition.html')


def face_recognition_stream(request):
    """
    View que retorna o StreamingHttpResponse para o frontend.
    Responsável apenas por gerar o fluxo de vídeo com as detecções.
    """

    def gen_recognize_face_stream():
        # Inicia a câmera para o streaming
        camera_detection.start_camera()
        while camera_detection.is_streaming:
            # A função recognize_face retorna o frame e o ID, mas aqui só usamos o frame
            frame, _ = camera_detection.recognize_face()
            if frame is None:
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        camera_detection.stop_camera()  # Para a câmera ao sair do loop

    return StreamingHttpResponse(gen_recognize_face_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def face_recognition_check(request):
    """
    View chamada via AJAX para verificar se um rosto foi reconhecido.
    Ela para a câmera e retorna o ID do funcionário, se houver sucesso.
    """
    # A chamada para `recognize_face` faz a detecção e o reconhecimento
    frame, funcionario_id = camera_detection.recognize_face()

    if funcionario_id:
        # Se o reconhecimento foi bem-sucedido, para a câmera e retorna o ID
        camera_detection.stop_camera()
        return JsonResponse({
            'status': 'success',
            'funcionario_id': funcionario_id
        })

    # Se nenhum rosto foi reconhecido ou o modelo não está carregado
    return JsonResponse({
        'status': 'waiting',
        'message': 'Nenhum rosto reconhecido ainda.'
    })

def usuario_reconhecido(request, funcionario_id):
    """
    View que exibe a página com os dados do usuário reconhecido.
    A data e hora são agora tratadas com reconhecimento de fuso horário.
    """
    funcionario = get_object_or_404(Funcionario, pk=funcionario_id)

    recognized_at_str = request.GET.get('recognized_at')
    recognized_at = None
    if recognized_at_str:
        try:
            # 1. Limpa a string removendo o 'Z' e milissegundos para usar fromisoformat()
            cleaned_str = recognized_at_str.replace('Z', '')

            # Se a string contiver milissegundos (ex: .123), a gente remove
            if '.' in cleaned_str:
                cleaned_str = cleaned_str.split('.')[0]

            # Cria um objeto datetime ingênuo a partir da string UTC
            dt_naive = datetime.fromisoformat(cleaned_str)

            # 2. Torna o objeto datetime timezone-aware, definindo-o explicitamente como UTC
            # O Django automaticamente o converterá para 'America/Sao_Paulo' na hora de renderizar no template.
            recognized_at = timezone.make_aware(dt_naive, timezone.utc)

        except ValueError:
            print(f"Erro ao converter a string de data: {recognized_at_str}")

    context = {
        'funcionario': funcionario,
        'recognized_at': recognized_at,
    }

    return render(request, 'usuario_reconhecido.html', context)
