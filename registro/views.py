import cv2
import os
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import StreamingHttpResponse
from registro.forms import FuncionarioForm, ColetaFacesForm
from registro.models import Funcionario, ColetaFaces
from registro.camera import VideoCamera

# Instância da classe VideoCamera
camera_detection = VideoCamera()


# Captura o frame com face detectada
def gen_detect_face(camera_detection):
    while True:
        frame = camera_detection.detect_face()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Cria streaming para detecção facial
def face_detection(request):
    return StreamingHttpResponse(gen_detect_face(camera_detection),
                                 content_type='multipart/x-mixed-replace; \
                                     boundary=frame')


# Cria um novo funcionário e o redireciona para a coleta de faces
def criar_funcionario(request):
    if request.method == 'POST':
        form = FuncionarioForm(request.POST, request.FILES)
        if form.is_valid():
            funcionario = form.save()
            # Redireciona para o primeiro passo do fluxo de coleta de faces
            return redirect(f'/criar_coleta_faces/{funcionario.id}?passo=1')
    else:
        form = FuncionarioForm()
    return render(request, 'criar_funcionario.html', {'form': form})


# Cria uma função para extrair e retornar o file_path
def extract(camera_detection, funcionario_slug):
    amostra = 0
    numeroAmostras = 30
    largura, altura = 220, 220
    file_paths = []

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

    camera_detection.restart()
    return file_paths


def face_extract(context, funcionario):
    num_coletas = ColetaFaces.objects.filter(
        funcionario__slug=funcionario.slug).count()

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
    # Obtém o passo da URL. Se não existir, define como 1.
    passo = int(request.GET.get('passo', 1))

    # A variável 'extracao_ok' verifica se as fotos foram tiradas com sucesso
    extracao_ok = request.GET.get('extracao_ok', 'False') == 'True'

    # Mapeia cada passo a uma imagem de instrução
    mapa_imagens = {
        1: 'centro.png',
        2: 'direita.png',
        3: 'esquerda.png',
    }
    instrucao_imagem = mapa_imagens.get(passo, 'centro.png')

    # Resgata o funcionário, necessário para a lógica e o template
    try:
        funcionario = Funcionario.objects.get(id=funcionario_id)
    except Funcionario.DoesNotExist:
        # Redireciona para uma página inicial se o funcionário não for encontrado
        return redirect('url_da_pagina_inicial')

    # Lógica principal: o que acontece quando o botão "Tirar Fotos" é clicado
    if request.method == 'GET' and request.GET.get('clicked') == 'True':
        print(f"Iniciando extração de faces no passo {passo}...")

        # Chama sua função de extração de faces
        face_extract({}, funcionario)

        # Redireciona para a mesma página, mas com o estado atualizado
        # 'extracao_ok=True' fará com que as fotos e o botão "CONTINUAR" apareçam
        return redirect(f'/criar_coleta_faces/{funcionario.id}?extracao_ok=True&passo={passo}')

    # Prepara o contexto para renderizar a página
    context = {
        'funcionario': funcionario,
        'passo': passo,
        'extracao_ok': extracao_ok,
        'instrucao_imagem': instrucao_imagem,
    }

    # Se a extração foi bem-sucedida, carregue as últimas fotos tiradas
    if extracao_ok:
        context['file_paths'] = ColetaFaces.objects.filter(
            funcionario=funcionario
        ).order_by('-id')[:30]

    return render(request, 'criar_coleta_faces.html', context)