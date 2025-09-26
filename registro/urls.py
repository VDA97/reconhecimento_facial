from django.urls import path
from registro.views import (
    criar_funcionario,
    criar_coleta_faces,
    face_detection,
    face_recognition,
    face_recognition_stream,  # Importa a nova view de streaming
    face_recognition_check,  # Importa a nova view de checagem AJAX
    usuario_reconhecido  # Importa a nova view de sucesso
)

urlpatterns = [
    # URL principal para criação de um novo funcionário
    path('', criar_funcionario, name='criar_funcionario'),

    # URL para a coleta de faces (fluxo em 3 passos)
    path('criar_coleta_faces/<int:funcionario_id>/', criar_coleta_faces, name='criar_coleta_faces'),

    # URL para o streaming de detecção facial (sem reconhecimento)
    path('face_detection/', face_detection, name='face_detection'),

    # URL principal para o reconhecimento facial (renderiza o template)
    path('face_recognition/', face_recognition, name='face_recognition'),

    # Novas URLs para o fluxo de reconhecimento
    # Esta URL fornece o streaming de vídeo para a tag <img> no template
    path('face_recognition_stream/', face_recognition_stream, name='face_recognition_stream'),

    # Esta URL é chamada via JavaScript para verificar o status do reconhecimento
    path('face_recognition_check/', face_recognition_check, name='face_recognition_check'),

    # Esta URL exibe a página de sucesso com os dados do usuário reconhecido
    path('usuario_reconhecido/<int:funcionario_id>/', usuario_reconhecido, name='usuario_reconhecido')
]