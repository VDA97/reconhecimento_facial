from django.shortcuts import render, redirect
from .forms import FuncionarioForm, ColetaFacesForm
from .models import Funcionario, ColetaFaces

#Dados serão enviados através do evento POST
#Verifica se o formulário é valido, condições para verificar se o funcionário é valido
#Salva o formulário do funcionario
#Pega a instância funcionário e redireciona para a rota criar-coleta_faces
def criar_funcionario(request):
    if request.method == 'POST':
        form = FuncionarioForm(request.POST, request.FILES)
        if form.is_valid():
            funcionario = form.save()
            return redirect('criar_coleta_faces', funcionario_id=funcionario.id)
    else:
        form = FuncionarioForm()
        return render(request, 'criar_funcionario.html', {'form': form})

def criar_coleta_faces(request, funcionario_id):
    funcionario = Funcionario.objects.get(id=funcionario_id)
    if request.method == 'POST':
        form = ColetaFacesForm(request.POST, request.FILES)
        if form.is_valid():
            # Itera sobre os arquivos enviados e cria uma entrada para cada imagem
            for image in request.FILES.getlist('images'):
                ColetaFaces.objects.create(funcionario=funcionario, image=image)
    else:
        form = ColetaFacesForm()

    context = {
        'funcionario': funcionario,
        'form': form
    }

    return render(request, 'criar_coleta_faces.html', context)