from django import forms
from .models import Funcionario, ColetaFaces

#ModelForm é do import forms.
class FuncionarioForm(forms.ModelForm):
    class Meta:
        model = Funcionario
        fields = ['foto', 'nome', 'cpf']
        #Esses campos foto, nome e cpf serão mostrados quando renderizar o modelo funcionario
        #O campo slug não entra em questão pois é um campo que será gerado automaticamente.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Pra cada campo do Modelo Funcionario, adiciona a classe do bootstrap "form-control" para padronizar o layout
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'

# DOC: https://docs.djangoproject.com/en/5.1/topics/http/file-uploads/#uploading-multiple-files
# Multiplos Arquivos

#O Django possui estrutura para trabalhar com multiplos arquivos.
class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result

class ColetaFacesForm(forms.ModelForm):
    images = MultipleFileField()
    
    class Meta:
        model = ColetaFaces
        fields = ['images']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'