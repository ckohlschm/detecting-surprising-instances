from django.shortcuts import render

from django.http import HttpResponseRedirect
from django.views.generic import FormView
from django.core.files.storage import FileSystemStorage
from .client import convert_log_to_features
import pandas as pd
from .client import discover_model_as_image, detect_surprising_instances_similarity_graph
from .backend.util import import_log_discover_features_and_model
from .backend.parameter_selection import process_parameters
from .backend.similarity_graph_method import apply_similarity_graph
from .backend.supervised_learning_method import apply_supervised_learning
from .backend.random_walk_method import apply_random_walk

def index(request):
    msg = None
    # clear session storage
    for key in list(request.session.keys()):
        del request.session[key]    
    if request.method == 'POST':
        print("Received method POST: " + str(request.FILES))
        uploaded_file_name = str(request.FILES['file'])
        if uploaded_file_name.endswith('xes') or uploaded_file_name.endswith('csv') or uploaded_file_name.endswith('xes.gz'):
            fs = FileSystemStorage()
            path = 'logs/' + str(request.FILES['file'])
            log_path = fs.save(path, request.FILES['file'])
            print("Successfully saved file to " + str(log_path))
            request.session['log_path'] = log_path
            return HttpResponseRedirect('/method_selection_similarity_graph')
        else:
            msg = "Please select a valid xes, xes.gz or csv file"
    return render(request, 'detection/index.html', {"msg": msg})

def not_found(request):
    return render(request, 'detection/page-404.html')

def method_selection_similarity_graph(request):
    context = {}
    context['selection_method'] = 'similarity_graph'
    if request.session.get('log_path', None):
        if request.method == 'POST':
            if 'supervised_learning' in request.POST:
                return HttpResponseRedirect('/method_selection_supervised_learning')
            if 'random_walk' in request.POST:
                return HttpResponseRedirect('/method_selection_random_walk')
            if 'submit_parameters_similarity_graph' in request.POST:
                context = process_parameters(request, context)
                return HttpResponseRedirect('/result_similarity_graph')
        context = import_log_discover_features_and_model(request, context)
        return render(request, 'detection/method_selection_similarity_graph.html', context)
    else:
        return render(request, 'detection/page-404.html', context)

def method_selection_supervised_learning(request):
    context = {}
    context['selection_method'] = 'supervised_learning'
    if request.session.get('log_path', None):
        if request.method == 'POST':
            if 'similarity_graph' in request.POST:
                return HttpResponseRedirect('/method_selection_similarity_graph')
            if 'random_walk' in request.POST:
                return HttpResponseRedirect('/method_selection_random_walk')
            if 'submit_parameters_supervised_learning' in request.POST:
                context = process_parameters(request, context)
                return HttpResponseRedirect('/result_supervised_learning')
        context = import_log_discover_features_and_model(request, context)
        return render(request, 'detection/method_selection_supervised_learning.html', context)
    else:
        return render(request, 'detection/page-404.html', context)

def method_selection_random_walk(request):
    context = {}
    context['selection_method'] = 'random_walk'
    if request.session.get('log_path', None):
        if request.method == 'POST':
            if 'similarity_graph' in request.POST:
                return HttpResponseRedirect('/method_selection_similarity_graph')
            if 'supervised_learning' in request.POST:
                return HttpResponseRedirect('/method_selection_supervised_learning')
            if 'submit_parameters_random_walk' in request.POST:
                context = process_parameters(request, context)
                return HttpResponseRedirect('/result_random_walk')
        context = import_log_discover_features_and_model(request, context)
        return render(request, 'detection/method_selection_random_walk.html', context)
    else:
        return render(request, 'detection/page-404.html', context)

def result_similarity_graph(request):
    context = {}
    #context = log_to_features_and_model(request)
    context = apply_similarity_graph(request, context)
    #json_data = detect_surprising_instances_similarity_graph(log_path=request.session['log_path'], pattern_id=None)
    #print(json_data)
    #pd_data = pd.read_json(json_data, orient='split')
    #data_length = len(pd_data.index)
    #print('Surprising instance count: ' + str(data_length))
    #context['surprising_instances'] = pd_data
    return render(request, 'detection/result_similarity_graph.html', context)

def result_supervised_learning(request):
    context = {}
    #context = log_to_features_and_model(request)
    context = apply_supervised_learning(request, context)
    return render(request, 'detection/result_supervised_learning.html', context)

def result_random_walk(request):
    context = {}
    #context = log_to_features_and_model(request)
    context = apply_random_walk(request, context)
    #json_data = detect_surprising_instances_similarity_graph(log_path=request.session['log_path'], pattern_id=None)
    #print(json_data)
    #pd_data = pd.read_json(json_data, orient='split')
    #data_length = len(pd_data.index)
    #print('Surprising instance count: ' + str(data_length))
    #context['surprising_instances'] = pd_data
    return render(request, 'detection/result_random_walk.html', context)