import json
import os.path

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect

from .client import discover_model_as_image, discover_workflow_patterns_as_json

def send_log_to_backend(request, context, model_name):
    request.session['model_name'] = model_name
    if request.session.get('log_path', None):
        result = discover_model_as_image(request.session['log_path'], model_name)
        if result:
            print("Model found")


def is_navigation_request(request):
    return any(x in request.POST for x in ['bpmn', 'workflow', 'pn'])


def handle_navigation(request, context):
    if 'workflow' in request.POST:
        return send_log_to_backend(request, context, 'workflow')
    elif 'bpmn' in request.POST:
        return send_log_to_backend(request, context, 'bpmn')
    elif 'pn' in request.POST:
        return send_log_to_backend(request, context, 'pn')
    else:
        return redirect('index')


def is_upload_request(request):
    return 'upload' in request.POST


def handle_upload(request, context):
    if request.FILES:
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        path = 'logs/' + uploaded_file.name
        file_path = fs.save(path, uploaded_file)
        request.session['log_path'] = file_path
        image_result = discover_model_as_image(file_path, 'workflow')
        json_result = discover_workflow_patterns_as_json(file_path)
        try:
            parse_json = json.loads(json_result)
            patterns = json.loads(parse_json["patterns"])
        except Exception:
            context['upload_error'] = 'File import not possible. Please select a valid XES or CSV file!'
            return render(request, 'detection/index.html', context)
        for pattern in patterns:
            pattern['checked'] = False
        for pattern in patterns:
            pattern['color'] = '#D3D3D3'
        request.session['patterns'] = patterns
        if json_result and image_result:
            return redirect('workflow')
        else:
            context['upload_error'] = 'File import not possible. Please select a valid XES or CSV file!'
            return render(request, 'detection/index.html', context)
    else:
        context['upload_error'] = 'Please select a valid file'
        return render(request, 'detection/index.html', context)


def is_merge_request(request):
    return 'aggregate-pattern' in request.POST


def handle_merge_request(request, context, model_name):
    if request.session.get('patterns', None):
        patterns = request.session['patterns']
        for pattern in patterns:
            pattern['checked'] = False
    if request.session.get('log_path', None):
        for item in request.POST:
            if item.endswith('split') or item.endswith('join'):
                for pattern in patterns:
                    if pattern['pattern_node'] == item:
                        pattern['checked'] = True
        return send_log_to_backend(request, context, model_name)
    else:
        context['upload_error'] = 'No event log found!'
        return render(request, 'detection/index.html', context)


def is_color_request(request):
    return 'color-pattern' in request.POST


def handle_color_request(request, context, model_name):
    if request.session.get('patterns', None):
        patterns = request.session['patterns']
        for pattern in patterns:
            pattern['color'] = '#D3D3D3'
    if request.session.get('log_path', None):
        for pattern in patterns:
            color_id = 'color-' + str(pattern['pattern_node'])
            color = request.POST.get(color_id, None)
            pattern['color'] = color

        return send_log_to_backend(request, context, model_name)
    else:
        context['upload_error'] = 'No event log found!'
        return render(request, 'detection/index.html', context)


def get_model_representation(request, model_name):
    context = {}
    context['model_name'] = None
    context['show_model'] = False

    if is_navigation_request(request):
        return handle_navigation(request, context)
    if is_upload_request(request):
        return handle_upload(request, context)
    if is_merge_request(request):
        return handle_merge_request(request, context, model_name)
    if is_color_request(request):
        return handle_color_request(request, context, model_name)
    image_path = 'detection/static/detection/models/' + model_name + '.png'
    if os.path.isfile(image_path):
        context['show_model'] = True
    context['model_name'] = model_name
    if request.session.get('patterns', None):
        context['patterns'] = request.session['patterns']
    return render(request, 'detection/' + model_name + '.html', context)
