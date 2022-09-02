from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('method_selection_similarity_graph', views.method_selection_similarity_graph, name='method_selection_similarity_graph'),
    path('method_selection_classification', views.method_selection_classification, name='method_selection_classification'),
    path('method_selection_clustering', views.method_selection_clustering, name='method_selection_clustering'),
    path('result_similarity_graph', views.result_similarity_graph, name='result_similarity_graph'),
    path('result_classification', views.result_classification, name='result_classification'),
    path('result_clustering', views.result_clustering, name='result_clustering'),
    path('root_cause_analysis_classification', views.root_cause_analysis_classification, name='root_cause_analysis_classification'),
    path('not_found', views.not_found, name='not_found')
]
