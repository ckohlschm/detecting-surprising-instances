from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('method_selection_similarity_graph', views.method_selection_similarity_graph, name='method_selection_similarity_graph'),
    path('method_selection_supervised_learning', views.method_selection_supervised_learning, name='method_selection_supervised_learning'),
    path('method_selection_random_walk', views.method_selection_random_walk, name='method_selection_random_walk'),
    path('result_similarity_graph', views.result_similarity_graph, name='result_similarity_graph'),
    path('result_supervised_learning', views.result_supervised_learning, name='result_supervised_learning'),
    path('result_random_walk', views.result_random_walk, name='result_random_walk'),
    path('not_found', views.not_found, name='not_found')
]
