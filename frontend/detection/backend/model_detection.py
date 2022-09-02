import json
from multiprocessing import Event

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as bpmn_converter
from pm4py import discover_process_tree_inductive as discover_process_tree_inductive
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.common import save as gsave
from pm4py.visualization.petrinet import visualizer as pn_vis

from .importer import import_file

def discover_pn_model(log_path, model_name, event_log=None):
    """
    Discover the petrinet models for an event log
    Parameters
    --------------
    log_path
        path to the event log
    model_name
        name of the pn models
    Returns
    --------------
    model_path
        path to the discovered pn models
    """
    print("Discovering Petri-Net")
    if not event_log:
        event_log = import_file(log_path, False)
    net, initial_marking, final_marking = inductive_miner.apply(event_log)
    gviz = pn_vis.apply(net, initial_marking, final_marking)
    model_path = 'detection/static/detection/models/' + model_name + '.png'
    pn_vis.save(gviz, model_path)
    model_path = model_path.replace('detection/static/', '', 1)
    return model_path


def discover_bpmn_model(log_path, model_name, event_log=None):
    """
    Discover the BPMN for an event log
    Parameters
    --------------
    log_path
        path to the event log
    model_name
        name of the BPMN
    Returns
    --------------
    model_path
        path to the discovered BPMN
    """
    print("Discovering BPMN Model")
    if not event_log:
        event_log = import_file(log_path, False)
    tree = discover_process_tree_inductive(event_log)
    bpmn_graph = bpmn_converter.apply(tree, variant=bpmn_converter.Variants.TO_BPMN)
    gviz = bpmn_visualizer.apply(bpmn_graph)
    model_path = 'detection/static/detection/models/' + model_name + '.png'
    gsave.save(gviz, model_path)
    model_path = model_path.replace('detection/static/', '', 1)
    return model_path
