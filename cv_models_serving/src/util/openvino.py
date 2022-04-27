from openvino.inference_engine import IENetwork, IEPlugin

import logging
import pathlib

logger = logging.getLogger(__name__)

OPENVINO_HOME = pathlib.Path('/opt/intel/computer_vision_sdk')
CPU_AVX_EXTENSION = OPENVINO_HOME / 'deployment_tools' / 'inference_engine' / 'lib' / 'ubuntu_16.04' / 'intel64' / 'libcpu_extension_avx2.so'
CPU_SSE4_EXTENSION = OPENVINO_HOME / 'deployment_tools' / 'inference_engine' / 'lib' / 'ubuntu_16.04' / 'intel64' / 'libcpu_extension_sse4.so'


def load(path_to_inference_graph, path_to_inference_weights):
    plugin = IEPlugin(device='CPU', plugin_dirs=None)

    if CPU_AVX_EXTENSION:
        plugin.add_cpu_extension(str(CPU_AVX_EXTENSION))
        plugin.add_cpu_extension(str(CPU_SSE4_EXTENSION))

    # print("Reading IR...")
    net = IENetwork.from_ir(model=path_to_inference_graph, weights=path_to_inference_weights)

    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        msg = 'Following layers are not supported by the plugin for specified device %s:\n %s' % (plugin.device, ', '.join(not_supported_layers))
        msg += '\nPlease try to specify cpu extensions library path in demo\'s command line parameters using -l or --cpu_extension command line argument'

        raise Exception(msg)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=1)

    # Read and pre-process input image
    input_shape = net.inputs[input_blob].shape

    logger.info('net_input_shape %s', input_shape)

    del net

    return exec_net, input_blob, out_blob, input_shape
