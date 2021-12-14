import argparse
import torch

from openpifpaf import datasets
from openpifpaf import decoder, network, visualizer, show, logger, Predictor
from openpifpaf.predict import out_name


parser = argparse.ArgumentParser(prog='python3 predict', usage='%(prog)s [options] images', description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True, help='Whether to output an image, with the option to specify the output path or directory')
parser.add_argument('--json-output', default=None, nargs='?', const=True,help='Whether to output a json file, with the option to specify the output path or directory')
parser.add_argument('--batch_size', default=1, type=int, help='processing batch size')
parser.add_argument('--long-edge', default=None, type=int, help='rescale the long side of the image (aspect ratio maintained)')
parser.add_argument('--loader-workers', default=None, type=int, help='number of workers for data loading')
parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=str, help='cuda device')
parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false', help='use more exact image rescaling (requires scipy)')
parser.add_argument('--checkpoint_', default='shufflenetv2k30', type=str, help='backbone model to use')
parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')

class PifPafPredictor():
    """
        Class definition for the predictor API using OpenPifpaf
    """
    def __init__(self, parser=parser):
        decoder.cli(parser)
        logger.cli(parser)
        network.Factory.cli(parser)
        show.cli(parser)
        visualizer.cli(parser)
        args = parser.parse_args()

        pifpaf_model = args.checkpoint_
        args.figure_width = 10
        args.dpi_factor = 1.0
        args.batch_size = 1
        args.keypoint_threshold_rel = 0.0
        args.instance_threshold = 0.2
        args.keypoint_threshold = 0
        args.force_complete_pose = True

        # Configure
        decoder.configure(args)
        network.Factory.configure(args)
        Predictor.configure(args)
        show.configure(args)
        visualizer.configure(args)
        self.predictor = Predictor(checkpoint=pifpaf_model)
    