import argparse

def depth_anything_parser(parser):
    group = parser.add_argument_group("Depth Anything v2 Arguments")
    group.add_argument('--input-size', type=int, default=518)
    group.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])

    #group.add_argument('--pretrained_model_path', type=str, default='checkpoints/DepthAnythingV2', help='pretrained model path')

    return parser

