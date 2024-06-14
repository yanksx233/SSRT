from .SSRT import SSRT

# other stereo image SR models
from .Other.NAFNetSR import NAFNetSR, NAFNetSRLocal
from .Other.DEFNet import SSRDEFNet
from .Other.iPASSR import iPASSR
from .Other.PASSRnet import PASSRnet

# other SISR models
from .Other.RCAN import RCAN
from .Other.RDN import RDN
from .Other.EDSR import EDSR


def build_model(args):
    assert (args.num_feats % 16 == 0)
    depths = list(map(int, args.depths.split('*')))
    depths = [depths[0]] * depths[1]
    
    if args.arch == 'ssrt':
        model = SSRT(dim=args.num_feats, upscale=args.upscale, num_heads=[args.num_heads]*len(depths), depths=depths, kernel_size=args.kernel_size,
                           window_size=args.window_size, drop_path_rate=args.drop_path_rate, num_cats=args.num_cats, use_checkpoint=args.use_checkpoint)

    elif args.arch == 'naf':
        if args.mode == 'train':
            model = NAFNetSR(up_scale=args.upscale, drop_path_rate=args.drop_path_rate)
        else:
            model = NAFNetSRLocal(up_scale=args.upscale, drop_path_rate=args.drop_path_rate)

    elif args.arch == 'ipass':
        model = iPASSR(args.upscale)

    elif args.arch == 'pass':
        model = PASSRnet(args.upscale)

    elif args.arch == 'defnet':
        model = SSRDEFNet(args.upscale)
    
    elif args.arch == 'swinir':
        model = SwinIR(upscale=args.upscale)

    elif args.arch == 'rcan':
        model = RCAN(args.upscale)

    elif args.arch == 'edsr':
        model = EDSR(args.upscale)

    elif args.arch == 'rdn':
        model = RDN(args.upscale)
    
    else:
        raise ValueError(f'Unknown arch: {args.arch}')

    return model