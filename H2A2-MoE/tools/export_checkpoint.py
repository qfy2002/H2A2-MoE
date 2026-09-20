#!/usr/bin/env python
"""Export model weights without optimizer state, hooks, or private metadata."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error('Use a separate destination; the source is preserved.')
    if args.destination.exists():
        parser.error('Destination already exists.')
    import torch
    from collections import OrderedDict

    # Only load checkpoints from a trusted source: older Torch uses pickle.
    checkpoint = torch.load(args.source, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor state_dict or an MMEngine checkpoint.')
    def strip_module(key):
        return key[7:] if key.startswith('module.') else key
    tensors = OrderedDict((strip_module(k), v.detach().cpu()) for k, v in state.items())
    if len(tensors) != len(state):
        raise ValueError('Duplicate keys after removing the DDP prefix.')
    # Module versions are needed by MMDetection3D checkpoint conversion logic.
    if hasattr(state, '_metadata'):
        tensors._metadata = OrderedDict(
            ('' if k == 'module' else strip_module(k),
             {'version': v['version']} if 'version' in v else {})
            for k, v in state._metadata.items())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open('xb') as handle:
        torch.save({'state_dict': tensors}, handle)
    print('Exported {} tensors to {}'.format(len(tensors), args.destination))


if __name__ == '__main__':
    main()
