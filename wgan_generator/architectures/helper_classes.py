import inspect

# Class to inhereit to have forward_reshape for flattening models
class FlatImageForwardReshape:
    def forward_reshape(self, z):
        if not hasattr(self, 'net'):
            raise NotImplementedError("Model must define 'net' for default forward_reshape.")
        if not hasattr(self, 'img_dim'):
            raise NotImplementedError("Model must define 'img_dim' for default forward_reshape.")

        out_flat = self.net(z)
        return out_flat.view(-1, *self.img_dim)

# Simple outputs forward reshape as forward
class IdentityForwardReshape:
    def forward_reshape(self, z):
        return self.forward(z)

# Has function to return passed inputs as hparam dict
class HParams:
    def hparams(self):
        # Get the signature of the constructor (only parameters passed to __init__)
        params = inspect.signature(self.__init__).parameters
        
        # Only include parameters that are passed to __init__
        hparams = {k: v.default for k, v in params.items() if v.default != inspect.Parameter.empty}
        return hparams