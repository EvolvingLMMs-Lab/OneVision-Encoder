import torch.nn as nn
from ..builder import NECKS


@NECKS.register_module()
class VSGNFPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_levels,
        scale_factor=2,
    ):
        super(VSGNFPN, self).__init__()

        self.num_levels = num_levels
        self.scale_factor = scale_factor

        # Transit from encoder to decoder
        self.levels_transit = self._make_levels_same(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        # Upsampling modules by fusing encoder features and decoder features
        self.levels_enc = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.levels_enc.append(
                self._make_levels_same(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )

        self.levels_dec = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.levels_dec.append(
                self._make_levels_up(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=scale_factor,
                    output_padding=1,
                )
            )

        self.levels_fuse = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.levels_fuse.append(
                self._make_levels_same(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )

    def _make_levels_up(self, in_channels, out_channels, stride=2, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=output_padding,
                groups=1,
            ),
            nn.ReLU(inplace=True),
        )

    def _make_levels_same(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward_one_layer(self, module_enc, module_dec, module_fuse, input_enc, input_dec):
        feat_enc = module_enc(input_enc)
        feat_dec = module_dec(input_dec)
        return module_fuse(feat_enc + feat_dec)

    def forward(self, feat_list, mask_list):
        feats = []

        # First decoder level
        feats.append(self.levels_transit(feat_list[-1]))

        # The rest decoder levels
        for i in range(1, self.num_levels):
            feat = self.forward_one_layer(
                self.levels_enc[i - 1],
                self.levels_dec[i - 1],
                self.levels_fuse[i - 1],
                feat_list[::-1][i],
                feats[i - 1],
            )
            feats.append(feat)

        # Reverse the order of feats
        feats = feats[::-1]  # [B,C,T], [B,C,T//2], [B,C,T//4] ...

        # Masking
        for i in range(len(feats)):
            feats[i] = feats[i] * mask_list[i].unsqueeze(1).float().detach()
        return feats, mask_list
