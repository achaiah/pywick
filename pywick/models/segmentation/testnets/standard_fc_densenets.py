from .fc_densenet import FCDenseNet


class FCDenseNet103(FCDenseNet):
    def __init__(self, in_channels=3, out_channels=1000, dropout=0.2):
        super(FCDenseNet103, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            initial_num_features=48,
            dropout=dropout,

            down_dense_growth_rates=16,
            down_dense_bottleneck_ratios=None,
            down_dense_num_layers=(4, 5, 7, 10, 12),
            down_transition_compression_factors=1.0,

            middle_dense_growth_rate=16,
            middle_dense_bottleneck=None,
            middle_dense_num_layers=15,

            up_dense_growth_rates=16,
            up_dense_bottleneck_ratios=None,
            up_dense_num_layers=(12, 10, 7, 5, 4)
        )
