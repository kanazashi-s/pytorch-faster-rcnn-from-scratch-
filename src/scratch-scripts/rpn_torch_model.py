#%%

from torch import nn

mid_channels = 512
in_channels = 512 # depends on the output feature map. in vgg 16 it is equal to 512
n_anchor = 9 # Number of anchors at each location

conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

# using softmax as an activate function.
cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

#%%

# Conv layer
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# Regression layer
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# Classification layer
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

#%%

x = conv1(out_feature_map) # out_map is obtained in section 1
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

#%%
