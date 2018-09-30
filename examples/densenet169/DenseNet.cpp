#include "DenseNet.h"


DenseNet DenseNetLoad(const char* filename)
{
	DenseNet ctx;
	t4::model_dict dict = t4::load(filename);
	dict.load(ctx.features_conv0_weight, "features.conv0.weight", 64, 3, 7, 7);
	dict.load(ctx.features_norm0_weight, "features.norm0.weight", 64);
	dict.load(ctx.features_norm0_bias, "features.norm0.bias", 64);
	dict.load(ctx.features_norm0_running_mean, "features.norm0.running_mean", 64);
	dict.load(ctx.features_norm0_running_var, "features.norm0.running_var", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_weight, "features.denseblock1.denselayer1.norm1.weight", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_bias, "features.denseblock1.denselayer1.norm1.bias", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_running_mean, "features.denseblock1.denselayer1.norm1.running_mean", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_running_var, "features.denseblock1.denselayer1.norm1.running_var", 64);
	dict.load(ctx.features_denseblock1_denselayer1_conv1_weight, "features.denseblock1.denselayer1.conv1.weight", 128, 64, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_weight, "features.denseblock1.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_bias, "features.denseblock1.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_running_mean, "features.denseblock1.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_running_var, "features.denseblock1.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer1_conv2_weight, "features.denseblock1.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_weight, "features.denseblock1.denselayer2.norm1.weight", 96);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_bias, "features.denseblock1.denselayer2.norm1.bias", 96);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_running_mean, "features.denseblock1.denselayer2.norm1.running_mean", 96);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_running_var, "features.denseblock1.denselayer2.norm1.running_var", 96);
	dict.load(ctx.features_denseblock1_denselayer2_conv1_weight, "features.denseblock1.denselayer2.conv1.weight", 128, 96, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_weight, "features.denseblock1.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_bias, "features.denseblock1.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_running_mean, "features.denseblock1.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_running_var, "features.denseblock1.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer2_conv2_weight, "features.denseblock1.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_weight, "features.denseblock1.denselayer3.norm1.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_bias, "features.denseblock1.denselayer3.norm1.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_running_mean, "features.denseblock1.denselayer3.norm1.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_running_var, "features.denseblock1.denselayer3.norm1.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer3_conv1_weight, "features.denseblock1.denselayer3.conv1.weight", 128, 128, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_weight, "features.denseblock1.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_bias, "features.denseblock1.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_running_mean, "features.denseblock1.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_running_var, "features.denseblock1.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer3_conv2_weight, "features.denseblock1.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_weight, "features.denseblock1.denselayer4.norm1.weight", 160);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_bias, "features.denseblock1.denselayer4.norm1.bias", 160);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_running_mean, "features.denseblock1.denselayer4.norm1.running_mean", 160);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_running_var, "features.denseblock1.denselayer4.norm1.running_var", 160);
	dict.load(ctx.features_denseblock1_denselayer4_conv1_weight, "features.denseblock1.denselayer4.conv1.weight", 128, 160, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_weight, "features.denseblock1.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_bias, "features.denseblock1.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_running_mean, "features.denseblock1.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_running_var, "features.denseblock1.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer4_conv2_weight, "features.denseblock1.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_weight, "features.denseblock1.denselayer5.norm1.weight", 192);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_bias, "features.denseblock1.denselayer5.norm1.bias", 192);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_running_mean, "features.denseblock1.denselayer5.norm1.running_mean", 192);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_running_var, "features.denseblock1.denselayer5.norm1.running_var", 192);
	dict.load(ctx.features_denseblock1_denselayer5_conv1_weight, "features.denseblock1.denselayer5.conv1.weight", 128, 192, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_weight, "features.denseblock1.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_bias, "features.denseblock1.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_running_mean, "features.denseblock1.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_running_var, "features.denseblock1.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer5_conv2_weight, "features.denseblock1.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_weight, "features.denseblock1.denselayer6.norm1.weight", 224);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_bias, "features.denseblock1.denselayer6.norm1.bias", 224);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_running_mean, "features.denseblock1.denselayer6.norm1.running_mean", 224);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_running_var, "features.denseblock1.denselayer6.norm1.running_var", 224);
	dict.load(ctx.features_denseblock1_denselayer6_conv1_weight, "features.denseblock1.denselayer6.conv1.weight", 128, 224, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_weight, "features.denseblock1.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_bias, "features.denseblock1.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_running_mean, "features.denseblock1.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_running_var, "features.denseblock1.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer6_conv2_weight, "features.denseblock1.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_transition1_norm_weight, "features.transition1.norm.weight", 256);
	dict.load(ctx.features_transition1_norm_bias, "features.transition1.norm.bias", 256);
	dict.load(ctx.features_transition1_norm_running_mean, "features.transition1.norm.running_mean", 256);
	dict.load(ctx.features_transition1_norm_running_var, "features.transition1.norm.running_var", 256);
	dict.load(ctx.features_transition1_conv_weight, "features.transition1.conv.weight", 128, 256, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_weight, "features.denseblock2.denselayer1.norm1.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_bias, "features.denseblock2.denselayer1.norm1.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_running_mean, "features.denseblock2.denselayer1.norm1.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_running_var, "features.denseblock2.denselayer1.norm1.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer1_conv1_weight, "features.denseblock2.denselayer1.conv1.weight", 128, 128, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_weight, "features.denseblock2.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_bias, "features.denseblock2.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_running_mean, "features.denseblock2.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_running_var, "features.denseblock2.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer1_conv2_weight, "features.denseblock2.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_weight, "features.denseblock2.denselayer2.norm1.weight", 160);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_bias, "features.denseblock2.denselayer2.norm1.bias", 160);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_running_mean, "features.denseblock2.denselayer2.norm1.running_mean", 160);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_running_var, "features.denseblock2.denselayer2.norm1.running_var", 160);
	dict.load(ctx.features_denseblock2_denselayer2_conv1_weight, "features.denseblock2.denselayer2.conv1.weight", 128, 160, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_weight, "features.denseblock2.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_bias, "features.denseblock2.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_running_mean, "features.denseblock2.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_running_var, "features.denseblock2.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer2_conv2_weight, "features.denseblock2.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_weight, "features.denseblock2.denselayer3.norm1.weight", 192);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_bias, "features.denseblock2.denselayer3.norm1.bias", 192);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_running_mean, "features.denseblock2.denselayer3.norm1.running_mean", 192);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_running_var, "features.denseblock2.denselayer3.norm1.running_var", 192);
	dict.load(ctx.features_denseblock2_denselayer3_conv1_weight, "features.denseblock2.denselayer3.conv1.weight", 128, 192, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_weight, "features.denseblock2.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_bias, "features.denseblock2.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_running_mean, "features.denseblock2.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_running_var, "features.denseblock2.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer3_conv2_weight, "features.denseblock2.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_weight, "features.denseblock2.denselayer4.norm1.weight", 224);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_bias, "features.denseblock2.denselayer4.norm1.bias", 224);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_running_mean, "features.denseblock2.denselayer4.norm1.running_mean", 224);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_running_var, "features.denseblock2.denselayer4.norm1.running_var", 224);
	dict.load(ctx.features_denseblock2_denselayer4_conv1_weight, "features.denseblock2.denselayer4.conv1.weight", 128, 224, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_weight, "features.denseblock2.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_bias, "features.denseblock2.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_running_mean, "features.denseblock2.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_running_var, "features.denseblock2.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer4_conv2_weight, "features.denseblock2.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_weight, "features.denseblock2.denselayer5.norm1.weight", 256);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_bias, "features.denseblock2.denselayer5.norm1.bias", 256);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_running_mean, "features.denseblock2.denselayer5.norm1.running_mean", 256);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_running_var, "features.denseblock2.denselayer5.norm1.running_var", 256);
	dict.load(ctx.features_denseblock2_denselayer5_conv1_weight, "features.denseblock2.denselayer5.conv1.weight", 128, 256, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_weight, "features.denseblock2.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_bias, "features.denseblock2.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_running_mean, "features.denseblock2.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_running_var, "features.denseblock2.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer5_conv2_weight, "features.denseblock2.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_weight, "features.denseblock2.denselayer6.norm1.weight", 288);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_bias, "features.denseblock2.denselayer6.norm1.bias", 288);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_running_mean, "features.denseblock2.denselayer6.norm1.running_mean", 288);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_running_var, "features.denseblock2.denselayer6.norm1.running_var", 288);
	dict.load(ctx.features_denseblock2_denselayer6_conv1_weight, "features.denseblock2.denselayer6.conv1.weight", 128, 288, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_weight, "features.denseblock2.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_bias, "features.denseblock2.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_running_mean, "features.denseblock2.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_running_var, "features.denseblock2.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer6_conv2_weight, "features.denseblock2.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_weight, "features.denseblock2.denselayer7.norm1.weight", 320);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_bias, "features.denseblock2.denselayer7.norm1.bias", 320);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_running_mean, "features.denseblock2.denselayer7.norm1.running_mean", 320);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_running_var, "features.denseblock2.denselayer7.norm1.running_var", 320);
	dict.load(ctx.features_denseblock2_denselayer7_conv1_weight, "features.denseblock2.denselayer7.conv1.weight", 128, 320, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_weight, "features.denseblock2.denselayer7.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_bias, "features.denseblock2.denselayer7.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_running_mean, "features.denseblock2.denselayer7.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_running_var, "features.denseblock2.denselayer7.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer7_conv2_weight, "features.denseblock2.denselayer7.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_weight, "features.denseblock2.denselayer8.norm1.weight", 352);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_bias, "features.denseblock2.denselayer8.norm1.bias", 352);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_running_mean, "features.denseblock2.denselayer8.norm1.running_mean", 352);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_running_var, "features.denseblock2.denselayer8.norm1.running_var", 352);
	dict.load(ctx.features_denseblock2_denselayer8_conv1_weight, "features.denseblock2.denselayer8.conv1.weight", 128, 352, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_weight, "features.denseblock2.denselayer8.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_bias, "features.denseblock2.denselayer8.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_running_mean, "features.denseblock2.denselayer8.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_running_var, "features.denseblock2.denselayer8.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer8_conv2_weight, "features.denseblock2.denselayer8.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_weight, "features.denseblock2.denselayer9.norm1.weight", 384);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_bias, "features.denseblock2.denselayer9.norm1.bias", 384);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_running_mean, "features.denseblock2.denselayer9.norm1.running_mean", 384);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_running_var, "features.denseblock2.denselayer9.norm1.running_var", 384);
	dict.load(ctx.features_denseblock2_denselayer9_conv1_weight, "features.denseblock2.denselayer9.conv1.weight", 128, 384, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_weight, "features.denseblock2.denselayer9.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_bias, "features.denseblock2.denselayer9.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_running_mean, "features.denseblock2.denselayer9.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_running_var, "features.denseblock2.denselayer9.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer9_conv2_weight, "features.denseblock2.denselayer9.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_weight, "features.denseblock2.denselayer10.norm1.weight", 416);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_bias, "features.denseblock2.denselayer10.norm1.bias", 416);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_running_mean, "features.denseblock2.denselayer10.norm1.running_mean", 416);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_running_var, "features.denseblock2.denselayer10.norm1.running_var", 416);
	dict.load(ctx.features_denseblock2_denselayer10_conv1_weight, "features.denseblock2.denselayer10.conv1.weight", 128, 416, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_weight, "features.denseblock2.denselayer10.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_bias, "features.denseblock2.denselayer10.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_running_mean, "features.denseblock2.denselayer10.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_running_var, "features.denseblock2.denselayer10.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer10_conv2_weight, "features.denseblock2.denselayer10.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_weight, "features.denseblock2.denselayer11.norm1.weight", 448);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_bias, "features.denseblock2.denselayer11.norm1.bias", 448);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_running_mean, "features.denseblock2.denselayer11.norm1.running_mean", 448);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_running_var, "features.denseblock2.denselayer11.norm1.running_var", 448);
	dict.load(ctx.features_denseblock2_denselayer11_conv1_weight, "features.denseblock2.denselayer11.conv1.weight", 128, 448, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_weight, "features.denseblock2.denselayer11.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_bias, "features.denseblock2.denselayer11.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_running_mean, "features.denseblock2.denselayer11.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_running_var, "features.denseblock2.denselayer11.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer11_conv2_weight, "features.denseblock2.denselayer11.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_weight, "features.denseblock2.denselayer12.norm1.weight", 480);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_bias, "features.denseblock2.denselayer12.norm1.bias", 480);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_running_mean, "features.denseblock2.denselayer12.norm1.running_mean", 480);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_running_var, "features.denseblock2.denselayer12.norm1.running_var", 480);
	dict.load(ctx.features_denseblock2_denselayer12_conv1_weight, "features.denseblock2.denselayer12.conv1.weight", 128, 480, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_weight, "features.denseblock2.denselayer12.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_bias, "features.denseblock2.denselayer12.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_running_mean, "features.denseblock2.denselayer12.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_running_var, "features.denseblock2.denselayer12.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer12_conv2_weight, "features.denseblock2.denselayer12.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_transition2_norm_weight, "features.transition2.norm.weight", 512);
	dict.load(ctx.features_transition2_norm_bias, "features.transition2.norm.bias", 512);
	dict.load(ctx.features_transition2_norm_running_mean, "features.transition2.norm.running_mean", 512);
	dict.load(ctx.features_transition2_norm_running_var, "features.transition2.norm.running_var", 512);
	dict.load(ctx.features_transition2_conv_weight, "features.transition2.conv.weight", 256, 512, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_weight, "features.denseblock3.denselayer1.norm1.weight", 256);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_bias, "features.denseblock3.denselayer1.norm1.bias", 256);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_running_mean, "features.denseblock3.denselayer1.norm1.running_mean", 256);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_running_var, "features.denseblock3.denselayer1.norm1.running_var", 256);
	dict.load(ctx.features_denseblock3_denselayer1_conv1_weight, "features.denseblock3.denselayer1.conv1.weight", 128, 256, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_weight, "features.denseblock3.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_bias, "features.denseblock3.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_running_mean, "features.denseblock3.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_running_var, "features.denseblock3.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer1_conv2_weight, "features.denseblock3.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_weight, "features.denseblock3.denselayer2.norm1.weight", 288);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_bias, "features.denseblock3.denselayer2.norm1.bias", 288);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_running_mean, "features.denseblock3.denselayer2.norm1.running_mean", 288);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_running_var, "features.denseblock3.denselayer2.norm1.running_var", 288);
	dict.load(ctx.features_denseblock3_denselayer2_conv1_weight, "features.denseblock3.denselayer2.conv1.weight", 128, 288, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_weight, "features.denseblock3.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_bias, "features.denseblock3.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_running_mean, "features.denseblock3.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_running_var, "features.denseblock3.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer2_conv2_weight, "features.denseblock3.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_weight, "features.denseblock3.denselayer3.norm1.weight", 320);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_bias, "features.denseblock3.denselayer3.norm1.bias", 320);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_running_mean, "features.denseblock3.denselayer3.norm1.running_mean", 320);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_running_var, "features.denseblock3.denselayer3.norm1.running_var", 320);
	dict.load(ctx.features_denseblock3_denselayer3_conv1_weight, "features.denseblock3.denselayer3.conv1.weight", 128, 320, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_weight, "features.denseblock3.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_bias, "features.denseblock3.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_running_mean, "features.denseblock3.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_running_var, "features.denseblock3.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer3_conv2_weight, "features.denseblock3.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_weight, "features.denseblock3.denselayer4.norm1.weight", 352);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_bias, "features.denseblock3.denselayer4.norm1.bias", 352);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_running_mean, "features.denseblock3.denselayer4.norm1.running_mean", 352);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_running_var, "features.denseblock3.denselayer4.norm1.running_var", 352);
	dict.load(ctx.features_denseblock3_denselayer4_conv1_weight, "features.denseblock3.denselayer4.conv1.weight", 128, 352, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_weight, "features.denseblock3.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_bias, "features.denseblock3.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_running_mean, "features.denseblock3.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_running_var, "features.denseblock3.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer4_conv2_weight, "features.denseblock3.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_weight, "features.denseblock3.denselayer5.norm1.weight", 384);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_bias, "features.denseblock3.denselayer5.norm1.bias", 384);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_running_mean, "features.denseblock3.denselayer5.norm1.running_mean", 384);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_running_var, "features.denseblock3.denselayer5.norm1.running_var", 384);
	dict.load(ctx.features_denseblock3_denselayer5_conv1_weight, "features.denseblock3.denselayer5.conv1.weight", 128, 384, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_weight, "features.denseblock3.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_bias, "features.denseblock3.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_running_mean, "features.denseblock3.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_running_var, "features.denseblock3.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer5_conv2_weight, "features.denseblock3.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_weight, "features.denseblock3.denselayer6.norm1.weight", 416);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_bias, "features.denseblock3.denselayer6.norm1.bias", 416);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_running_mean, "features.denseblock3.denselayer6.norm1.running_mean", 416);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_running_var, "features.denseblock3.denselayer6.norm1.running_var", 416);
	dict.load(ctx.features_denseblock3_denselayer6_conv1_weight, "features.denseblock3.denselayer6.conv1.weight", 128, 416, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_weight, "features.denseblock3.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_bias, "features.denseblock3.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_running_mean, "features.denseblock3.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_running_var, "features.denseblock3.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer6_conv2_weight, "features.denseblock3.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_weight, "features.denseblock3.denselayer7.norm1.weight", 448);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_bias, "features.denseblock3.denselayer7.norm1.bias", 448);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_running_mean, "features.denseblock3.denselayer7.norm1.running_mean", 448);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_running_var, "features.denseblock3.denselayer7.norm1.running_var", 448);
	dict.load(ctx.features_denseblock3_denselayer7_conv1_weight, "features.denseblock3.denselayer7.conv1.weight", 128, 448, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_weight, "features.denseblock3.denselayer7.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_bias, "features.denseblock3.denselayer7.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_running_mean, "features.denseblock3.denselayer7.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_running_var, "features.denseblock3.denselayer7.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer7_conv2_weight, "features.denseblock3.denselayer7.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_weight, "features.denseblock3.denselayer8.norm1.weight", 480);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_bias, "features.denseblock3.denselayer8.norm1.bias", 480);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_running_mean, "features.denseblock3.denselayer8.norm1.running_mean", 480);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_running_var, "features.denseblock3.denselayer8.norm1.running_var", 480);
	dict.load(ctx.features_denseblock3_denselayer8_conv1_weight, "features.denseblock3.denselayer8.conv1.weight", 128, 480, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_weight, "features.denseblock3.denselayer8.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_bias, "features.denseblock3.denselayer8.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_running_mean, "features.denseblock3.denselayer8.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_running_var, "features.denseblock3.denselayer8.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer8_conv2_weight, "features.denseblock3.denselayer8.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_weight, "features.denseblock3.denselayer9.norm1.weight", 512);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_bias, "features.denseblock3.denselayer9.norm1.bias", 512);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_running_mean, "features.denseblock3.denselayer9.norm1.running_mean", 512);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_running_var, "features.denseblock3.denselayer9.norm1.running_var", 512);
	dict.load(ctx.features_denseblock3_denselayer9_conv1_weight, "features.denseblock3.denselayer9.conv1.weight", 128, 512, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_weight, "features.denseblock3.denselayer9.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_bias, "features.denseblock3.denselayer9.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_running_mean, "features.denseblock3.denselayer9.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_running_var, "features.denseblock3.denselayer9.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer9_conv2_weight, "features.denseblock3.denselayer9.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_weight, "features.denseblock3.denselayer10.norm1.weight", 544);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_bias, "features.denseblock3.denselayer10.norm1.bias", 544);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_running_mean, "features.denseblock3.denselayer10.norm1.running_mean", 544);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_running_var, "features.denseblock3.denselayer10.norm1.running_var", 544);
	dict.load(ctx.features_denseblock3_denselayer10_conv1_weight, "features.denseblock3.denselayer10.conv1.weight", 128, 544, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_weight, "features.denseblock3.denselayer10.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_bias, "features.denseblock3.denselayer10.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_running_mean, "features.denseblock3.denselayer10.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_running_var, "features.denseblock3.denselayer10.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer10_conv2_weight, "features.denseblock3.denselayer10.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_weight, "features.denseblock3.denselayer11.norm1.weight", 576);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_bias, "features.denseblock3.denselayer11.norm1.bias", 576);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_running_mean, "features.denseblock3.denselayer11.norm1.running_mean", 576);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_running_var, "features.denseblock3.denselayer11.norm1.running_var", 576);
	dict.load(ctx.features_denseblock3_denselayer11_conv1_weight, "features.denseblock3.denselayer11.conv1.weight", 128, 576, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_weight, "features.denseblock3.denselayer11.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_bias, "features.denseblock3.denselayer11.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_running_mean, "features.denseblock3.denselayer11.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_running_var, "features.denseblock3.denselayer11.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer11_conv2_weight, "features.denseblock3.denselayer11.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_weight, "features.denseblock3.denselayer12.norm1.weight", 608);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_bias, "features.denseblock3.denselayer12.norm1.bias", 608);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_running_mean, "features.denseblock3.denselayer12.norm1.running_mean", 608);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_running_var, "features.denseblock3.denselayer12.norm1.running_var", 608);
	dict.load(ctx.features_denseblock3_denselayer12_conv1_weight, "features.denseblock3.denselayer12.conv1.weight", 128, 608, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_weight, "features.denseblock3.denselayer12.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_bias, "features.denseblock3.denselayer12.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_running_mean, "features.denseblock3.denselayer12.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_running_var, "features.denseblock3.denselayer12.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer12_conv2_weight, "features.denseblock3.denselayer12.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_weight, "features.denseblock3.denselayer13.norm1.weight", 640);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_bias, "features.denseblock3.denselayer13.norm1.bias", 640);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_running_mean, "features.denseblock3.denselayer13.norm1.running_mean", 640);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_running_var, "features.denseblock3.denselayer13.norm1.running_var", 640);
	dict.load(ctx.features_denseblock3_denselayer13_conv1_weight, "features.denseblock3.denselayer13.conv1.weight", 128, 640, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_weight, "features.denseblock3.denselayer13.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_bias, "features.denseblock3.denselayer13.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_running_mean, "features.denseblock3.denselayer13.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_running_var, "features.denseblock3.denselayer13.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer13_conv2_weight, "features.denseblock3.denselayer13.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_weight, "features.denseblock3.denselayer14.norm1.weight", 672);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_bias, "features.denseblock3.denselayer14.norm1.bias", 672);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_running_mean, "features.denseblock3.denselayer14.norm1.running_mean", 672);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_running_var, "features.denseblock3.denselayer14.norm1.running_var", 672);
	dict.load(ctx.features_denseblock3_denselayer14_conv1_weight, "features.denseblock3.denselayer14.conv1.weight", 128, 672, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_weight, "features.denseblock3.denselayer14.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_bias, "features.denseblock3.denselayer14.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_running_mean, "features.denseblock3.denselayer14.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_running_var, "features.denseblock3.denselayer14.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer14_conv2_weight, "features.denseblock3.denselayer14.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_weight, "features.denseblock3.denselayer15.norm1.weight", 704);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_bias, "features.denseblock3.denselayer15.norm1.bias", 704);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_running_mean, "features.denseblock3.denselayer15.norm1.running_mean", 704);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_running_var, "features.denseblock3.denselayer15.norm1.running_var", 704);
	dict.load(ctx.features_denseblock3_denselayer15_conv1_weight, "features.denseblock3.denselayer15.conv1.weight", 128, 704, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_weight, "features.denseblock3.denselayer15.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_bias, "features.denseblock3.denselayer15.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_running_mean, "features.denseblock3.denselayer15.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_running_var, "features.denseblock3.denselayer15.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer15_conv2_weight, "features.denseblock3.denselayer15.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_weight, "features.denseblock3.denselayer16.norm1.weight", 736);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_bias, "features.denseblock3.denselayer16.norm1.bias", 736);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_running_mean, "features.denseblock3.denselayer16.norm1.running_mean", 736);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_running_var, "features.denseblock3.denselayer16.norm1.running_var", 736);
	dict.load(ctx.features_denseblock3_denselayer16_conv1_weight, "features.denseblock3.denselayer16.conv1.weight", 128, 736, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_weight, "features.denseblock3.denselayer16.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_bias, "features.denseblock3.denselayer16.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_running_mean, "features.denseblock3.denselayer16.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_running_var, "features.denseblock3.denselayer16.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer16_conv2_weight, "features.denseblock3.denselayer16.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_weight, "features.denseblock3.denselayer17.norm1.weight", 768);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_bias, "features.denseblock3.denselayer17.norm1.bias", 768);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_running_mean, "features.denseblock3.denselayer17.norm1.running_mean", 768);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_running_var, "features.denseblock3.denselayer17.norm1.running_var", 768);
	dict.load(ctx.features_denseblock3_denselayer17_conv1_weight, "features.denseblock3.denselayer17.conv1.weight", 128, 768, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_weight, "features.denseblock3.denselayer17.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_bias, "features.denseblock3.denselayer17.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_running_mean, "features.denseblock3.denselayer17.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_running_var, "features.denseblock3.denselayer17.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer17_conv2_weight, "features.denseblock3.denselayer17.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_weight, "features.denseblock3.denselayer18.norm1.weight", 800);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_bias, "features.denseblock3.denselayer18.norm1.bias", 800);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_running_mean, "features.denseblock3.denselayer18.norm1.running_mean", 800);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_running_var, "features.denseblock3.denselayer18.norm1.running_var", 800);
	dict.load(ctx.features_denseblock3_denselayer18_conv1_weight, "features.denseblock3.denselayer18.conv1.weight", 128, 800, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_weight, "features.denseblock3.denselayer18.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_bias, "features.denseblock3.denselayer18.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_running_mean, "features.denseblock3.denselayer18.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_running_var, "features.denseblock3.denselayer18.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer18_conv2_weight, "features.denseblock3.denselayer18.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_weight, "features.denseblock3.denselayer19.norm1.weight", 832);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_bias, "features.denseblock3.denselayer19.norm1.bias", 832);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_running_mean, "features.denseblock3.denselayer19.norm1.running_mean", 832);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_running_var, "features.denseblock3.denselayer19.norm1.running_var", 832);
	dict.load(ctx.features_denseblock3_denselayer19_conv1_weight, "features.denseblock3.denselayer19.conv1.weight", 128, 832, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_weight, "features.denseblock3.denselayer19.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_bias, "features.denseblock3.denselayer19.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_running_mean, "features.denseblock3.denselayer19.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_running_var, "features.denseblock3.denselayer19.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer19_conv2_weight, "features.denseblock3.denselayer19.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_weight, "features.denseblock3.denselayer20.norm1.weight", 864);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_bias, "features.denseblock3.denselayer20.norm1.bias", 864);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_running_mean, "features.denseblock3.denselayer20.norm1.running_mean", 864);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_running_var, "features.denseblock3.denselayer20.norm1.running_var", 864);
	dict.load(ctx.features_denseblock3_denselayer20_conv1_weight, "features.denseblock3.denselayer20.conv1.weight", 128, 864, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_weight, "features.denseblock3.denselayer20.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_bias, "features.denseblock3.denselayer20.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_running_mean, "features.denseblock3.denselayer20.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_running_var, "features.denseblock3.denselayer20.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer20_conv2_weight, "features.denseblock3.denselayer20.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_weight, "features.denseblock3.denselayer21.norm1.weight", 896);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_bias, "features.denseblock3.denselayer21.norm1.bias", 896);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_running_mean, "features.denseblock3.denselayer21.norm1.running_mean", 896);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_running_var, "features.denseblock3.denselayer21.norm1.running_var", 896);
	dict.load(ctx.features_denseblock3_denselayer21_conv1_weight, "features.denseblock3.denselayer21.conv1.weight", 128, 896, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_weight, "features.denseblock3.denselayer21.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_bias, "features.denseblock3.denselayer21.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_running_mean, "features.denseblock3.denselayer21.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_running_var, "features.denseblock3.denselayer21.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer21_conv2_weight, "features.denseblock3.denselayer21.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_weight, "features.denseblock3.denselayer22.norm1.weight", 928);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_bias, "features.denseblock3.denselayer22.norm1.bias", 928);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_running_mean, "features.denseblock3.denselayer22.norm1.running_mean", 928);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_running_var, "features.denseblock3.denselayer22.norm1.running_var", 928);
	dict.load(ctx.features_denseblock3_denselayer22_conv1_weight, "features.denseblock3.denselayer22.conv1.weight", 128, 928, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_weight, "features.denseblock3.denselayer22.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_bias, "features.denseblock3.denselayer22.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_running_mean, "features.denseblock3.denselayer22.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_running_var, "features.denseblock3.denselayer22.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer22_conv2_weight, "features.denseblock3.denselayer22.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_weight, "features.denseblock3.denselayer23.norm1.weight", 960);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_bias, "features.denseblock3.denselayer23.norm1.bias", 960);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_running_mean, "features.denseblock3.denselayer23.norm1.running_mean", 960);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_running_var, "features.denseblock3.denselayer23.norm1.running_var", 960);
	dict.load(ctx.features_denseblock3_denselayer23_conv1_weight, "features.denseblock3.denselayer23.conv1.weight", 128, 960, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_weight, "features.denseblock3.denselayer23.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_bias, "features.denseblock3.denselayer23.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_running_mean, "features.denseblock3.denselayer23.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_running_var, "features.denseblock3.denselayer23.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer23_conv2_weight, "features.denseblock3.denselayer23.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_weight, "features.denseblock3.denselayer24.norm1.weight", 992);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_bias, "features.denseblock3.denselayer24.norm1.bias", 992);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_running_mean, "features.denseblock3.denselayer24.norm1.running_mean", 992);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_running_var, "features.denseblock3.denselayer24.norm1.running_var", 992);
	dict.load(ctx.features_denseblock3_denselayer24_conv1_weight, "features.denseblock3.denselayer24.conv1.weight", 128, 992, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_weight, "features.denseblock3.denselayer24.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_bias, "features.denseblock3.denselayer24.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_running_mean, "features.denseblock3.denselayer24.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_running_var, "features.denseblock3.denselayer24.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer24_conv2_weight, "features.denseblock3.denselayer24.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_weight, "features.denseblock3.denselayer25.norm1.weight", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_bias, "features.denseblock3.denselayer25.norm1.bias", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_running_mean, "features.denseblock3.denselayer25.norm1.running_mean", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_running_var, "features.denseblock3.denselayer25.norm1.running_var", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_conv1_weight, "features.denseblock3.denselayer25.conv1.weight", 128, 1024, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_weight, "features.denseblock3.denselayer25.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_bias, "features.denseblock3.denselayer25.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_running_mean, "features.denseblock3.denselayer25.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_running_var, "features.denseblock3.denselayer25.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer25_conv2_weight, "features.denseblock3.denselayer25.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_weight, "features.denseblock3.denselayer26.norm1.weight", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_bias, "features.denseblock3.denselayer26.norm1.bias", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_running_mean, "features.denseblock3.denselayer26.norm1.running_mean", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_running_var, "features.denseblock3.denselayer26.norm1.running_var", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_conv1_weight, "features.denseblock3.denselayer26.conv1.weight", 128, 1056, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_weight, "features.denseblock3.denselayer26.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_bias, "features.denseblock3.denselayer26.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_running_mean, "features.denseblock3.denselayer26.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_running_var, "features.denseblock3.denselayer26.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer26_conv2_weight, "features.denseblock3.denselayer26.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_weight, "features.denseblock3.denselayer27.norm1.weight", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_bias, "features.denseblock3.denselayer27.norm1.bias", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_running_mean, "features.denseblock3.denselayer27.norm1.running_mean", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_running_var, "features.denseblock3.denselayer27.norm1.running_var", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_conv1_weight, "features.denseblock3.denselayer27.conv1.weight", 128, 1088, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_weight, "features.denseblock3.denselayer27.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_bias, "features.denseblock3.denselayer27.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_running_mean, "features.denseblock3.denselayer27.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_running_var, "features.denseblock3.denselayer27.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer27_conv2_weight, "features.denseblock3.denselayer27.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_weight, "features.denseblock3.denselayer28.norm1.weight", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_bias, "features.denseblock3.denselayer28.norm1.bias", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_running_mean, "features.denseblock3.denselayer28.norm1.running_mean", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_running_var, "features.denseblock3.denselayer28.norm1.running_var", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_conv1_weight, "features.denseblock3.denselayer28.conv1.weight", 128, 1120, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_weight, "features.denseblock3.denselayer28.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_bias, "features.denseblock3.denselayer28.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_running_mean, "features.denseblock3.denselayer28.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_running_var, "features.denseblock3.denselayer28.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer28_conv2_weight, "features.denseblock3.denselayer28.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_weight, "features.denseblock3.denselayer29.norm1.weight", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_bias, "features.denseblock3.denselayer29.norm1.bias", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_running_mean, "features.denseblock3.denselayer29.norm1.running_mean", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_running_var, "features.denseblock3.denselayer29.norm1.running_var", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_conv1_weight, "features.denseblock3.denselayer29.conv1.weight", 128, 1152, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_weight, "features.denseblock3.denselayer29.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_bias, "features.denseblock3.denselayer29.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_running_mean, "features.denseblock3.denselayer29.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_running_var, "features.denseblock3.denselayer29.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer29_conv2_weight, "features.denseblock3.denselayer29.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_weight, "features.denseblock3.denselayer30.norm1.weight", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_bias, "features.denseblock3.denselayer30.norm1.bias", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_running_mean, "features.denseblock3.denselayer30.norm1.running_mean", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_running_var, "features.denseblock3.denselayer30.norm1.running_var", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_conv1_weight, "features.denseblock3.denselayer30.conv1.weight", 128, 1184, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_weight, "features.denseblock3.denselayer30.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_bias, "features.denseblock3.denselayer30.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_running_mean, "features.denseblock3.denselayer30.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_running_var, "features.denseblock3.denselayer30.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer30_conv2_weight, "features.denseblock3.denselayer30.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_weight, "features.denseblock3.denselayer31.norm1.weight", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_bias, "features.denseblock3.denselayer31.norm1.bias", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_running_mean, "features.denseblock3.denselayer31.norm1.running_mean", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_running_var, "features.denseblock3.denselayer31.norm1.running_var", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_conv1_weight, "features.denseblock3.denselayer31.conv1.weight", 128, 1216, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_weight, "features.denseblock3.denselayer31.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_bias, "features.denseblock3.denselayer31.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_running_mean, "features.denseblock3.denselayer31.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_running_var, "features.denseblock3.denselayer31.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer31_conv2_weight, "features.denseblock3.denselayer31.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_weight, "features.denseblock3.denselayer32.norm1.weight", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_bias, "features.denseblock3.denselayer32.norm1.bias", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_running_mean, "features.denseblock3.denselayer32.norm1.running_mean", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_running_var, "features.denseblock3.denselayer32.norm1.running_var", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_conv1_weight, "features.denseblock3.denselayer32.conv1.weight", 128, 1248, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_weight, "features.denseblock3.denselayer32.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_bias, "features.denseblock3.denselayer32.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_running_mean, "features.denseblock3.denselayer32.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_running_var, "features.denseblock3.denselayer32.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer32_conv2_weight, "features.denseblock3.denselayer32.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_transition3_norm_weight, "features.transition3.norm.weight", 1280);
	dict.load(ctx.features_transition3_norm_bias, "features.transition3.norm.bias", 1280);
	dict.load(ctx.features_transition3_norm_running_mean, "features.transition3.norm.running_mean", 1280);
	dict.load(ctx.features_transition3_norm_running_var, "features.transition3.norm.running_var", 1280);
	dict.load(ctx.features_transition3_conv_weight, "features.transition3.conv.weight", 640, 1280, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_weight, "features.denseblock4.denselayer1.norm1.weight", 640);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_bias, "features.denseblock4.denselayer1.norm1.bias", 640);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_running_mean, "features.denseblock4.denselayer1.norm1.running_mean", 640);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_running_var, "features.denseblock4.denselayer1.norm1.running_var", 640);
	dict.load(ctx.features_denseblock4_denselayer1_conv1_weight, "features.denseblock4.denselayer1.conv1.weight", 128, 640, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_weight, "features.denseblock4.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_bias, "features.denseblock4.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_running_mean, "features.denseblock4.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_running_var, "features.denseblock4.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer1_conv2_weight, "features.denseblock4.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_weight, "features.denseblock4.denselayer2.norm1.weight", 672);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_bias, "features.denseblock4.denselayer2.norm1.bias", 672);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_running_mean, "features.denseblock4.denselayer2.norm1.running_mean", 672);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_running_var, "features.denseblock4.denselayer2.norm1.running_var", 672);
	dict.load(ctx.features_denseblock4_denselayer2_conv1_weight, "features.denseblock4.denselayer2.conv1.weight", 128, 672, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_weight, "features.denseblock4.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_bias, "features.denseblock4.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_running_mean, "features.denseblock4.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_running_var, "features.denseblock4.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer2_conv2_weight, "features.denseblock4.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_weight, "features.denseblock4.denselayer3.norm1.weight", 704);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_bias, "features.denseblock4.denselayer3.norm1.bias", 704);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_running_mean, "features.denseblock4.denselayer3.norm1.running_mean", 704);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_running_var, "features.denseblock4.denselayer3.norm1.running_var", 704);
	dict.load(ctx.features_denseblock4_denselayer3_conv1_weight, "features.denseblock4.denselayer3.conv1.weight", 128, 704, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_weight, "features.denseblock4.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_bias, "features.denseblock4.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_running_mean, "features.denseblock4.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_running_var, "features.denseblock4.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer3_conv2_weight, "features.denseblock4.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_weight, "features.denseblock4.denselayer4.norm1.weight", 736);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_bias, "features.denseblock4.denselayer4.norm1.bias", 736);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_running_mean, "features.denseblock4.denselayer4.norm1.running_mean", 736);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_running_var, "features.denseblock4.denselayer4.norm1.running_var", 736);
	dict.load(ctx.features_denseblock4_denselayer4_conv1_weight, "features.denseblock4.denselayer4.conv1.weight", 128, 736, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_weight, "features.denseblock4.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_bias, "features.denseblock4.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_running_mean, "features.denseblock4.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_running_var, "features.denseblock4.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer4_conv2_weight, "features.denseblock4.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_weight, "features.denseblock4.denselayer5.norm1.weight", 768);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_bias, "features.denseblock4.denselayer5.norm1.bias", 768);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_running_mean, "features.denseblock4.denselayer5.norm1.running_mean", 768);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_running_var, "features.denseblock4.denselayer5.norm1.running_var", 768);
	dict.load(ctx.features_denseblock4_denselayer5_conv1_weight, "features.denseblock4.denselayer5.conv1.weight", 128, 768, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_weight, "features.denseblock4.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_bias, "features.denseblock4.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_running_mean, "features.denseblock4.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_running_var, "features.denseblock4.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer5_conv2_weight, "features.denseblock4.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_weight, "features.denseblock4.denselayer6.norm1.weight", 800);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_bias, "features.denseblock4.denselayer6.norm1.bias", 800);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_running_mean, "features.denseblock4.denselayer6.norm1.running_mean", 800);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_running_var, "features.denseblock4.denselayer6.norm1.running_var", 800);
	dict.load(ctx.features_denseblock4_denselayer6_conv1_weight, "features.denseblock4.denselayer6.conv1.weight", 128, 800, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_weight, "features.denseblock4.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_bias, "features.denseblock4.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_running_mean, "features.denseblock4.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_running_var, "features.denseblock4.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer6_conv2_weight, "features.denseblock4.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_weight, "features.denseblock4.denselayer7.norm1.weight", 832);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_bias, "features.denseblock4.denselayer7.norm1.bias", 832);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_running_mean, "features.denseblock4.denselayer7.norm1.running_mean", 832);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_running_var, "features.denseblock4.denselayer7.norm1.running_var", 832);
	dict.load(ctx.features_denseblock4_denselayer7_conv1_weight, "features.denseblock4.denselayer7.conv1.weight", 128, 832, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_weight, "features.denseblock4.denselayer7.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_bias, "features.denseblock4.denselayer7.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_running_mean, "features.denseblock4.denselayer7.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_running_var, "features.denseblock4.denselayer7.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer7_conv2_weight, "features.denseblock4.denselayer7.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_weight, "features.denseblock4.denselayer8.norm1.weight", 864);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_bias, "features.denseblock4.denselayer8.norm1.bias", 864);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_running_mean, "features.denseblock4.denselayer8.norm1.running_mean", 864);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_running_var, "features.denseblock4.denselayer8.norm1.running_var", 864);
	dict.load(ctx.features_denseblock4_denselayer8_conv1_weight, "features.denseblock4.denselayer8.conv1.weight", 128, 864, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_weight, "features.denseblock4.denselayer8.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_bias, "features.denseblock4.denselayer8.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_running_mean, "features.denseblock4.denselayer8.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_running_var, "features.denseblock4.denselayer8.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer8_conv2_weight, "features.denseblock4.denselayer8.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_weight, "features.denseblock4.denselayer9.norm1.weight", 896);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_bias, "features.denseblock4.denselayer9.norm1.bias", 896);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_running_mean, "features.denseblock4.denselayer9.norm1.running_mean", 896);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_running_var, "features.denseblock4.denselayer9.norm1.running_var", 896);
	dict.load(ctx.features_denseblock4_denselayer9_conv1_weight, "features.denseblock4.denselayer9.conv1.weight", 128, 896, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_weight, "features.denseblock4.denselayer9.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_bias, "features.denseblock4.denselayer9.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_running_mean, "features.denseblock4.denselayer9.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_running_var, "features.denseblock4.denselayer9.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer9_conv2_weight, "features.denseblock4.denselayer9.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_weight, "features.denseblock4.denselayer10.norm1.weight", 928);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_bias, "features.denseblock4.denselayer10.norm1.bias", 928);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_running_mean, "features.denseblock4.denselayer10.norm1.running_mean", 928);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_running_var, "features.denseblock4.denselayer10.norm1.running_var", 928);
	dict.load(ctx.features_denseblock4_denselayer10_conv1_weight, "features.denseblock4.denselayer10.conv1.weight", 128, 928, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_weight, "features.denseblock4.denselayer10.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_bias, "features.denseblock4.denselayer10.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_running_mean, "features.denseblock4.denselayer10.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_running_var, "features.denseblock4.denselayer10.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer10_conv2_weight, "features.denseblock4.denselayer10.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_weight, "features.denseblock4.denselayer11.norm1.weight", 960);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_bias, "features.denseblock4.denselayer11.norm1.bias", 960);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_running_mean, "features.denseblock4.denselayer11.norm1.running_mean", 960);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_running_var, "features.denseblock4.denselayer11.norm1.running_var", 960);
	dict.load(ctx.features_denseblock4_denselayer11_conv1_weight, "features.denseblock4.denselayer11.conv1.weight", 128, 960, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_weight, "features.denseblock4.denselayer11.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_bias, "features.denseblock4.denselayer11.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_running_mean, "features.denseblock4.denselayer11.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_running_var, "features.denseblock4.denselayer11.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer11_conv2_weight, "features.denseblock4.denselayer11.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_weight, "features.denseblock4.denselayer12.norm1.weight", 992);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_bias, "features.denseblock4.denselayer12.norm1.bias", 992);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_running_mean, "features.denseblock4.denselayer12.norm1.running_mean", 992);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_running_var, "features.denseblock4.denselayer12.norm1.running_var", 992);
	dict.load(ctx.features_denseblock4_denselayer12_conv1_weight, "features.denseblock4.denselayer12.conv1.weight", 128, 992, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_weight, "features.denseblock4.denselayer12.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_bias, "features.denseblock4.denselayer12.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_running_mean, "features.denseblock4.denselayer12.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_running_var, "features.denseblock4.denselayer12.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer12_conv2_weight, "features.denseblock4.denselayer12.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_weight, "features.denseblock4.denselayer13.norm1.weight", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_bias, "features.denseblock4.denselayer13.norm1.bias", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_running_mean, "features.denseblock4.denselayer13.norm1.running_mean", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_running_var, "features.denseblock4.denselayer13.norm1.running_var", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_conv1_weight, "features.denseblock4.denselayer13.conv1.weight", 128, 1024, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_weight, "features.denseblock4.denselayer13.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_bias, "features.denseblock4.denselayer13.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_running_mean, "features.denseblock4.denselayer13.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_running_var, "features.denseblock4.denselayer13.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer13_conv2_weight, "features.denseblock4.denselayer13.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_weight, "features.denseblock4.denselayer14.norm1.weight", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_bias, "features.denseblock4.denselayer14.norm1.bias", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_running_mean, "features.denseblock4.denselayer14.norm1.running_mean", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_running_var, "features.denseblock4.denselayer14.norm1.running_var", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_conv1_weight, "features.denseblock4.denselayer14.conv1.weight", 128, 1056, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_weight, "features.denseblock4.denselayer14.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_bias, "features.denseblock4.denselayer14.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_running_mean, "features.denseblock4.denselayer14.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_running_var, "features.denseblock4.denselayer14.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer14_conv2_weight, "features.denseblock4.denselayer14.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_weight, "features.denseblock4.denselayer15.norm1.weight", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_bias, "features.denseblock4.denselayer15.norm1.bias", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_running_mean, "features.denseblock4.denselayer15.norm1.running_mean", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_running_var, "features.denseblock4.denselayer15.norm1.running_var", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_conv1_weight, "features.denseblock4.denselayer15.conv1.weight", 128, 1088, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_weight, "features.denseblock4.denselayer15.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_bias, "features.denseblock4.denselayer15.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_running_mean, "features.denseblock4.denselayer15.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_running_var, "features.denseblock4.denselayer15.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer15_conv2_weight, "features.denseblock4.denselayer15.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_weight, "features.denseblock4.denselayer16.norm1.weight", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_bias, "features.denseblock4.denselayer16.norm1.bias", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_running_mean, "features.denseblock4.denselayer16.norm1.running_mean", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_running_var, "features.denseblock4.denselayer16.norm1.running_var", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_conv1_weight, "features.denseblock4.denselayer16.conv1.weight", 128, 1120, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_weight, "features.denseblock4.denselayer16.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_bias, "features.denseblock4.denselayer16.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_running_mean, "features.denseblock4.denselayer16.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_running_var, "features.denseblock4.denselayer16.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer16_conv2_weight, "features.denseblock4.denselayer16.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_weight, "features.denseblock4.denselayer17.norm1.weight", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_bias, "features.denseblock4.denselayer17.norm1.bias", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_running_mean, "features.denseblock4.denselayer17.norm1.running_mean", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_running_var, "features.denseblock4.denselayer17.norm1.running_var", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_conv1_weight, "features.denseblock4.denselayer17.conv1.weight", 128, 1152, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_weight, "features.denseblock4.denselayer17.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_bias, "features.denseblock4.denselayer17.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_running_mean, "features.denseblock4.denselayer17.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_running_var, "features.denseblock4.denselayer17.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer17_conv2_weight, "features.denseblock4.denselayer17.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_weight, "features.denseblock4.denselayer18.norm1.weight", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_bias, "features.denseblock4.denselayer18.norm1.bias", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_running_mean, "features.denseblock4.denselayer18.norm1.running_mean", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_running_var, "features.denseblock4.denselayer18.norm1.running_var", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_conv1_weight, "features.denseblock4.denselayer18.conv1.weight", 128, 1184, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_weight, "features.denseblock4.denselayer18.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_bias, "features.denseblock4.denselayer18.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_running_mean, "features.denseblock4.denselayer18.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_running_var, "features.denseblock4.denselayer18.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer18_conv2_weight, "features.denseblock4.denselayer18.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_weight, "features.denseblock4.denselayer19.norm1.weight", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_bias, "features.denseblock4.denselayer19.norm1.bias", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_running_mean, "features.denseblock4.denselayer19.norm1.running_mean", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_running_var, "features.denseblock4.denselayer19.norm1.running_var", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_conv1_weight, "features.denseblock4.denselayer19.conv1.weight", 128, 1216, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_weight, "features.denseblock4.denselayer19.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_bias, "features.denseblock4.denselayer19.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_running_mean, "features.denseblock4.denselayer19.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_running_var, "features.denseblock4.denselayer19.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer19_conv2_weight, "features.denseblock4.denselayer19.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_weight, "features.denseblock4.denselayer20.norm1.weight", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_bias, "features.denseblock4.denselayer20.norm1.bias", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_running_mean, "features.denseblock4.denselayer20.norm1.running_mean", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_running_var, "features.denseblock4.denselayer20.norm1.running_var", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_conv1_weight, "features.denseblock4.denselayer20.conv1.weight", 128, 1248, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_weight, "features.denseblock4.denselayer20.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_bias, "features.denseblock4.denselayer20.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_running_mean, "features.denseblock4.denselayer20.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_running_var, "features.denseblock4.denselayer20.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer20_conv2_weight, "features.denseblock4.denselayer20.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_weight, "features.denseblock4.denselayer21.norm1.weight", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_bias, "features.denseblock4.denselayer21.norm1.bias", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_running_mean, "features.denseblock4.denselayer21.norm1.running_mean", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_running_var, "features.denseblock4.denselayer21.norm1.running_var", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_conv1_weight, "features.denseblock4.denselayer21.conv1.weight", 128, 1280, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_weight, "features.denseblock4.denselayer21.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_bias, "features.denseblock4.denselayer21.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_running_mean, "features.denseblock4.denselayer21.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_running_var, "features.denseblock4.denselayer21.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer21_conv2_weight, "features.denseblock4.denselayer21.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_weight, "features.denseblock4.denselayer22.norm1.weight", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_bias, "features.denseblock4.denselayer22.norm1.bias", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_running_mean, "features.denseblock4.denselayer22.norm1.running_mean", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_running_var, "features.denseblock4.denselayer22.norm1.running_var", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_conv1_weight, "features.denseblock4.denselayer22.conv1.weight", 128, 1312, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_weight, "features.denseblock4.denselayer22.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_bias, "features.denseblock4.denselayer22.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_running_mean, "features.denseblock4.denselayer22.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_running_var, "features.denseblock4.denselayer22.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer22_conv2_weight, "features.denseblock4.denselayer22.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_weight, "features.denseblock4.denselayer23.norm1.weight", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_bias, "features.denseblock4.denselayer23.norm1.bias", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_running_mean, "features.denseblock4.denselayer23.norm1.running_mean", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_running_var, "features.denseblock4.denselayer23.norm1.running_var", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_conv1_weight, "features.denseblock4.denselayer23.conv1.weight", 128, 1344, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_weight, "features.denseblock4.denselayer23.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_bias, "features.denseblock4.denselayer23.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_running_mean, "features.denseblock4.denselayer23.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_running_var, "features.denseblock4.denselayer23.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer23_conv2_weight, "features.denseblock4.denselayer23.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_weight, "features.denseblock4.denselayer24.norm1.weight", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_bias, "features.denseblock4.denselayer24.norm1.bias", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_running_mean, "features.denseblock4.denselayer24.norm1.running_mean", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_running_var, "features.denseblock4.denselayer24.norm1.running_var", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_conv1_weight, "features.denseblock4.denselayer24.conv1.weight", 128, 1376, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_weight, "features.denseblock4.denselayer24.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_bias, "features.denseblock4.denselayer24.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_running_mean, "features.denseblock4.denselayer24.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_running_var, "features.denseblock4.denselayer24.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer24_conv2_weight, "features.denseblock4.denselayer24.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_weight, "features.denseblock4.denselayer25.norm1.weight", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_bias, "features.denseblock4.denselayer25.norm1.bias", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_running_mean, "features.denseblock4.denselayer25.norm1.running_mean", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_running_var, "features.denseblock4.denselayer25.norm1.running_var", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_conv1_weight, "features.denseblock4.denselayer25.conv1.weight", 128, 1408, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_weight, "features.denseblock4.denselayer25.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_bias, "features.denseblock4.denselayer25.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_running_mean, "features.denseblock4.denselayer25.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_running_var, "features.denseblock4.denselayer25.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer25_conv2_weight, "features.denseblock4.denselayer25.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_weight, "features.denseblock4.denselayer26.norm1.weight", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_bias, "features.denseblock4.denselayer26.norm1.bias", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_running_mean, "features.denseblock4.denselayer26.norm1.running_mean", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_running_var, "features.denseblock4.denselayer26.norm1.running_var", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_conv1_weight, "features.denseblock4.denselayer26.conv1.weight", 128, 1440, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_weight, "features.denseblock4.denselayer26.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_bias, "features.denseblock4.denselayer26.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_running_mean, "features.denseblock4.denselayer26.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_running_var, "features.denseblock4.denselayer26.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer26_conv2_weight, "features.denseblock4.denselayer26.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_weight, "features.denseblock4.denselayer27.norm1.weight", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_bias, "features.denseblock4.denselayer27.norm1.bias", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_running_mean, "features.denseblock4.denselayer27.norm1.running_mean", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_running_var, "features.denseblock4.denselayer27.norm1.running_var", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_conv1_weight, "features.denseblock4.denselayer27.conv1.weight", 128, 1472, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_weight, "features.denseblock4.denselayer27.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_bias, "features.denseblock4.denselayer27.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_running_mean, "features.denseblock4.denselayer27.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_running_var, "features.denseblock4.denselayer27.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer27_conv2_weight, "features.denseblock4.denselayer27.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_weight, "features.denseblock4.denselayer28.norm1.weight", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_bias, "features.denseblock4.denselayer28.norm1.bias", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_running_mean, "features.denseblock4.denselayer28.norm1.running_mean", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_running_var, "features.denseblock4.denselayer28.norm1.running_var", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_conv1_weight, "features.denseblock4.denselayer28.conv1.weight", 128, 1504, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_weight, "features.denseblock4.denselayer28.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_bias, "features.denseblock4.denselayer28.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_running_mean, "features.denseblock4.denselayer28.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_running_var, "features.denseblock4.denselayer28.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer28_conv2_weight, "features.denseblock4.denselayer28.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_weight, "features.denseblock4.denselayer29.norm1.weight", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_bias, "features.denseblock4.denselayer29.norm1.bias", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_running_mean, "features.denseblock4.denselayer29.norm1.running_mean", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_running_var, "features.denseblock4.denselayer29.norm1.running_var", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_conv1_weight, "features.denseblock4.denselayer29.conv1.weight", 128, 1536, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_weight, "features.denseblock4.denselayer29.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_bias, "features.denseblock4.denselayer29.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_running_mean, "features.denseblock4.denselayer29.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_running_var, "features.denseblock4.denselayer29.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer29_conv2_weight, "features.denseblock4.denselayer29.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_weight, "features.denseblock4.denselayer30.norm1.weight", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_bias, "features.denseblock4.denselayer30.norm1.bias", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_running_mean, "features.denseblock4.denselayer30.norm1.running_mean", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_running_var, "features.denseblock4.denselayer30.norm1.running_var", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_conv1_weight, "features.denseblock4.denselayer30.conv1.weight", 128, 1568, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_weight, "features.denseblock4.denselayer30.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_bias, "features.denseblock4.denselayer30.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_running_mean, "features.denseblock4.denselayer30.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_running_var, "features.denseblock4.denselayer30.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer30_conv2_weight, "features.denseblock4.denselayer30.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_weight, "features.denseblock4.denselayer31.norm1.weight", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_bias, "features.denseblock4.denselayer31.norm1.bias", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_running_mean, "features.denseblock4.denselayer31.norm1.running_mean", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_running_var, "features.denseblock4.denselayer31.norm1.running_var", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_conv1_weight, "features.denseblock4.denselayer31.conv1.weight", 128, 1600, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_weight, "features.denseblock4.denselayer31.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_bias, "features.denseblock4.denselayer31.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_running_mean, "features.denseblock4.denselayer31.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_running_var, "features.denseblock4.denselayer31.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer31_conv2_weight, "features.denseblock4.denselayer31.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_weight, "features.denseblock4.denselayer32.norm1.weight", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_bias, "features.denseblock4.denselayer32.norm1.bias", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_running_mean, "features.denseblock4.denselayer32.norm1.running_mean", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_running_var, "features.denseblock4.denselayer32.norm1.running_var", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_conv1_weight, "features.denseblock4.denselayer32.conv1.weight", 128, 1632, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_weight, "features.denseblock4.denselayer32.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_bias, "features.denseblock4.denselayer32.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_running_mean, "features.denseblock4.denselayer32.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_running_var, "features.denseblock4.denselayer32.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer32_conv2_weight, "features.denseblock4.denselayer32.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_norm5_weight, "features.norm5.weight", 1664);
	dict.load(ctx.features_norm5_bias, "features.norm5.bias", 1664);
	dict.load(ctx.features_norm5_running_mean, "features.norm5.running_mean", 1664);
	dict.load(ctx.features_norm5_running_var, "features.norm5.running_var", 1664);
	dict.load(ctx.classifier_weight, "classifier.weight", 1000, 1664);
	dict.load(ctx.classifier_bias, "classifier.bias", 1000);
	return ctx;
}


t4::tensor2f DenseNetForward(const DenseNet& ctx, t4::tensor4f x0)
{
	t4::tensor4f x847 = t4::Conv2d<7, 7, 2, 2, 3, 3, 1, 1>(x0, ctx.features_conv0_weight); //features.conv0
	t4::release(x0);
	t4::tensor4f x848 = t4::BatchNormalizationInplace(x847, ctx.features_norm0_weight, ctx.features_norm0_bias, ctx.features_norm0_running_mean, ctx.features_norm0_running_var, 1e-05f); //features.norm0
	t4::release(x847);
	t4::tensor4f x849 = t4::ReluInplace(x848); //features.relu0
	t4::release(x848);
	t4::tensor4f x850 = t4::MaxPool2d<3, 3, 2, 2, 1, 1>(x849); //features.pool0
	t4::release(x849);
	t4::tensor4f x851 = t4::BatchNormalization(x850, ctx.features_denseblock1_denselayer1_norm1_weight, ctx.features_denseblock1_denselayer1_norm1_bias, ctx.features_denseblock1_denselayer1_norm1_running_mean, ctx.features_denseblock1_denselayer1_norm1_running_var, 1e-05f); //features.denseblock1.denselayer1.norm1
	t4::tensor4f x852 = t4::ReluInplace(x851); //features.denseblock1.denselayer1.relu1
	t4::release(x851);
	t4::tensor4f x853 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x852, ctx.features_denseblock1_denselayer1_conv1_weight); //features.denseblock1.denselayer1.conv1
	t4::release(x852);
	t4::tensor4f x854 = t4::BatchNormalizationInplace(x853, ctx.features_denseblock1_denselayer1_norm2_weight, ctx.features_denseblock1_denselayer1_norm2_bias, ctx.features_denseblock1_denselayer1_norm2_running_mean, ctx.features_denseblock1_denselayer1_norm2_running_var, 1e-05f); //features.denseblock1.denselayer1.norm2
	t4::release(x853);
	t4::tensor4f x855 = t4::ReluInplace(x854); //features.denseblock1.denselayer1.relu2
	t4::release(x854);
	t4::tensor4f x856 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x855, ctx.features_denseblock1_denselayer1_conv2_weight); //features.denseblock1.denselayer1.conv2
	t4::release(x855);
	t4::tensor4f x857 = t4::Concat<1>(x850, x856); //features.denseblock1.denselayer1
	t4::release(x850, x856);
	t4::tensor4f x858 = t4::BatchNormalization(x857, ctx.features_denseblock1_denselayer2_norm1_weight, ctx.features_denseblock1_denselayer2_norm1_bias, ctx.features_denseblock1_denselayer2_norm1_running_mean, ctx.features_denseblock1_denselayer2_norm1_running_var, 1e-05f); //features.denseblock1.denselayer2.norm1
	t4::tensor4f x859 = t4::ReluInplace(x858); //features.denseblock1.denselayer2.relu1
	t4::release(x858);
	t4::tensor4f x860 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x859, ctx.features_denseblock1_denselayer2_conv1_weight); //features.denseblock1.denselayer2.conv1
	t4::release(x859);
	t4::tensor4f x861 = t4::BatchNormalizationInplace(x860, ctx.features_denseblock1_denselayer2_norm2_weight, ctx.features_denseblock1_denselayer2_norm2_bias, ctx.features_denseblock1_denselayer2_norm2_running_mean, ctx.features_denseblock1_denselayer2_norm2_running_var, 1e-05f); //features.denseblock1.denselayer2.norm2
	t4::release(x860);
	t4::tensor4f x862 = t4::ReluInplace(x861); //features.denseblock1.denselayer2.relu2
	t4::release(x861);
	t4::tensor4f x863 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x862, ctx.features_denseblock1_denselayer2_conv2_weight); //features.denseblock1.denselayer2.conv2
	t4::release(x862);
	t4::tensor4f x864 = t4::Concat<1>(x857, x863); //features.denseblock1.denselayer2
	t4::release(x857, x863);
	t4::tensor4f x865 = t4::BatchNormalization(x864, ctx.features_denseblock1_denselayer3_norm1_weight, ctx.features_denseblock1_denselayer3_norm1_bias, ctx.features_denseblock1_denselayer3_norm1_running_mean, ctx.features_denseblock1_denselayer3_norm1_running_var, 1e-05f); //features.denseblock1.denselayer3.norm1
	t4::tensor4f x866 = t4::ReluInplace(x865); //features.denseblock1.denselayer3.relu1
	t4::release(x865);
	t4::tensor4f x867 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x866, ctx.features_denseblock1_denselayer3_conv1_weight); //features.denseblock1.denselayer3.conv1
	t4::release(x866);
	t4::tensor4f x868 = t4::BatchNormalizationInplace(x867, ctx.features_denseblock1_denselayer3_norm2_weight, ctx.features_denseblock1_denselayer3_norm2_bias, ctx.features_denseblock1_denselayer3_norm2_running_mean, ctx.features_denseblock1_denselayer3_norm2_running_var, 1e-05f); //features.denseblock1.denselayer3.norm2
	t4::release(x867);
	t4::tensor4f x869 = t4::ReluInplace(x868); //features.denseblock1.denselayer3.relu2
	t4::release(x868);
	t4::tensor4f x870 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x869, ctx.features_denseblock1_denselayer3_conv2_weight); //features.denseblock1.denselayer3.conv2
	t4::release(x869);
	t4::tensor4f x871 = t4::Concat<1>(x864, x870); //features.denseblock1.denselayer3
	t4::release(x864, x870);
	t4::tensor4f x872 = t4::BatchNormalization(x871, ctx.features_denseblock1_denselayer4_norm1_weight, ctx.features_denseblock1_denselayer4_norm1_bias, ctx.features_denseblock1_denselayer4_norm1_running_mean, ctx.features_denseblock1_denselayer4_norm1_running_var, 1e-05f); //features.denseblock1.denselayer4.norm1
	t4::tensor4f x873 = t4::ReluInplace(x872); //features.denseblock1.denselayer4.relu1
	t4::release(x872);
	t4::tensor4f x874 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x873, ctx.features_denseblock1_denselayer4_conv1_weight); //features.denseblock1.denselayer4.conv1
	t4::release(x873);
	t4::tensor4f x875 = t4::BatchNormalizationInplace(x874, ctx.features_denseblock1_denselayer4_norm2_weight, ctx.features_denseblock1_denselayer4_norm2_bias, ctx.features_denseblock1_denselayer4_norm2_running_mean, ctx.features_denseblock1_denselayer4_norm2_running_var, 1e-05f); //features.denseblock1.denselayer4.norm2
	t4::release(x874);
	t4::tensor4f x876 = t4::ReluInplace(x875); //features.denseblock1.denselayer4.relu2
	t4::release(x875);
	t4::tensor4f x877 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x876, ctx.features_denseblock1_denselayer4_conv2_weight); //features.denseblock1.denselayer4.conv2
	t4::release(x876);
	t4::tensor4f x878 = t4::Concat<1>(x871, x877); //features.denseblock1.denselayer4
	t4::release(x871, x877);
	t4::tensor4f x879 = t4::BatchNormalization(x878, ctx.features_denseblock1_denselayer5_norm1_weight, ctx.features_denseblock1_denselayer5_norm1_bias, ctx.features_denseblock1_denselayer5_norm1_running_mean, ctx.features_denseblock1_denselayer5_norm1_running_var, 1e-05f); //features.denseblock1.denselayer5.norm1
	t4::tensor4f x880 = t4::ReluInplace(x879); //features.denseblock1.denselayer5.relu1
	t4::release(x879);
	t4::tensor4f x881 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x880, ctx.features_denseblock1_denselayer5_conv1_weight); //features.denseblock1.denselayer5.conv1
	t4::release(x880);
	t4::tensor4f x882 = t4::BatchNormalizationInplace(x881, ctx.features_denseblock1_denselayer5_norm2_weight, ctx.features_denseblock1_denselayer5_norm2_bias, ctx.features_denseblock1_denselayer5_norm2_running_mean, ctx.features_denseblock1_denselayer5_norm2_running_var, 1e-05f); //features.denseblock1.denselayer5.norm2
	t4::release(x881);
	t4::tensor4f x883 = t4::ReluInplace(x882); //features.denseblock1.denselayer5.relu2
	t4::release(x882);
	t4::tensor4f x884 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x883, ctx.features_denseblock1_denselayer5_conv2_weight); //features.denseblock1.denselayer5.conv2
	t4::release(x883);
	t4::tensor4f x885 = t4::Concat<1>(x878, x884); //features.denseblock1.denselayer5
	t4::release(x878, x884);
	t4::tensor4f x886 = t4::BatchNormalization(x885, ctx.features_denseblock1_denselayer6_norm1_weight, ctx.features_denseblock1_denselayer6_norm1_bias, ctx.features_denseblock1_denselayer6_norm1_running_mean, ctx.features_denseblock1_denselayer6_norm1_running_var, 1e-05f); //features.denseblock1.denselayer6.norm1
	t4::tensor4f x887 = t4::ReluInplace(x886); //features.denseblock1.denselayer6.relu1
	t4::release(x886);
	t4::tensor4f x888 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x887, ctx.features_denseblock1_denselayer6_conv1_weight); //features.denseblock1.denselayer6.conv1
	t4::release(x887);
	t4::tensor4f x889 = t4::BatchNormalizationInplace(x888, ctx.features_denseblock1_denselayer6_norm2_weight, ctx.features_denseblock1_denselayer6_norm2_bias, ctx.features_denseblock1_denselayer6_norm2_running_mean, ctx.features_denseblock1_denselayer6_norm2_running_var, 1e-05f); //features.denseblock1.denselayer6.norm2
	t4::release(x888);
	t4::tensor4f x890 = t4::ReluInplace(x889); //features.denseblock1.denselayer6.relu2
	t4::release(x889);
	t4::tensor4f x891 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x890, ctx.features_denseblock1_denselayer6_conv2_weight); //features.denseblock1.denselayer6.conv2
	t4::release(x890);
	t4::tensor4f x892 = t4::Concat<1>(x885, x891); //features.denseblock1.denselayer6
	t4::release(x885, x891);
	t4::tensor4f x893 = t4::BatchNormalizationInplace(x892, ctx.features_transition1_norm_weight, ctx.features_transition1_norm_bias, ctx.features_transition1_norm_running_mean, ctx.features_transition1_norm_running_var, 1e-05f); //features.transition1.norm
	t4::release(x892);
	t4::tensor4f x894 = t4::ReluInplace(x893); //features.transition1.relu
	t4::release(x893);
	t4::tensor4f x895 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x894, ctx.features_transition1_conv_weight); //features.transition1.conv
	t4::release(x894);
	t4::tensor4f x896 = t4::AveragePool2d<2, 2, 2, 2, 0, 0>(x895); //features.transition1.pool
	t4::release(x895);
	t4::tensor4f x897 = t4::BatchNormalization(x896, ctx.features_denseblock2_denselayer1_norm1_weight, ctx.features_denseblock2_denselayer1_norm1_bias, ctx.features_denseblock2_denselayer1_norm1_running_mean, ctx.features_denseblock2_denselayer1_norm1_running_var, 1e-05f); //features.denseblock2.denselayer1.norm1
	t4::tensor4f x898 = t4::ReluInplace(x897); //features.denseblock2.denselayer1.relu1
	t4::release(x897);
	t4::tensor4f x899 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x898, ctx.features_denseblock2_denselayer1_conv1_weight); //features.denseblock2.denselayer1.conv1
	t4::release(x898);
	t4::tensor4f x900 = t4::BatchNormalizationInplace(x899, ctx.features_denseblock2_denselayer1_norm2_weight, ctx.features_denseblock2_denselayer1_norm2_bias, ctx.features_denseblock2_denselayer1_norm2_running_mean, ctx.features_denseblock2_denselayer1_norm2_running_var, 1e-05f); //features.denseblock2.denselayer1.norm2
	t4::release(x899);
	t4::tensor4f x901 = t4::ReluInplace(x900); //features.denseblock2.denselayer1.relu2
	t4::release(x900);
	t4::tensor4f x902 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x901, ctx.features_denseblock2_denselayer1_conv2_weight); //features.denseblock2.denselayer1.conv2
	t4::release(x901);
	t4::tensor4f x903 = t4::Concat<1>(x896, x902); //features.denseblock2.denselayer1
	t4::release(x896, x902);
	t4::tensor4f x904 = t4::BatchNormalization(x903, ctx.features_denseblock2_denselayer2_norm1_weight, ctx.features_denseblock2_denselayer2_norm1_bias, ctx.features_denseblock2_denselayer2_norm1_running_mean, ctx.features_denseblock2_denselayer2_norm1_running_var, 1e-05f); //features.denseblock2.denselayer2.norm1
	t4::tensor4f x905 = t4::ReluInplace(x904); //features.denseblock2.denselayer2.relu1
	t4::release(x904);
	t4::tensor4f x906 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x905, ctx.features_denseblock2_denselayer2_conv1_weight); //features.denseblock2.denselayer2.conv1
	t4::release(x905);
	t4::tensor4f x907 = t4::BatchNormalizationInplace(x906, ctx.features_denseblock2_denselayer2_norm2_weight, ctx.features_denseblock2_denselayer2_norm2_bias, ctx.features_denseblock2_denselayer2_norm2_running_mean, ctx.features_denseblock2_denselayer2_norm2_running_var, 1e-05f); //features.denseblock2.denselayer2.norm2
	t4::release(x906);
	t4::tensor4f x908 = t4::ReluInplace(x907); //features.denseblock2.denselayer2.relu2
	t4::release(x907);
	t4::tensor4f x909 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x908, ctx.features_denseblock2_denselayer2_conv2_weight); //features.denseblock2.denselayer2.conv2
	t4::release(x908);
	t4::tensor4f x910 = t4::Concat<1>(x903, x909); //features.denseblock2.denselayer2
	t4::release(x903, x909);
	t4::tensor4f x911 = t4::BatchNormalization(x910, ctx.features_denseblock2_denselayer3_norm1_weight, ctx.features_denseblock2_denselayer3_norm1_bias, ctx.features_denseblock2_denselayer3_norm1_running_mean, ctx.features_denseblock2_denselayer3_norm1_running_var, 1e-05f); //features.denseblock2.denselayer3.norm1
	t4::tensor4f x912 = t4::ReluInplace(x911); //features.denseblock2.denselayer3.relu1
	t4::release(x911);
	t4::tensor4f x913 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x912, ctx.features_denseblock2_denselayer3_conv1_weight); //features.denseblock2.denselayer3.conv1
	t4::release(x912);
	t4::tensor4f x914 = t4::BatchNormalizationInplace(x913, ctx.features_denseblock2_denselayer3_norm2_weight, ctx.features_denseblock2_denselayer3_norm2_bias, ctx.features_denseblock2_denselayer3_norm2_running_mean, ctx.features_denseblock2_denselayer3_norm2_running_var, 1e-05f); //features.denseblock2.denselayer3.norm2
	t4::release(x913);
	t4::tensor4f x915 = t4::ReluInplace(x914); //features.denseblock2.denselayer3.relu2
	t4::release(x914);
	t4::tensor4f x916 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x915, ctx.features_denseblock2_denselayer3_conv2_weight); //features.denseblock2.denselayer3.conv2
	t4::release(x915);
	t4::tensor4f x917 = t4::Concat<1>(x910, x916); //features.denseblock2.denselayer3
	t4::release(x910, x916);
	t4::tensor4f x918 = t4::BatchNormalization(x917, ctx.features_denseblock2_denselayer4_norm1_weight, ctx.features_denseblock2_denselayer4_norm1_bias, ctx.features_denseblock2_denselayer4_norm1_running_mean, ctx.features_denseblock2_denselayer4_norm1_running_var, 1e-05f); //features.denseblock2.denselayer4.norm1
	t4::tensor4f x919 = t4::ReluInplace(x918); //features.denseblock2.denselayer4.relu1
	t4::release(x918);
	t4::tensor4f x920 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x919, ctx.features_denseblock2_denselayer4_conv1_weight); //features.denseblock2.denselayer4.conv1
	t4::release(x919);
	t4::tensor4f x921 = t4::BatchNormalizationInplace(x920, ctx.features_denseblock2_denselayer4_norm2_weight, ctx.features_denseblock2_denselayer4_norm2_bias, ctx.features_denseblock2_denselayer4_norm2_running_mean, ctx.features_denseblock2_denselayer4_norm2_running_var, 1e-05f); //features.denseblock2.denselayer4.norm2
	t4::release(x920);
	t4::tensor4f x922 = t4::ReluInplace(x921); //features.denseblock2.denselayer4.relu2
	t4::release(x921);
	t4::tensor4f x923 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x922, ctx.features_denseblock2_denselayer4_conv2_weight); //features.denseblock2.denselayer4.conv2
	t4::release(x922);
	t4::tensor4f x924 = t4::Concat<1>(x917, x923); //features.denseblock2.denselayer4
	t4::release(x917, x923);
	t4::tensor4f x925 = t4::BatchNormalization(x924, ctx.features_denseblock2_denselayer5_norm1_weight, ctx.features_denseblock2_denselayer5_norm1_bias, ctx.features_denseblock2_denselayer5_norm1_running_mean, ctx.features_denseblock2_denselayer5_norm1_running_var, 1e-05f); //features.denseblock2.denselayer5.norm1
	t4::tensor4f x926 = t4::ReluInplace(x925); //features.denseblock2.denselayer5.relu1
	t4::release(x925);
	t4::tensor4f x927 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x926, ctx.features_denseblock2_denselayer5_conv1_weight); //features.denseblock2.denselayer5.conv1
	t4::release(x926);
	t4::tensor4f x928 = t4::BatchNormalizationInplace(x927, ctx.features_denseblock2_denselayer5_norm2_weight, ctx.features_denseblock2_denselayer5_norm2_bias, ctx.features_denseblock2_denselayer5_norm2_running_mean, ctx.features_denseblock2_denselayer5_norm2_running_var, 1e-05f); //features.denseblock2.denselayer5.norm2
	t4::release(x927);
	t4::tensor4f x929 = t4::ReluInplace(x928); //features.denseblock2.denselayer5.relu2
	t4::release(x928);
	t4::tensor4f x930 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x929, ctx.features_denseblock2_denselayer5_conv2_weight); //features.denseblock2.denselayer5.conv2
	t4::release(x929);
	t4::tensor4f x931 = t4::Concat<1>(x924, x930); //features.denseblock2.denselayer5
	t4::release(x924, x930);
	t4::tensor4f x932 = t4::BatchNormalization(x931, ctx.features_denseblock2_denselayer6_norm1_weight, ctx.features_denseblock2_denselayer6_norm1_bias, ctx.features_denseblock2_denselayer6_norm1_running_mean, ctx.features_denseblock2_denselayer6_norm1_running_var, 1e-05f); //features.denseblock2.denselayer6.norm1
	t4::tensor4f x933 = t4::ReluInplace(x932); //features.denseblock2.denselayer6.relu1
	t4::release(x932);
	t4::tensor4f x934 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x933, ctx.features_denseblock2_denselayer6_conv1_weight); //features.denseblock2.denselayer6.conv1
	t4::release(x933);
	t4::tensor4f x935 = t4::BatchNormalizationInplace(x934, ctx.features_denseblock2_denselayer6_norm2_weight, ctx.features_denseblock2_denselayer6_norm2_bias, ctx.features_denseblock2_denselayer6_norm2_running_mean, ctx.features_denseblock2_denselayer6_norm2_running_var, 1e-05f); //features.denseblock2.denselayer6.norm2
	t4::release(x934);
	t4::tensor4f x936 = t4::ReluInplace(x935); //features.denseblock2.denselayer6.relu2
	t4::release(x935);
	t4::tensor4f x937 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x936, ctx.features_denseblock2_denselayer6_conv2_weight); //features.denseblock2.denselayer6.conv2
	t4::release(x936);
	t4::tensor4f x938 = t4::Concat<1>(x931, x937); //features.denseblock2.denselayer6
	t4::release(x931, x937);
	t4::tensor4f x939 = t4::BatchNormalization(x938, ctx.features_denseblock2_denselayer7_norm1_weight, ctx.features_denseblock2_denselayer7_norm1_bias, ctx.features_denseblock2_denselayer7_norm1_running_mean, ctx.features_denseblock2_denselayer7_norm1_running_var, 1e-05f); //features.denseblock2.denselayer7.norm1
	t4::tensor4f x940 = t4::ReluInplace(x939); //features.denseblock2.denselayer7.relu1
	t4::release(x939);
	t4::tensor4f x941 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x940, ctx.features_denseblock2_denselayer7_conv1_weight); //features.denseblock2.denselayer7.conv1
	t4::release(x940);
	t4::tensor4f x942 = t4::BatchNormalizationInplace(x941, ctx.features_denseblock2_denselayer7_norm2_weight, ctx.features_denseblock2_denselayer7_norm2_bias, ctx.features_denseblock2_denselayer7_norm2_running_mean, ctx.features_denseblock2_denselayer7_norm2_running_var, 1e-05f); //features.denseblock2.denselayer7.norm2
	t4::release(x941);
	t4::tensor4f x943 = t4::ReluInplace(x942); //features.denseblock2.denselayer7.relu2
	t4::release(x942);
	t4::tensor4f x944 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x943, ctx.features_denseblock2_denselayer7_conv2_weight); //features.denseblock2.denselayer7.conv2
	t4::release(x943);
	t4::tensor4f x945 = t4::Concat<1>(x938, x944); //features.denseblock2.denselayer7
	t4::release(x938, x944);
	t4::tensor4f x946 = t4::BatchNormalization(x945, ctx.features_denseblock2_denselayer8_norm1_weight, ctx.features_denseblock2_denselayer8_norm1_bias, ctx.features_denseblock2_denselayer8_norm1_running_mean, ctx.features_denseblock2_denselayer8_norm1_running_var, 1e-05f); //features.denseblock2.denselayer8.norm1
	t4::tensor4f x947 = t4::ReluInplace(x946); //features.denseblock2.denselayer8.relu1
	t4::release(x946);
	t4::tensor4f x948 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x947, ctx.features_denseblock2_denselayer8_conv1_weight); //features.denseblock2.denselayer8.conv1
	t4::release(x947);
	t4::tensor4f x949 = t4::BatchNormalizationInplace(x948, ctx.features_denseblock2_denselayer8_norm2_weight, ctx.features_denseblock2_denselayer8_norm2_bias, ctx.features_denseblock2_denselayer8_norm2_running_mean, ctx.features_denseblock2_denselayer8_norm2_running_var, 1e-05f); //features.denseblock2.denselayer8.norm2
	t4::release(x948);
	t4::tensor4f x950 = t4::ReluInplace(x949); //features.denseblock2.denselayer8.relu2
	t4::release(x949);
	t4::tensor4f x951 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x950, ctx.features_denseblock2_denselayer8_conv2_weight); //features.denseblock2.denselayer8.conv2
	t4::release(x950);
	t4::tensor4f x952 = t4::Concat<1>(x945, x951); //features.denseblock2.denselayer8
	t4::release(x945, x951);
	t4::tensor4f x953 = t4::BatchNormalization(x952, ctx.features_denseblock2_denselayer9_norm1_weight, ctx.features_denseblock2_denselayer9_norm1_bias, ctx.features_denseblock2_denselayer9_norm1_running_mean, ctx.features_denseblock2_denselayer9_norm1_running_var, 1e-05f); //features.denseblock2.denselayer9.norm1
	t4::tensor4f x954 = t4::ReluInplace(x953); //features.denseblock2.denselayer9.relu1
	t4::release(x953);
	t4::tensor4f x955 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x954, ctx.features_denseblock2_denselayer9_conv1_weight); //features.denseblock2.denselayer9.conv1
	t4::release(x954);
	t4::tensor4f x956 = t4::BatchNormalizationInplace(x955, ctx.features_denseblock2_denselayer9_norm2_weight, ctx.features_denseblock2_denselayer9_norm2_bias, ctx.features_denseblock2_denselayer9_norm2_running_mean, ctx.features_denseblock2_denselayer9_norm2_running_var, 1e-05f); //features.denseblock2.denselayer9.norm2
	t4::release(x955);
	t4::tensor4f x957 = t4::ReluInplace(x956); //features.denseblock2.denselayer9.relu2
	t4::release(x956);
	t4::tensor4f x958 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x957, ctx.features_denseblock2_denselayer9_conv2_weight); //features.denseblock2.denselayer9.conv2
	t4::release(x957);
	t4::tensor4f x959 = t4::Concat<1>(x952, x958); //features.denseblock2.denselayer9
	t4::release(x952, x958);
	t4::tensor4f x960 = t4::BatchNormalization(x959, ctx.features_denseblock2_denselayer10_norm1_weight, ctx.features_denseblock2_denselayer10_norm1_bias, ctx.features_denseblock2_denselayer10_norm1_running_mean, ctx.features_denseblock2_denselayer10_norm1_running_var, 1e-05f); //features.denseblock2.denselayer10.norm1
	t4::tensor4f x961 = t4::ReluInplace(x960); //features.denseblock2.denselayer10.relu1
	t4::release(x960);
	t4::tensor4f x962 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x961, ctx.features_denseblock2_denselayer10_conv1_weight); //features.denseblock2.denselayer10.conv1
	t4::release(x961);
	t4::tensor4f x963 = t4::BatchNormalizationInplace(x962, ctx.features_denseblock2_denselayer10_norm2_weight, ctx.features_denseblock2_denselayer10_norm2_bias, ctx.features_denseblock2_denselayer10_norm2_running_mean, ctx.features_denseblock2_denselayer10_norm2_running_var, 1e-05f); //features.denseblock2.denselayer10.norm2
	t4::release(x962);
	t4::tensor4f x964 = t4::ReluInplace(x963); //features.denseblock2.denselayer10.relu2
	t4::release(x963);
	t4::tensor4f x965 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x964, ctx.features_denseblock2_denselayer10_conv2_weight); //features.denseblock2.denselayer10.conv2
	t4::release(x964);
	t4::tensor4f x966 = t4::Concat<1>(x959, x965); //features.denseblock2.denselayer10
	t4::release(x959, x965);
	t4::tensor4f x967 = t4::BatchNormalization(x966, ctx.features_denseblock2_denselayer11_norm1_weight, ctx.features_denseblock2_denselayer11_norm1_bias, ctx.features_denseblock2_denselayer11_norm1_running_mean, ctx.features_denseblock2_denselayer11_norm1_running_var, 1e-05f); //features.denseblock2.denselayer11.norm1
	t4::tensor4f x968 = t4::ReluInplace(x967); //features.denseblock2.denselayer11.relu1
	t4::release(x967);
	t4::tensor4f x969 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x968, ctx.features_denseblock2_denselayer11_conv1_weight); //features.denseblock2.denselayer11.conv1
	t4::release(x968);
	t4::tensor4f x970 = t4::BatchNormalizationInplace(x969, ctx.features_denseblock2_denselayer11_norm2_weight, ctx.features_denseblock2_denselayer11_norm2_bias, ctx.features_denseblock2_denselayer11_norm2_running_mean, ctx.features_denseblock2_denselayer11_norm2_running_var, 1e-05f); //features.denseblock2.denselayer11.norm2
	t4::release(x969);
	t4::tensor4f x971 = t4::ReluInplace(x970); //features.denseblock2.denselayer11.relu2
	t4::release(x970);
	t4::tensor4f x972 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x971, ctx.features_denseblock2_denselayer11_conv2_weight); //features.denseblock2.denselayer11.conv2
	t4::release(x971);
	t4::tensor4f x973 = t4::Concat<1>(x966, x972); //features.denseblock2.denselayer11
	t4::release(x966, x972);
	t4::tensor4f x974 = t4::BatchNormalization(x973, ctx.features_denseblock2_denselayer12_norm1_weight, ctx.features_denseblock2_denselayer12_norm1_bias, ctx.features_denseblock2_denselayer12_norm1_running_mean, ctx.features_denseblock2_denselayer12_norm1_running_var, 1e-05f); //features.denseblock2.denselayer12.norm1
	t4::tensor4f x975 = t4::ReluInplace(x974); //features.denseblock2.denselayer12.relu1
	t4::release(x974);
	t4::tensor4f x976 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x975, ctx.features_denseblock2_denselayer12_conv1_weight); //features.denseblock2.denselayer12.conv1
	t4::release(x975);
	t4::tensor4f x977 = t4::BatchNormalizationInplace(x976, ctx.features_denseblock2_denselayer12_norm2_weight, ctx.features_denseblock2_denselayer12_norm2_bias, ctx.features_denseblock2_denselayer12_norm2_running_mean, ctx.features_denseblock2_denselayer12_norm2_running_var, 1e-05f); //features.denseblock2.denselayer12.norm2
	t4::release(x976);
	t4::tensor4f x978 = t4::ReluInplace(x977); //features.denseblock2.denselayer12.relu2
	t4::release(x977);
	t4::tensor4f x979 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x978, ctx.features_denseblock2_denselayer12_conv2_weight); //features.denseblock2.denselayer12.conv2
	t4::release(x978);
	t4::tensor4f x980 = t4::Concat<1>(x973, x979); //features.denseblock2.denselayer12
	t4::release(x973, x979);
	t4::tensor4f x981 = t4::BatchNormalizationInplace(x980, ctx.features_transition2_norm_weight, ctx.features_transition2_norm_bias, ctx.features_transition2_norm_running_mean, ctx.features_transition2_norm_running_var, 1e-05f); //features.transition2.norm
	t4::release(x980);
	t4::tensor4f x982 = t4::ReluInplace(x981); //features.transition2.relu
	t4::release(x981);
	t4::tensor4f x983 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x982, ctx.features_transition2_conv_weight); //features.transition2.conv
	t4::release(x982);
	t4::tensor4f x984 = t4::AveragePool2d<2, 2, 2, 2, 0, 0>(x983); //features.transition2.pool
	t4::release(x983);
	t4::tensor4f x985 = t4::BatchNormalization(x984, ctx.features_denseblock3_denselayer1_norm1_weight, ctx.features_denseblock3_denselayer1_norm1_bias, ctx.features_denseblock3_denselayer1_norm1_running_mean, ctx.features_denseblock3_denselayer1_norm1_running_var, 1e-05f); //features.denseblock3.denselayer1.norm1
	t4::tensor4f x986 = t4::ReluInplace(x985); //features.denseblock3.denselayer1.relu1
	t4::release(x985);
	t4::tensor4f x987 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x986, ctx.features_denseblock3_denselayer1_conv1_weight); //features.denseblock3.denselayer1.conv1
	t4::release(x986);
	t4::tensor4f x988 = t4::BatchNormalizationInplace(x987, ctx.features_denseblock3_denselayer1_norm2_weight, ctx.features_denseblock3_denselayer1_norm2_bias, ctx.features_denseblock3_denselayer1_norm2_running_mean, ctx.features_denseblock3_denselayer1_norm2_running_var, 1e-05f); //features.denseblock3.denselayer1.norm2
	t4::release(x987);
	t4::tensor4f x989 = t4::ReluInplace(x988); //features.denseblock3.denselayer1.relu2
	t4::release(x988);
	t4::tensor4f x990 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x989, ctx.features_denseblock3_denselayer1_conv2_weight); //features.denseblock3.denselayer1.conv2
	t4::release(x989);
	t4::tensor4f x991 = t4::Concat<1>(x984, x990); //features.denseblock3.denselayer1
	t4::release(x984, x990);
	t4::tensor4f x992 = t4::BatchNormalization(x991, ctx.features_denseblock3_denselayer2_norm1_weight, ctx.features_denseblock3_denselayer2_norm1_bias, ctx.features_denseblock3_denselayer2_norm1_running_mean, ctx.features_denseblock3_denselayer2_norm1_running_var, 1e-05f); //features.denseblock3.denselayer2.norm1
	t4::tensor4f x993 = t4::ReluInplace(x992); //features.denseblock3.denselayer2.relu1
	t4::release(x992);
	t4::tensor4f x994 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x993, ctx.features_denseblock3_denselayer2_conv1_weight); //features.denseblock3.denselayer2.conv1
	t4::release(x993);
	t4::tensor4f x995 = t4::BatchNormalizationInplace(x994, ctx.features_denseblock3_denselayer2_norm2_weight, ctx.features_denseblock3_denselayer2_norm2_bias, ctx.features_denseblock3_denselayer2_norm2_running_mean, ctx.features_denseblock3_denselayer2_norm2_running_var, 1e-05f); //features.denseblock3.denselayer2.norm2
	t4::release(x994);
	t4::tensor4f x996 = t4::ReluInplace(x995); //features.denseblock3.denselayer2.relu2
	t4::release(x995);
	t4::tensor4f x997 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x996, ctx.features_denseblock3_denselayer2_conv2_weight); //features.denseblock3.denselayer2.conv2
	t4::release(x996);
	t4::tensor4f x998 = t4::Concat<1>(x991, x997); //features.denseblock3.denselayer2
	t4::release(x991, x997);
	t4::tensor4f x999 = t4::BatchNormalization(x998, ctx.features_denseblock3_denselayer3_norm1_weight, ctx.features_denseblock3_denselayer3_norm1_bias, ctx.features_denseblock3_denselayer3_norm1_running_mean, ctx.features_denseblock3_denselayer3_norm1_running_var, 1e-05f); //features.denseblock3.denselayer3.norm1
	t4::tensor4f x1000 = t4::ReluInplace(x999); //features.denseblock3.denselayer3.relu1
	t4::release(x999);
	t4::tensor4f x1001 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1000, ctx.features_denseblock3_denselayer3_conv1_weight); //features.denseblock3.denselayer3.conv1
	t4::release(x1000);
	t4::tensor4f x1002 = t4::BatchNormalizationInplace(x1001, ctx.features_denseblock3_denselayer3_norm2_weight, ctx.features_denseblock3_denselayer3_norm2_bias, ctx.features_denseblock3_denselayer3_norm2_running_mean, ctx.features_denseblock3_denselayer3_norm2_running_var, 1e-05f); //features.denseblock3.denselayer3.norm2
	t4::release(x1001);
	t4::tensor4f x1003 = t4::ReluInplace(x1002); //features.denseblock3.denselayer3.relu2
	t4::release(x1002);
	t4::tensor4f x1004 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1003, ctx.features_denseblock3_denselayer3_conv2_weight); //features.denseblock3.denselayer3.conv2
	t4::release(x1003);
	t4::tensor4f x1005 = t4::Concat<1>(x998, x1004); //features.denseblock3.denselayer3
	t4::release(x998, x1004);
	t4::tensor4f x1006 = t4::BatchNormalization(x1005, ctx.features_denseblock3_denselayer4_norm1_weight, ctx.features_denseblock3_denselayer4_norm1_bias, ctx.features_denseblock3_denselayer4_norm1_running_mean, ctx.features_denseblock3_denselayer4_norm1_running_var, 1e-05f); //features.denseblock3.denselayer4.norm1
	t4::tensor4f x1007 = t4::ReluInplace(x1006); //features.denseblock3.denselayer4.relu1
	t4::release(x1006);
	t4::tensor4f x1008 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1007, ctx.features_denseblock3_denselayer4_conv1_weight); //features.denseblock3.denselayer4.conv1
	t4::release(x1007);
	t4::tensor4f x1009 = t4::BatchNormalizationInplace(x1008, ctx.features_denseblock3_denselayer4_norm2_weight, ctx.features_denseblock3_denselayer4_norm2_bias, ctx.features_denseblock3_denselayer4_norm2_running_mean, ctx.features_denseblock3_denselayer4_norm2_running_var, 1e-05f); //features.denseblock3.denselayer4.norm2
	t4::release(x1008);
	t4::tensor4f x1010 = t4::ReluInplace(x1009); //features.denseblock3.denselayer4.relu2
	t4::release(x1009);
	t4::tensor4f x1011 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1010, ctx.features_denseblock3_denselayer4_conv2_weight); //features.denseblock3.denselayer4.conv2
	t4::release(x1010);
	t4::tensor4f x1012 = t4::Concat<1>(x1005, x1011); //features.denseblock3.denselayer4
	t4::release(x1005, x1011);
	t4::tensor4f x1013 = t4::BatchNormalization(x1012, ctx.features_denseblock3_denselayer5_norm1_weight, ctx.features_denseblock3_denselayer5_norm1_bias, ctx.features_denseblock3_denselayer5_norm1_running_mean, ctx.features_denseblock3_denselayer5_norm1_running_var, 1e-05f); //features.denseblock3.denselayer5.norm1
	t4::tensor4f x1014 = t4::ReluInplace(x1013); //features.denseblock3.denselayer5.relu1
	t4::release(x1013);
	t4::tensor4f x1015 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1014, ctx.features_denseblock3_denselayer5_conv1_weight); //features.denseblock3.denselayer5.conv1
	t4::release(x1014);
	t4::tensor4f x1016 = t4::BatchNormalizationInplace(x1015, ctx.features_denseblock3_denselayer5_norm2_weight, ctx.features_denseblock3_denselayer5_norm2_bias, ctx.features_denseblock3_denselayer5_norm2_running_mean, ctx.features_denseblock3_denselayer5_norm2_running_var, 1e-05f); //features.denseblock3.denselayer5.norm2
	t4::release(x1015);
	t4::tensor4f x1017 = t4::ReluInplace(x1016); //features.denseblock3.denselayer5.relu2
	t4::release(x1016);
	t4::tensor4f x1018 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1017, ctx.features_denseblock3_denselayer5_conv2_weight); //features.denseblock3.denselayer5.conv2
	t4::release(x1017);
	t4::tensor4f x1019 = t4::Concat<1>(x1012, x1018); //features.denseblock3.denselayer5
	t4::release(x1012, x1018);
	t4::tensor4f x1020 = t4::BatchNormalization(x1019, ctx.features_denseblock3_denselayer6_norm1_weight, ctx.features_denseblock3_denselayer6_norm1_bias, ctx.features_denseblock3_denselayer6_norm1_running_mean, ctx.features_denseblock3_denselayer6_norm1_running_var, 1e-05f); //features.denseblock3.denselayer6.norm1
	t4::tensor4f x1021 = t4::ReluInplace(x1020); //features.denseblock3.denselayer6.relu1
	t4::release(x1020);
	t4::tensor4f x1022 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1021, ctx.features_denseblock3_denselayer6_conv1_weight); //features.denseblock3.denselayer6.conv1
	t4::release(x1021);
	t4::tensor4f x1023 = t4::BatchNormalizationInplace(x1022, ctx.features_denseblock3_denselayer6_norm2_weight, ctx.features_denseblock3_denselayer6_norm2_bias, ctx.features_denseblock3_denselayer6_norm2_running_mean, ctx.features_denseblock3_denselayer6_norm2_running_var, 1e-05f); //features.denseblock3.denselayer6.norm2
	t4::release(x1022);
	t4::tensor4f x1024 = t4::ReluInplace(x1023); //features.denseblock3.denselayer6.relu2
	t4::release(x1023);
	t4::tensor4f x1025 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1024, ctx.features_denseblock3_denselayer6_conv2_weight); //features.denseblock3.denselayer6.conv2
	t4::release(x1024);
	t4::tensor4f x1026 = t4::Concat<1>(x1019, x1025); //features.denseblock3.denselayer6
	t4::release(x1019, x1025);
	t4::tensor4f x1027 = t4::BatchNormalization(x1026, ctx.features_denseblock3_denselayer7_norm1_weight, ctx.features_denseblock3_denselayer7_norm1_bias, ctx.features_denseblock3_denselayer7_norm1_running_mean, ctx.features_denseblock3_denselayer7_norm1_running_var, 1e-05f); //features.denseblock3.denselayer7.norm1
	t4::tensor4f x1028 = t4::ReluInplace(x1027); //features.denseblock3.denselayer7.relu1
	t4::release(x1027);
	t4::tensor4f x1029 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1028, ctx.features_denseblock3_denselayer7_conv1_weight); //features.denseblock3.denselayer7.conv1
	t4::release(x1028);
	t4::tensor4f x1030 = t4::BatchNormalizationInplace(x1029, ctx.features_denseblock3_denselayer7_norm2_weight, ctx.features_denseblock3_denselayer7_norm2_bias, ctx.features_denseblock3_denselayer7_norm2_running_mean, ctx.features_denseblock3_denselayer7_norm2_running_var, 1e-05f); //features.denseblock3.denselayer7.norm2
	t4::release(x1029);
	t4::tensor4f x1031 = t4::ReluInplace(x1030); //features.denseblock3.denselayer7.relu2
	t4::release(x1030);
	t4::tensor4f x1032 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1031, ctx.features_denseblock3_denselayer7_conv2_weight); //features.denseblock3.denselayer7.conv2
	t4::release(x1031);
	t4::tensor4f x1033 = t4::Concat<1>(x1026, x1032); //features.denseblock3.denselayer7
	t4::release(x1026, x1032);
	t4::tensor4f x1034 = t4::BatchNormalization(x1033, ctx.features_denseblock3_denselayer8_norm1_weight, ctx.features_denseblock3_denselayer8_norm1_bias, ctx.features_denseblock3_denselayer8_norm1_running_mean, ctx.features_denseblock3_denselayer8_norm1_running_var, 1e-05f); //features.denseblock3.denselayer8.norm1
	t4::tensor4f x1035 = t4::ReluInplace(x1034); //features.denseblock3.denselayer8.relu1
	t4::release(x1034);
	t4::tensor4f x1036 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1035, ctx.features_denseblock3_denselayer8_conv1_weight); //features.denseblock3.denselayer8.conv1
	t4::release(x1035);
	t4::tensor4f x1037 = t4::BatchNormalizationInplace(x1036, ctx.features_denseblock3_denselayer8_norm2_weight, ctx.features_denseblock3_denselayer8_norm2_bias, ctx.features_denseblock3_denselayer8_norm2_running_mean, ctx.features_denseblock3_denselayer8_norm2_running_var, 1e-05f); //features.denseblock3.denselayer8.norm2
	t4::release(x1036);
	t4::tensor4f x1038 = t4::ReluInplace(x1037); //features.denseblock3.denselayer8.relu2
	t4::release(x1037);
	t4::tensor4f x1039 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1038, ctx.features_denseblock3_denselayer8_conv2_weight); //features.denseblock3.denselayer8.conv2
	t4::release(x1038);
	t4::tensor4f x1040 = t4::Concat<1>(x1033, x1039); //features.denseblock3.denselayer8
	t4::release(x1033, x1039);
	t4::tensor4f x1041 = t4::BatchNormalization(x1040, ctx.features_denseblock3_denselayer9_norm1_weight, ctx.features_denseblock3_denselayer9_norm1_bias, ctx.features_denseblock3_denselayer9_norm1_running_mean, ctx.features_denseblock3_denselayer9_norm1_running_var, 1e-05f); //features.denseblock3.denselayer9.norm1
	t4::tensor4f x1042 = t4::ReluInplace(x1041); //features.denseblock3.denselayer9.relu1
	t4::release(x1041);
	t4::tensor4f x1043 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1042, ctx.features_denseblock3_denselayer9_conv1_weight); //features.denseblock3.denselayer9.conv1
	t4::release(x1042);
	t4::tensor4f x1044 = t4::BatchNormalizationInplace(x1043, ctx.features_denseblock3_denselayer9_norm2_weight, ctx.features_denseblock3_denselayer9_norm2_bias, ctx.features_denseblock3_denselayer9_norm2_running_mean, ctx.features_denseblock3_denselayer9_norm2_running_var, 1e-05f); //features.denseblock3.denselayer9.norm2
	t4::release(x1043);
	t4::tensor4f x1045 = t4::ReluInplace(x1044); //features.denseblock3.denselayer9.relu2
	t4::release(x1044);
	t4::tensor4f x1046 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1045, ctx.features_denseblock3_denselayer9_conv2_weight); //features.denseblock3.denselayer9.conv2
	t4::release(x1045);
	t4::tensor4f x1047 = t4::Concat<1>(x1040, x1046); //features.denseblock3.denselayer9
	t4::release(x1040, x1046);
	t4::tensor4f x1048 = t4::BatchNormalization(x1047, ctx.features_denseblock3_denselayer10_norm1_weight, ctx.features_denseblock3_denselayer10_norm1_bias, ctx.features_denseblock3_denselayer10_norm1_running_mean, ctx.features_denseblock3_denselayer10_norm1_running_var, 1e-05f); //features.denseblock3.denselayer10.norm1
	t4::tensor4f x1049 = t4::ReluInplace(x1048); //features.denseblock3.denselayer10.relu1
	t4::release(x1048);
	t4::tensor4f x1050 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1049, ctx.features_denseblock3_denselayer10_conv1_weight); //features.denseblock3.denselayer10.conv1
	t4::release(x1049);
	t4::tensor4f x1051 = t4::BatchNormalizationInplace(x1050, ctx.features_denseblock3_denselayer10_norm2_weight, ctx.features_denseblock3_denselayer10_norm2_bias, ctx.features_denseblock3_denselayer10_norm2_running_mean, ctx.features_denseblock3_denselayer10_norm2_running_var, 1e-05f); //features.denseblock3.denselayer10.norm2
	t4::release(x1050);
	t4::tensor4f x1052 = t4::ReluInplace(x1051); //features.denseblock3.denselayer10.relu2
	t4::release(x1051);
	t4::tensor4f x1053 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1052, ctx.features_denseblock3_denselayer10_conv2_weight); //features.denseblock3.denselayer10.conv2
	t4::release(x1052);
	t4::tensor4f x1054 = t4::Concat<1>(x1047, x1053); //features.denseblock3.denselayer10
	t4::release(x1047, x1053);
	t4::tensor4f x1055 = t4::BatchNormalization(x1054, ctx.features_denseblock3_denselayer11_norm1_weight, ctx.features_denseblock3_denselayer11_norm1_bias, ctx.features_denseblock3_denselayer11_norm1_running_mean, ctx.features_denseblock3_denselayer11_norm1_running_var, 1e-05f); //features.denseblock3.denselayer11.norm1
	t4::tensor4f x1056 = t4::ReluInplace(x1055); //features.denseblock3.denselayer11.relu1
	t4::release(x1055);
	t4::tensor4f x1057 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1056, ctx.features_denseblock3_denselayer11_conv1_weight); //features.denseblock3.denselayer11.conv1
	t4::release(x1056);
	t4::tensor4f x1058 = t4::BatchNormalizationInplace(x1057, ctx.features_denseblock3_denselayer11_norm2_weight, ctx.features_denseblock3_denselayer11_norm2_bias, ctx.features_denseblock3_denselayer11_norm2_running_mean, ctx.features_denseblock3_denselayer11_norm2_running_var, 1e-05f); //features.denseblock3.denselayer11.norm2
	t4::release(x1057);
	t4::tensor4f x1059 = t4::ReluInplace(x1058); //features.denseblock3.denselayer11.relu2
	t4::release(x1058);
	t4::tensor4f x1060 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1059, ctx.features_denseblock3_denselayer11_conv2_weight); //features.denseblock3.denselayer11.conv2
	t4::release(x1059);
	t4::tensor4f x1061 = t4::Concat<1>(x1054, x1060); //features.denseblock3.denselayer11
	t4::release(x1054, x1060);
	t4::tensor4f x1062 = t4::BatchNormalization(x1061, ctx.features_denseblock3_denselayer12_norm1_weight, ctx.features_denseblock3_denselayer12_norm1_bias, ctx.features_denseblock3_denselayer12_norm1_running_mean, ctx.features_denseblock3_denselayer12_norm1_running_var, 1e-05f); //features.denseblock3.denselayer12.norm1
	t4::tensor4f x1063 = t4::ReluInplace(x1062); //features.denseblock3.denselayer12.relu1
	t4::release(x1062);
	t4::tensor4f x1064 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1063, ctx.features_denseblock3_denselayer12_conv1_weight); //features.denseblock3.denselayer12.conv1
	t4::release(x1063);
	t4::tensor4f x1065 = t4::BatchNormalizationInplace(x1064, ctx.features_denseblock3_denselayer12_norm2_weight, ctx.features_denseblock3_denselayer12_norm2_bias, ctx.features_denseblock3_denselayer12_norm2_running_mean, ctx.features_denseblock3_denselayer12_norm2_running_var, 1e-05f); //features.denseblock3.denselayer12.norm2
	t4::release(x1064);
	t4::tensor4f x1066 = t4::ReluInplace(x1065); //features.denseblock3.denselayer12.relu2
	t4::release(x1065);
	t4::tensor4f x1067 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1066, ctx.features_denseblock3_denselayer12_conv2_weight); //features.denseblock3.denselayer12.conv2
	t4::release(x1066);
	t4::tensor4f x1068 = t4::Concat<1>(x1061, x1067); //features.denseblock3.denselayer12
	t4::release(x1061, x1067);
	t4::tensor4f x1069 = t4::BatchNormalization(x1068, ctx.features_denseblock3_denselayer13_norm1_weight, ctx.features_denseblock3_denselayer13_norm1_bias, ctx.features_denseblock3_denselayer13_norm1_running_mean, ctx.features_denseblock3_denselayer13_norm1_running_var, 1e-05f); //features.denseblock3.denselayer13.norm1
	t4::tensor4f x1070 = t4::ReluInplace(x1069); //features.denseblock3.denselayer13.relu1
	t4::release(x1069);
	t4::tensor4f x1071 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1070, ctx.features_denseblock3_denselayer13_conv1_weight); //features.denseblock3.denselayer13.conv1
	t4::release(x1070);
	t4::tensor4f x1072 = t4::BatchNormalizationInplace(x1071, ctx.features_denseblock3_denselayer13_norm2_weight, ctx.features_denseblock3_denselayer13_norm2_bias, ctx.features_denseblock3_denselayer13_norm2_running_mean, ctx.features_denseblock3_denselayer13_norm2_running_var, 1e-05f); //features.denseblock3.denselayer13.norm2
	t4::release(x1071);
	t4::tensor4f x1073 = t4::ReluInplace(x1072); //features.denseblock3.denselayer13.relu2
	t4::release(x1072);
	t4::tensor4f x1074 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1073, ctx.features_denseblock3_denselayer13_conv2_weight); //features.denseblock3.denselayer13.conv2
	t4::release(x1073);
	t4::tensor4f x1075 = t4::Concat<1>(x1068, x1074); //features.denseblock3.denselayer13
	t4::release(x1068, x1074);
	t4::tensor4f x1076 = t4::BatchNormalization(x1075, ctx.features_denseblock3_denselayer14_norm1_weight, ctx.features_denseblock3_denselayer14_norm1_bias, ctx.features_denseblock3_denselayer14_norm1_running_mean, ctx.features_denseblock3_denselayer14_norm1_running_var, 1e-05f); //features.denseblock3.denselayer14.norm1
	t4::tensor4f x1077 = t4::ReluInplace(x1076); //features.denseblock3.denselayer14.relu1
	t4::release(x1076);
	t4::tensor4f x1078 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1077, ctx.features_denseblock3_denselayer14_conv1_weight); //features.denseblock3.denselayer14.conv1
	t4::release(x1077);
	t4::tensor4f x1079 = t4::BatchNormalizationInplace(x1078, ctx.features_denseblock3_denselayer14_norm2_weight, ctx.features_denseblock3_denselayer14_norm2_bias, ctx.features_denseblock3_denselayer14_norm2_running_mean, ctx.features_denseblock3_denselayer14_norm2_running_var, 1e-05f); //features.denseblock3.denselayer14.norm2
	t4::release(x1078);
	t4::tensor4f x1080 = t4::ReluInplace(x1079); //features.denseblock3.denselayer14.relu2
	t4::release(x1079);
	t4::tensor4f x1081 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1080, ctx.features_denseblock3_denselayer14_conv2_weight); //features.denseblock3.denselayer14.conv2
	t4::release(x1080);
	t4::tensor4f x1082 = t4::Concat<1>(x1075, x1081); //features.denseblock3.denselayer14
	t4::release(x1075, x1081);
	t4::tensor4f x1083 = t4::BatchNormalization(x1082, ctx.features_denseblock3_denselayer15_norm1_weight, ctx.features_denseblock3_denselayer15_norm1_bias, ctx.features_denseblock3_denselayer15_norm1_running_mean, ctx.features_denseblock3_denselayer15_norm1_running_var, 1e-05f); //features.denseblock3.denselayer15.norm1
	t4::tensor4f x1084 = t4::ReluInplace(x1083); //features.denseblock3.denselayer15.relu1
	t4::release(x1083);
	t4::tensor4f x1085 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1084, ctx.features_denseblock3_denselayer15_conv1_weight); //features.denseblock3.denselayer15.conv1
	t4::release(x1084);
	t4::tensor4f x1086 = t4::BatchNormalizationInplace(x1085, ctx.features_denseblock3_denselayer15_norm2_weight, ctx.features_denseblock3_denselayer15_norm2_bias, ctx.features_denseblock3_denselayer15_norm2_running_mean, ctx.features_denseblock3_denselayer15_norm2_running_var, 1e-05f); //features.denseblock3.denselayer15.norm2
	t4::release(x1085);
	t4::tensor4f x1087 = t4::ReluInplace(x1086); //features.denseblock3.denselayer15.relu2
	t4::release(x1086);
	t4::tensor4f x1088 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1087, ctx.features_denseblock3_denselayer15_conv2_weight); //features.denseblock3.denselayer15.conv2
	t4::release(x1087);
	t4::tensor4f x1089 = t4::Concat<1>(x1082, x1088); //features.denseblock3.denselayer15
	t4::release(x1082, x1088);
	t4::tensor4f x1090 = t4::BatchNormalization(x1089, ctx.features_denseblock3_denselayer16_norm1_weight, ctx.features_denseblock3_denselayer16_norm1_bias, ctx.features_denseblock3_denselayer16_norm1_running_mean, ctx.features_denseblock3_denselayer16_norm1_running_var, 1e-05f); //features.denseblock3.denselayer16.norm1
	t4::tensor4f x1091 = t4::ReluInplace(x1090); //features.denseblock3.denselayer16.relu1
	t4::release(x1090);
	t4::tensor4f x1092 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1091, ctx.features_denseblock3_denselayer16_conv1_weight); //features.denseblock3.denselayer16.conv1
	t4::release(x1091);
	t4::tensor4f x1093 = t4::BatchNormalizationInplace(x1092, ctx.features_denseblock3_denselayer16_norm2_weight, ctx.features_denseblock3_denselayer16_norm2_bias, ctx.features_denseblock3_denselayer16_norm2_running_mean, ctx.features_denseblock3_denselayer16_norm2_running_var, 1e-05f); //features.denseblock3.denselayer16.norm2
	t4::release(x1092);
	t4::tensor4f x1094 = t4::ReluInplace(x1093); //features.denseblock3.denselayer16.relu2
	t4::release(x1093);
	t4::tensor4f x1095 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1094, ctx.features_denseblock3_denselayer16_conv2_weight); //features.denseblock3.denselayer16.conv2
	t4::release(x1094);
	t4::tensor4f x1096 = t4::Concat<1>(x1089, x1095); //features.denseblock3.denselayer16
	t4::release(x1089, x1095);
	t4::tensor4f x1097 = t4::BatchNormalization(x1096, ctx.features_denseblock3_denselayer17_norm1_weight, ctx.features_denseblock3_denselayer17_norm1_bias, ctx.features_denseblock3_denselayer17_norm1_running_mean, ctx.features_denseblock3_denselayer17_norm1_running_var, 1e-05f); //features.denseblock3.denselayer17.norm1
	t4::tensor4f x1098 = t4::ReluInplace(x1097); //features.denseblock3.denselayer17.relu1
	t4::release(x1097);
	t4::tensor4f x1099 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1098, ctx.features_denseblock3_denselayer17_conv1_weight); //features.denseblock3.denselayer17.conv1
	t4::release(x1098);
	t4::tensor4f x1100 = t4::BatchNormalizationInplace(x1099, ctx.features_denseblock3_denselayer17_norm2_weight, ctx.features_denseblock3_denselayer17_norm2_bias, ctx.features_denseblock3_denselayer17_norm2_running_mean, ctx.features_denseblock3_denselayer17_norm2_running_var, 1e-05f); //features.denseblock3.denselayer17.norm2
	t4::release(x1099);
	t4::tensor4f x1101 = t4::ReluInplace(x1100); //features.denseblock3.denselayer17.relu2
	t4::release(x1100);
	t4::tensor4f x1102 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1101, ctx.features_denseblock3_denselayer17_conv2_weight); //features.denseblock3.denselayer17.conv2
	t4::release(x1101);
	t4::tensor4f x1103 = t4::Concat<1>(x1096, x1102); //features.denseblock3.denselayer17
	t4::release(x1096, x1102);
	t4::tensor4f x1104 = t4::BatchNormalization(x1103, ctx.features_denseblock3_denselayer18_norm1_weight, ctx.features_denseblock3_denselayer18_norm1_bias, ctx.features_denseblock3_denselayer18_norm1_running_mean, ctx.features_denseblock3_denselayer18_norm1_running_var, 1e-05f); //features.denseblock3.denselayer18.norm1
	t4::tensor4f x1105 = t4::ReluInplace(x1104); //features.denseblock3.denselayer18.relu1
	t4::release(x1104);
	t4::tensor4f x1106 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1105, ctx.features_denseblock3_denselayer18_conv1_weight); //features.denseblock3.denselayer18.conv1
	t4::release(x1105);
	t4::tensor4f x1107 = t4::BatchNormalizationInplace(x1106, ctx.features_denseblock3_denselayer18_norm2_weight, ctx.features_denseblock3_denselayer18_norm2_bias, ctx.features_denseblock3_denselayer18_norm2_running_mean, ctx.features_denseblock3_denselayer18_norm2_running_var, 1e-05f); //features.denseblock3.denselayer18.norm2
	t4::release(x1106);
	t4::tensor4f x1108 = t4::ReluInplace(x1107); //features.denseblock3.denselayer18.relu2
	t4::release(x1107);
	t4::tensor4f x1109 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1108, ctx.features_denseblock3_denselayer18_conv2_weight); //features.denseblock3.denselayer18.conv2
	t4::release(x1108);
	t4::tensor4f x1110 = t4::Concat<1>(x1103, x1109); //features.denseblock3.denselayer18
	t4::release(x1103, x1109);
	t4::tensor4f x1111 = t4::BatchNormalization(x1110, ctx.features_denseblock3_denselayer19_norm1_weight, ctx.features_denseblock3_denselayer19_norm1_bias, ctx.features_denseblock3_denselayer19_norm1_running_mean, ctx.features_denseblock3_denselayer19_norm1_running_var, 1e-05f); //features.denseblock3.denselayer19.norm1
	t4::tensor4f x1112 = t4::ReluInplace(x1111); //features.denseblock3.denselayer19.relu1
	t4::release(x1111);
	t4::tensor4f x1113 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1112, ctx.features_denseblock3_denselayer19_conv1_weight); //features.denseblock3.denselayer19.conv1
	t4::release(x1112);
	t4::tensor4f x1114 = t4::BatchNormalizationInplace(x1113, ctx.features_denseblock3_denselayer19_norm2_weight, ctx.features_denseblock3_denselayer19_norm2_bias, ctx.features_denseblock3_denselayer19_norm2_running_mean, ctx.features_denseblock3_denselayer19_norm2_running_var, 1e-05f); //features.denseblock3.denselayer19.norm2
	t4::release(x1113);
	t4::tensor4f x1115 = t4::ReluInplace(x1114); //features.denseblock3.denselayer19.relu2
	t4::release(x1114);
	t4::tensor4f x1116 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1115, ctx.features_denseblock3_denselayer19_conv2_weight); //features.denseblock3.denselayer19.conv2
	t4::release(x1115);
	t4::tensor4f x1117 = t4::Concat<1>(x1110, x1116); //features.denseblock3.denselayer19
	t4::release(x1110, x1116);
	t4::tensor4f x1118 = t4::BatchNormalization(x1117, ctx.features_denseblock3_denselayer20_norm1_weight, ctx.features_denseblock3_denselayer20_norm1_bias, ctx.features_denseblock3_denselayer20_norm1_running_mean, ctx.features_denseblock3_denselayer20_norm1_running_var, 1e-05f); //features.denseblock3.denselayer20.norm1
	t4::tensor4f x1119 = t4::ReluInplace(x1118); //features.denseblock3.denselayer20.relu1
	t4::release(x1118);
	t4::tensor4f x1120 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1119, ctx.features_denseblock3_denselayer20_conv1_weight); //features.denseblock3.denselayer20.conv1
	t4::release(x1119);
	t4::tensor4f x1121 = t4::BatchNormalizationInplace(x1120, ctx.features_denseblock3_denselayer20_norm2_weight, ctx.features_denseblock3_denselayer20_norm2_bias, ctx.features_denseblock3_denselayer20_norm2_running_mean, ctx.features_denseblock3_denselayer20_norm2_running_var, 1e-05f); //features.denseblock3.denselayer20.norm2
	t4::release(x1120);
	t4::tensor4f x1122 = t4::ReluInplace(x1121); //features.denseblock3.denselayer20.relu2
	t4::release(x1121);
	t4::tensor4f x1123 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1122, ctx.features_denseblock3_denselayer20_conv2_weight); //features.denseblock3.denselayer20.conv2
	t4::release(x1122);
	t4::tensor4f x1124 = t4::Concat<1>(x1117, x1123); //features.denseblock3.denselayer20
	t4::release(x1117, x1123);
	t4::tensor4f x1125 = t4::BatchNormalization(x1124, ctx.features_denseblock3_denselayer21_norm1_weight, ctx.features_denseblock3_denselayer21_norm1_bias, ctx.features_denseblock3_denselayer21_norm1_running_mean, ctx.features_denseblock3_denselayer21_norm1_running_var, 1e-05f); //features.denseblock3.denselayer21.norm1
	t4::tensor4f x1126 = t4::ReluInplace(x1125); //features.denseblock3.denselayer21.relu1
	t4::release(x1125);
	t4::tensor4f x1127 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1126, ctx.features_denseblock3_denselayer21_conv1_weight); //features.denseblock3.denselayer21.conv1
	t4::release(x1126);
	t4::tensor4f x1128 = t4::BatchNormalizationInplace(x1127, ctx.features_denseblock3_denselayer21_norm2_weight, ctx.features_denseblock3_denselayer21_norm2_bias, ctx.features_denseblock3_denselayer21_norm2_running_mean, ctx.features_denseblock3_denselayer21_norm2_running_var, 1e-05f); //features.denseblock3.denselayer21.norm2
	t4::release(x1127);
	t4::tensor4f x1129 = t4::ReluInplace(x1128); //features.denseblock3.denselayer21.relu2
	t4::release(x1128);
	t4::tensor4f x1130 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1129, ctx.features_denseblock3_denselayer21_conv2_weight); //features.denseblock3.denselayer21.conv2
	t4::release(x1129);
	t4::tensor4f x1131 = t4::Concat<1>(x1124, x1130); //features.denseblock3.denselayer21
	t4::release(x1124, x1130);
	t4::tensor4f x1132 = t4::BatchNormalization(x1131, ctx.features_denseblock3_denselayer22_norm1_weight, ctx.features_denseblock3_denselayer22_norm1_bias, ctx.features_denseblock3_denselayer22_norm1_running_mean, ctx.features_denseblock3_denselayer22_norm1_running_var, 1e-05f); //features.denseblock3.denselayer22.norm1
	t4::tensor4f x1133 = t4::ReluInplace(x1132); //features.denseblock3.denselayer22.relu1
	t4::release(x1132);
	t4::tensor4f x1134 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1133, ctx.features_denseblock3_denselayer22_conv1_weight); //features.denseblock3.denselayer22.conv1
	t4::release(x1133);
	t4::tensor4f x1135 = t4::BatchNormalizationInplace(x1134, ctx.features_denseblock3_denselayer22_norm2_weight, ctx.features_denseblock3_denselayer22_norm2_bias, ctx.features_denseblock3_denselayer22_norm2_running_mean, ctx.features_denseblock3_denselayer22_norm2_running_var, 1e-05f); //features.denseblock3.denselayer22.norm2
	t4::release(x1134);
	t4::tensor4f x1136 = t4::ReluInplace(x1135); //features.denseblock3.denselayer22.relu2
	t4::release(x1135);
	t4::tensor4f x1137 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1136, ctx.features_denseblock3_denselayer22_conv2_weight); //features.denseblock3.denselayer22.conv2
	t4::release(x1136);
	t4::tensor4f x1138 = t4::Concat<1>(x1131, x1137); //features.denseblock3.denselayer22
	t4::release(x1131, x1137);
	t4::tensor4f x1139 = t4::BatchNormalization(x1138, ctx.features_denseblock3_denselayer23_norm1_weight, ctx.features_denseblock3_denselayer23_norm1_bias, ctx.features_denseblock3_denselayer23_norm1_running_mean, ctx.features_denseblock3_denselayer23_norm1_running_var, 1e-05f); //features.denseblock3.denselayer23.norm1
	t4::tensor4f x1140 = t4::ReluInplace(x1139); //features.denseblock3.denselayer23.relu1
	t4::release(x1139);
	t4::tensor4f x1141 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1140, ctx.features_denseblock3_denselayer23_conv1_weight); //features.denseblock3.denselayer23.conv1
	t4::release(x1140);
	t4::tensor4f x1142 = t4::BatchNormalizationInplace(x1141, ctx.features_denseblock3_denselayer23_norm2_weight, ctx.features_denseblock3_denselayer23_norm2_bias, ctx.features_denseblock3_denselayer23_norm2_running_mean, ctx.features_denseblock3_denselayer23_norm2_running_var, 1e-05f); //features.denseblock3.denselayer23.norm2
	t4::release(x1141);
	t4::tensor4f x1143 = t4::ReluInplace(x1142); //features.denseblock3.denselayer23.relu2
	t4::release(x1142);
	t4::tensor4f x1144 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1143, ctx.features_denseblock3_denselayer23_conv2_weight); //features.denseblock3.denselayer23.conv2
	t4::release(x1143);
	t4::tensor4f x1145 = t4::Concat<1>(x1138, x1144); //features.denseblock3.denselayer23
	t4::release(x1138, x1144);
	t4::tensor4f x1146 = t4::BatchNormalization(x1145, ctx.features_denseblock3_denselayer24_norm1_weight, ctx.features_denseblock3_denselayer24_norm1_bias, ctx.features_denseblock3_denselayer24_norm1_running_mean, ctx.features_denseblock3_denselayer24_norm1_running_var, 1e-05f); //features.denseblock3.denselayer24.norm1
	t4::tensor4f x1147 = t4::ReluInplace(x1146); //features.denseblock3.denselayer24.relu1
	t4::release(x1146);
	t4::tensor4f x1148 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1147, ctx.features_denseblock3_denselayer24_conv1_weight); //features.denseblock3.denselayer24.conv1
	t4::release(x1147);
	t4::tensor4f x1149 = t4::BatchNormalizationInplace(x1148, ctx.features_denseblock3_denselayer24_norm2_weight, ctx.features_denseblock3_denselayer24_norm2_bias, ctx.features_denseblock3_denselayer24_norm2_running_mean, ctx.features_denseblock3_denselayer24_norm2_running_var, 1e-05f); //features.denseblock3.denselayer24.norm2
	t4::release(x1148);
	t4::tensor4f x1150 = t4::ReluInplace(x1149); //features.denseblock3.denselayer24.relu2
	t4::release(x1149);
	t4::tensor4f x1151 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1150, ctx.features_denseblock3_denselayer24_conv2_weight); //features.denseblock3.denselayer24.conv2
	t4::release(x1150);
	t4::tensor4f x1152 = t4::Concat<1>(x1145, x1151); //features.denseblock3.denselayer24
	t4::release(x1145, x1151);
	t4::tensor4f x1153 = t4::BatchNormalization(x1152, ctx.features_denseblock3_denselayer25_norm1_weight, ctx.features_denseblock3_denselayer25_norm1_bias, ctx.features_denseblock3_denselayer25_norm1_running_mean, ctx.features_denseblock3_denselayer25_norm1_running_var, 1e-05f); //features.denseblock3.denselayer25.norm1
	t4::tensor4f x1154 = t4::ReluInplace(x1153); //features.denseblock3.denselayer25.relu1
	t4::release(x1153);
	t4::tensor4f x1155 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1154, ctx.features_denseblock3_denselayer25_conv1_weight); //features.denseblock3.denselayer25.conv1
	t4::release(x1154);
	t4::tensor4f x1156 = t4::BatchNormalizationInplace(x1155, ctx.features_denseblock3_denselayer25_norm2_weight, ctx.features_denseblock3_denselayer25_norm2_bias, ctx.features_denseblock3_denselayer25_norm2_running_mean, ctx.features_denseblock3_denselayer25_norm2_running_var, 1e-05f); //features.denseblock3.denselayer25.norm2
	t4::release(x1155);
	t4::tensor4f x1157 = t4::ReluInplace(x1156); //features.denseblock3.denselayer25.relu2
	t4::release(x1156);
	t4::tensor4f x1158 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1157, ctx.features_denseblock3_denselayer25_conv2_weight); //features.denseblock3.denselayer25.conv2
	t4::release(x1157);
	t4::tensor4f x1159 = t4::Concat<1>(x1152, x1158); //features.denseblock3.denselayer25
	t4::release(x1152, x1158);
	t4::tensor4f x1160 = t4::BatchNormalization(x1159, ctx.features_denseblock3_denselayer26_norm1_weight, ctx.features_denseblock3_denselayer26_norm1_bias, ctx.features_denseblock3_denselayer26_norm1_running_mean, ctx.features_denseblock3_denselayer26_norm1_running_var, 1e-05f); //features.denseblock3.denselayer26.norm1
	t4::tensor4f x1161 = t4::ReluInplace(x1160); //features.denseblock3.denselayer26.relu1
	t4::release(x1160);
	t4::tensor4f x1162 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1161, ctx.features_denseblock3_denselayer26_conv1_weight); //features.denseblock3.denselayer26.conv1
	t4::release(x1161);
	t4::tensor4f x1163 = t4::BatchNormalizationInplace(x1162, ctx.features_denseblock3_denselayer26_norm2_weight, ctx.features_denseblock3_denselayer26_norm2_bias, ctx.features_denseblock3_denselayer26_norm2_running_mean, ctx.features_denseblock3_denselayer26_norm2_running_var, 1e-05f); //features.denseblock3.denselayer26.norm2
	t4::release(x1162);
	t4::tensor4f x1164 = t4::ReluInplace(x1163); //features.denseblock3.denselayer26.relu2
	t4::release(x1163);
	t4::tensor4f x1165 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1164, ctx.features_denseblock3_denselayer26_conv2_weight); //features.denseblock3.denselayer26.conv2
	t4::release(x1164);
	t4::tensor4f x1166 = t4::Concat<1>(x1159, x1165); //features.denseblock3.denselayer26
	t4::release(x1159, x1165);
	t4::tensor4f x1167 = t4::BatchNormalization(x1166, ctx.features_denseblock3_denselayer27_norm1_weight, ctx.features_denseblock3_denselayer27_norm1_bias, ctx.features_denseblock3_denselayer27_norm1_running_mean, ctx.features_denseblock3_denselayer27_norm1_running_var, 1e-05f); //features.denseblock3.denselayer27.norm1
	t4::tensor4f x1168 = t4::ReluInplace(x1167); //features.denseblock3.denselayer27.relu1
	t4::release(x1167);
	t4::tensor4f x1169 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1168, ctx.features_denseblock3_denselayer27_conv1_weight); //features.denseblock3.denselayer27.conv1
	t4::release(x1168);
	t4::tensor4f x1170 = t4::BatchNormalizationInplace(x1169, ctx.features_denseblock3_denselayer27_norm2_weight, ctx.features_denseblock3_denselayer27_norm2_bias, ctx.features_denseblock3_denselayer27_norm2_running_mean, ctx.features_denseblock3_denselayer27_norm2_running_var, 1e-05f); //features.denseblock3.denselayer27.norm2
	t4::release(x1169);
	t4::tensor4f x1171 = t4::ReluInplace(x1170); //features.denseblock3.denselayer27.relu2
	t4::release(x1170);
	t4::tensor4f x1172 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1171, ctx.features_denseblock3_denselayer27_conv2_weight); //features.denseblock3.denselayer27.conv2
	t4::release(x1171);
	t4::tensor4f x1173 = t4::Concat<1>(x1166, x1172); //features.denseblock3.denselayer27
	t4::release(x1166, x1172);
	t4::tensor4f x1174 = t4::BatchNormalization(x1173, ctx.features_denseblock3_denselayer28_norm1_weight, ctx.features_denseblock3_denselayer28_norm1_bias, ctx.features_denseblock3_denselayer28_norm1_running_mean, ctx.features_denseblock3_denselayer28_norm1_running_var, 1e-05f); //features.denseblock3.denselayer28.norm1
	t4::tensor4f x1175 = t4::ReluInplace(x1174); //features.denseblock3.denselayer28.relu1
	t4::release(x1174);
	t4::tensor4f x1176 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1175, ctx.features_denseblock3_denselayer28_conv1_weight); //features.denseblock3.denselayer28.conv1
	t4::release(x1175);
	t4::tensor4f x1177 = t4::BatchNormalizationInplace(x1176, ctx.features_denseblock3_denselayer28_norm2_weight, ctx.features_denseblock3_denselayer28_norm2_bias, ctx.features_denseblock3_denselayer28_norm2_running_mean, ctx.features_denseblock3_denselayer28_norm2_running_var, 1e-05f); //features.denseblock3.denselayer28.norm2
	t4::release(x1176);
	t4::tensor4f x1178 = t4::ReluInplace(x1177); //features.denseblock3.denselayer28.relu2
	t4::release(x1177);
	t4::tensor4f x1179 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1178, ctx.features_denseblock3_denselayer28_conv2_weight); //features.denseblock3.denselayer28.conv2
	t4::release(x1178);
	t4::tensor4f x1180 = t4::Concat<1>(x1173, x1179); //features.denseblock3.denselayer28
	t4::release(x1173, x1179);
	t4::tensor4f x1181 = t4::BatchNormalization(x1180, ctx.features_denseblock3_denselayer29_norm1_weight, ctx.features_denseblock3_denselayer29_norm1_bias, ctx.features_denseblock3_denselayer29_norm1_running_mean, ctx.features_denseblock3_denselayer29_norm1_running_var, 1e-05f); //features.denseblock3.denselayer29.norm1
	t4::tensor4f x1182 = t4::ReluInplace(x1181); //features.denseblock3.denselayer29.relu1
	t4::release(x1181);
	t4::tensor4f x1183 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1182, ctx.features_denseblock3_denselayer29_conv1_weight); //features.denseblock3.denselayer29.conv1
	t4::release(x1182);
	t4::tensor4f x1184 = t4::BatchNormalizationInplace(x1183, ctx.features_denseblock3_denselayer29_norm2_weight, ctx.features_denseblock3_denselayer29_norm2_bias, ctx.features_denseblock3_denselayer29_norm2_running_mean, ctx.features_denseblock3_denselayer29_norm2_running_var, 1e-05f); //features.denseblock3.denselayer29.norm2
	t4::release(x1183);
	t4::tensor4f x1185 = t4::ReluInplace(x1184); //features.denseblock3.denselayer29.relu2
	t4::release(x1184);
	t4::tensor4f x1186 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1185, ctx.features_denseblock3_denselayer29_conv2_weight); //features.denseblock3.denselayer29.conv2
	t4::release(x1185);
	t4::tensor4f x1187 = t4::Concat<1>(x1180, x1186); //features.denseblock3.denselayer29
	t4::release(x1180, x1186);
	t4::tensor4f x1188 = t4::BatchNormalization(x1187, ctx.features_denseblock3_denselayer30_norm1_weight, ctx.features_denseblock3_denselayer30_norm1_bias, ctx.features_denseblock3_denselayer30_norm1_running_mean, ctx.features_denseblock3_denselayer30_norm1_running_var, 1e-05f); //features.denseblock3.denselayer30.norm1
	t4::tensor4f x1189 = t4::ReluInplace(x1188); //features.denseblock3.denselayer30.relu1
	t4::release(x1188);
	t4::tensor4f x1190 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1189, ctx.features_denseblock3_denselayer30_conv1_weight); //features.denseblock3.denselayer30.conv1
	t4::release(x1189);
	t4::tensor4f x1191 = t4::BatchNormalizationInplace(x1190, ctx.features_denseblock3_denselayer30_norm2_weight, ctx.features_denseblock3_denselayer30_norm2_bias, ctx.features_denseblock3_denselayer30_norm2_running_mean, ctx.features_denseblock3_denselayer30_norm2_running_var, 1e-05f); //features.denseblock3.denselayer30.norm2
	t4::release(x1190);
	t4::tensor4f x1192 = t4::ReluInplace(x1191); //features.denseblock3.denselayer30.relu2
	t4::release(x1191);
	t4::tensor4f x1193 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1192, ctx.features_denseblock3_denselayer30_conv2_weight); //features.denseblock3.denselayer30.conv2
	t4::release(x1192);
	t4::tensor4f x1194 = t4::Concat<1>(x1187, x1193); //features.denseblock3.denselayer30
	t4::release(x1187, x1193);
	t4::tensor4f x1195 = t4::BatchNormalization(x1194, ctx.features_denseblock3_denselayer31_norm1_weight, ctx.features_denseblock3_denselayer31_norm1_bias, ctx.features_denseblock3_denselayer31_norm1_running_mean, ctx.features_denseblock3_denselayer31_norm1_running_var, 1e-05f); //features.denseblock3.denselayer31.norm1
	t4::tensor4f x1196 = t4::ReluInplace(x1195); //features.denseblock3.denselayer31.relu1
	t4::release(x1195);
	t4::tensor4f x1197 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1196, ctx.features_denseblock3_denselayer31_conv1_weight); //features.denseblock3.denselayer31.conv1
	t4::release(x1196);
	t4::tensor4f x1198 = t4::BatchNormalizationInplace(x1197, ctx.features_denseblock3_denselayer31_norm2_weight, ctx.features_denseblock3_denselayer31_norm2_bias, ctx.features_denseblock3_denselayer31_norm2_running_mean, ctx.features_denseblock3_denselayer31_norm2_running_var, 1e-05f); //features.denseblock3.denselayer31.norm2
	t4::release(x1197);
	t4::tensor4f x1199 = t4::ReluInplace(x1198); //features.denseblock3.denselayer31.relu2
	t4::release(x1198);
	t4::tensor4f x1200 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1199, ctx.features_denseblock3_denselayer31_conv2_weight); //features.denseblock3.denselayer31.conv2
	t4::release(x1199);
	t4::tensor4f x1201 = t4::Concat<1>(x1194, x1200); //features.denseblock3.denselayer31
	t4::release(x1194, x1200);
	t4::tensor4f x1202 = t4::BatchNormalization(x1201, ctx.features_denseblock3_denselayer32_norm1_weight, ctx.features_denseblock3_denselayer32_norm1_bias, ctx.features_denseblock3_denselayer32_norm1_running_mean, ctx.features_denseblock3_denselayer32_norm1_running_var, 1e-05f); //features.denseblock3.denselayer32.norm1
	t4::tensor4f x1203 = t4::ReluInplace(x1202); //features.denseblock3.denselayer32.relu1
	t4::release(x1202);
	t4::tensor4f x1204 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1203, ctx.features_denseblock3_denselayer32_conv1_weight); //features.denseblock3.denselayer32.conv1
	t4::release(x1203);
	t4::tensor4f x1205 = t4::BatchNormalizationInplace(x1204, ctx.features_denseblock3_denselayer32_norm2_weight, ctx.features_denseblock3_denselayer32_norm2_bias, ctx.features_denseblock3_denselayer32_norm2_running_mean, ctx.features_denseblock3_denselayer32_norm2_running_var, 1e-05f); //features.denseblock3.denselayer32.norm2
	t4::release(x1204);
	t4::tensor4f x1206 = t4::ReluInplace(x1205); //features.denseblock3.denselayer32.relu2
	t4::release(x1205);
	t4::tensor4f x1207 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1206, ctx.features_denseblock3_denselayer32_conv2_weight); //features.denseblock3.denselayer32.conv2
	t4::release(x1206);
	t4::tensor4f x1208 = t4::Concat<1>(x1201, x1207); //features.denseblock3.denselayer32
	t4::release(x1201, x1207);
	t4::tensor4f x1209 = t4::BatchNormalizationInplace(x1208, ctx.features_transition3_norm_weight, ctx.features_transition3_norm_bias, ctx.features_transition3_norm_running_mean, ctx.features_transition3_norm_running_var, 1e-05f); //features.transition3.norm
	t4::release(x1208);
	t4::tensor4f x1210 = t4::ReluInplace(x1209); //features.transition3.relu
	t4::release(x1209);
	t4::tensor4f x1211 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1210, ctx.features_transition3_conv_weight); //features.transition3.conv
	t4::release(x1210);
	t4::tensor4f x1212 = t4::AveragePool2d<2, 2, 2, 2, 0, 0>(x1211); //features.transition3.pool
	t4::release(x1211);
	t4::tensor4f x1213 = t4::BatchNormalization(x1212, ctx.features_denseblock4_denselayer1_norm1_weight, ctx.features_denseblock4_denselayer1_norm1_bias, ctx.features_denseblock4_denselayer1_norm1_running_mean, ctx.features_denseblock4_denselayer1_norm1_running_var, 1e-05f); //features.denseblock4.denselayer1.norm1
	t4::tensor4f x1214 = t4::ReluInplace(x1213); //features.denseblock4.denselayer1.relu1
	t4::release(x1213);
	t4::tensor4f x1215 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1214, ctx.features_denseblock4_denselayer1_conv1_weight); //features.denseblock4.denselayer1.conv1
	t4::release(x1214);
	t4::tensor4f x1216 = t4::BatchNormalizationInplace(x1215, ctx.features_denseblock4_denselayer1_norm2_weight, ctx.features_denseblock4_denselayer1_norm2_bias, ctx.features_denseblock4_denselayer1_norm2_running_mean, ctx.features_denseblock4_denselayer1_norm2_running_var, 1e-05f); //features.denseblock4.denselayer1.norm2
	t4::release(x1215);
	t4::tensor4f x1217 = t4::ReluInplace(x1216); //features.denseblock4.denselayer1.relu2
	t4::release(x1216);
	t4::tensor4f x1218 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1217, ctx.features_denseblock4_denselayer1_conv2_weight); //features.denseblock4.denselayer1.conv2
	t4::release(x1217);
	t4::tensor4f x1219 = t4::Concat<1>(x1212, x1218); //features.denseblock4.denselayer1
	t4::release(x1212, x1218);
	t4::tensor4f x1220 = t4::BatchNormalization(x1219, ctx.features_denseblock4_denselayer2_norm1_weight, ctx.features_denseblock4_denselayer2_norm1_bias, ctx.features_denseblock4_denselayer2_norm1_running_mean, ctx.features_denseblock4_denselayer2_norm1_running_var, 1e-05f); //features.denseblock4.denselayer2.norm1
	t4::tensor4f x1221 = t4::ReluInplace(x1220); //features.denseblock4.denselayer2.relu1
	t4::release(x1220);
	t4::tensor4f x1222 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1221, ctx.features_denseblock4_denselayer2_conv1_weight); //features.denseblock4.denselayer2.conv1
	t4::release(x1221);
	t4::tensor4f x1223 = t4::BatchNormalizationInplace(x1222, ctx.features_denseblock4_denselayer2_norm2_weight, ctx.features_denseblock4_denselayer2_norm2_bias, ctx.features_denseblock4_denselayer2_norm2_running_mean, ctx.features_denseblock4_denselayer2_norm2_running_var, 1e-05f); //features.denseblock4.denselayer2.norm2
	t4::release(x1222);
	t4::tensor4f x1224 = t4::ReluInplace(x1223); //features.denseblock4.denselayer2.relu2
	t4::release(x1223);
	t4::tensor4f x1225 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1224, ctx.features_denseblock4_denselayer2_conv2_weight); //features.denseblock4.denselayer2.conv2
	t4::release(x1224);
	t4::tensor4f x1226 = t4::Concat<1>(x1219, x1225); //features.denseblock4.denselayer2
	t4::release(x1219, x1225);
	t4::tensor4f x1227 = t4::BatchNormalization(x1226, ctx.features_denseblock4_denselayer3_norm1_weight, ctx.features_denseblock4_denselayer3_norm1_bias, ctx.features_denseblock4_denselayer3_norm1_running_mean, ctx.features_denseblock4_denselayer3_norm1_running_var, 1e-05f); //features.denseblock4.denselayer3.norm1
	t4::tensor4f x1228 = t4::ReluInplace(x1227); //features.denseblock4.denselayer3.relu1
	t4::release(x1227);
	t4::tensor4f x1229 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1228, ctx.features_denseblock4_denselayer3_conv1_weight); //features.denseblock4.denselayer3.conv1
	t4::release(x1228);
	t4::tensor4f x1230 = t4::BatchNormalizationInplace(x1229, ctx.features_denseblock4_denselayer3_norm2_weight, ctx.features_denseblock4_denselayer3_norm2_bias, ctx.features_denseblock4_denselayer3_norm2_running_mean, ctx.features_denseblock4_denselayer3_norm2_running_var, 1e-05f); //features.denseblock4.denselayer3.norm2
	t4::release(x1229);
	t4::tensor4f x1231 = t4::ReluInplace(x1230); //features.denseblock4.denselayer3.relu2
	t4::release(x1230);
	t4::tensor4f x1232 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1231, ctx.features_denseblock4_denselayer3_conv2_weight); //features.denseblock4.denselayer3.conv2
	t4::release(x1231);
	t4::tensor4f x1233 = t4::Concat<1>(x1226, x1232); //features.denseblock4.denselayer3
	t4::release(x1226, x1232);
	t4::tensor4f x1234 = t4::BatchNormalization(x1233, ctx.features_denseblock4_denselayer4_norm1_weight, ctx.features_denseblock4_denselayer4_norm1_bias, ctx.features_denseblock4_denselayer4_norm1_running_mean, ctx.features_denseblock4_denselayer4_norm1_running_var, 1e-05f); //features.denseblock4.denselayer4.norm1
	t4::tensor4f x1235 = t4::ReluInplace(x1234); //features.denseblock4.denselayer4.relu1
	t4::release(x1234);
	t4::tensor4f x1236 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1235, ctx.features_denseblock4_denselayer4_conv1_weight); //features.denseblock4.denselayer4.conv1
	t4::release(x1235);
	t4::tensor4f x1237 = t4::BatchNormalizationInplace(x1236, ctx.features_denseblock4_denselayer4_norm2_weight, ctx.features_denseblock4_denselayer4_norm2_bias, ctx.features_denseblock4_denselayer4_norm2_running_mean, ctx.features_denseblock4_denselayer4_norm2_running_var, 1e-05f); //features.denseblock4.denselayer4.norm2
	t4::release(x1236);
	t4::tensor4f x1238 = t4::ReluInplace(x1237); //features.denseblock4.denselayer4.relu2
	t4::release(x1237);
	t4::tensor4f x1239 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1238, ctx.features_denseblock4_denselayer4_conv2_weight); //features.denseblock4.denselayer4.conv2
	t4::release(x1238);
	t4::tensor4f x1240 = t4::Concat<1>(x1233, x1239); //features.denseblock4.denselayer4
	t4::release(x1233, x1239);
	t4::tensor4f x1241 = t4::BatchNormalization(x1240, ctx.features_denseblock4_denselayer5_norm1_weight, ctx.features_denseblock4_denselayer5_norm1_bias, ctx.features_denseblock4_denselayer5_norm1_running_mean, ctx.features_denseblock4_denselayer5_norm1_running_var, 1e-05f); //features.denseblock4.denselayer5.norm1
	t4::tensor4f x1242 = t4::ReluInplace(x1241); //features.denseblock4.denselayer5.relu1
	t4::release(x1241);
	t4::tensor4f x1243 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1242, ctx.features_denseblock4_denselayer5_conv1_weight); //features.denseblock4.denselayer5.conv1
	t4::release(x1242);
	t4::tensor4f x1244 = t4::BatchNormalizationInplace(x1243, ctx.features_denseblock4_denselayer5_norm2_weight, ctx.features_denseblock4_denselayer5_norm2_bias, ctx.features_denseblock4_denselayer5_norm2_running_mean, ctx.features_denseblock4_denselayer5_norm2_running_var, 1e-05f); //features.denseblock4.denselayer5.norm2
	t4::release(x1243);
	t4::tensor4f x1245 = t4::ReluInplace(x1244); //features.denseblock4.denselayer5.relu2
	t4::release(x1244);
	t4::tensor4f x1246 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1245, ctx.features_denseblock4_denselayer5_conv2_weight); //features.denseblock4.denselayer5.conv2
	t4::release(x1245);
	t4::tensor4f x1247 = t4::Concat<1>(x1240, x1246); //features.denseblock4.denselayer5
	t4::release(x1240, x1246);
	t4::tensor4f x1248 = t4::BatchNormalization(x1247, ctx.features_denseblock4_denselayer6_norm1_weight, ctx.features_denseblock4_denselayer6_norm1_bias, ctx.features_denseblock4_denselayer6_norm1_running_mean, ctx.features_denseblock4_denselayer6_norm1_running_var, 1e-05f); //features.denseblock4.denselayer6.norm1
	t4::tensor4f x1249 = t4::ReluInplace(x1248); //features.denseblock4.denselayer6.relu1
	t4::release(x1248);
	t4::tensor4f x1250 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1249, ctx.features_denseblock4_denselayer6_conv1_weight); //features.denseblock4.denselayer6.conv1
	t4::release(x1249);
	t4::tensor4f x1251 = t4::BatchNormalizationInplace(x1250, ctx.features_denseblock4_denselayer6_norm2_weight, ctx.features_denseblock4_denselayer6_norm2_bias, ctx.features_denseblock4_denselayer6_norm2_running_mean, ctx.features_denseblock4_denselayer6_norm2_running_var, 1e-05f); //features.denseblock4.denselayer6.norm2
	t4::release(x1250);
	t4::tensor4f x1252 = t4::ReluInplace(x1251); //features.denseblock4.denselayer6.relu2
	t4::release(x1251);
	t4::tensor4f x1253 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1252, ctx.features_denseblock4_denselayer6_conv2_weight); //features.denseblock4.denselayer6.conv2
	t4::release(x1252);
	t4::tensor4f x1254 = t4::Concat<1>(x1247, x1253); //features.denseblock4.denselayer6
	t4::release(x1247, x1253);
	t4::tensor4f x1255 = t4::BatchNormalization(x1254, ctx.features_denseblock4_denselayer7_norm1_weight, ctx.features_denseblock4_denselayer7_norm1_bias, ctx.features_denseblock4_denselayer7_norm1_running_mean, ctx.features_denseblock4_denselayer7_norm1_running_var, 1e-05f); //features.denseblock4.denselayer7.norm1
	t4::tensor4f x1256 = t4::ReluInplace(x1255); //features.denseblock4.denselayer7.relu1
	t4::release(x1255);
	t4::tensor4f x1257 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1256, ctx.features_denseblock4_denselayer7_conv1_weight); //features.denseblock4.denselayer7.conv1
	t4::release(x1256);
	t4::tensor4f x1258 = t4::BatchNormalizationInplace(x1257, ctx.features_denseblock4_denselayer7_norm2_weight, ctx.features_denseblock4_denselayer7_norm2_bias, ctx.features_denseblock4_denselayer7_norm2_running_mean, ctx.features_denseblock4_denselayer7_norm2_running_var, 1e-05f); //features.denseblock4.denselayer7.norm2
	t4::release(x1257);
	t4::tensor4f x1259 = t4::ReluInplace(x1258); //features.denseblock4.denselayer7.relu2
	t4::release(x1258);
	t4::tensor4f x1260 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1259, ctx.features_denseblock4_denselayer7_conv2_weight); //features.denseblock4.denselayer7.conv2
	t4::release(x1259);
	t4::tensor4f x1261 = t4::Concat<1>(x1254, x1260); //features.denseblock4.denselayer7
	t4::release(x1254, x1260);
	t4::tensor4f x1262 = t4::BatchNormalization(x1261, ctx.features_denseblock4_denselayer8_norm1_weight, ctx.features_denseblock4_denselayer8_norm1_bias, ctx.features_denseblock4_denselayer8_norm1_running_mean, ctx.features_denseblock4_denselayer8_norm1_running_var, 1e-05f); //features.denseblock4.denselayer8.norm1
	t4::tensor4f x1263 = t4::ReluInplace(x1262); //features.denseblock4.denselayer8.relu1
	t4::release(x1262);
	t4::tensor4f x1264 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1263, ctx.features_denseblock4_denselayer8_conv1_weight); //features.denseblock4.denselayer8.conv1
	t4::release(x1263);
	t4::tensor4f x1265 = t4::BatchNormalizationInplace(x1264, ctx.features_denseblock4_denselayer8_norm2_weight, ctx.features_denseblock4_denselayer8_norm2_bias, ctx.features_denseblock4_denselayer8_norm2_running_mean, ctx.features_denseblock4_denselayer8_norm2_running_var, 1e-05f); //features.denseblock4.denselayer8.norm2
	t4::release(x1264);
	t4::tensor4f x1266 = t4::ReluInplace(x1265); //features.denseblock4.denselayer8.relu2
	t4::release(x1265);
	t4::tensor4f x1267 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1266, ctx.features_denseblock4_denselayer8_conv2_weight); //features.denseblock4.denselayer8.conv2
	t4::release(x1266);
	t4::tensor4f x1268 = t4::Concat<1>(x1261, x1267); //features.denseblock4.denselayer8
	t4::release(x1261, x1267);
	t4::tensor4f x1269 = t4::BatchNormalization(x1268, ctx.features_denseblock4_denselayer9_norm1_weight, ctx.features_denseblock4_denselayer9_norm1_bias, ctx.features_denseblock4_denselayer9_norm1_running_mean, ctx.features_denseblock4_denselayer9_norm1_running_var, 1e-05f); //features.denseblock4.denselayer9.norm1
	t4::tensor4f x1270 = t4::ReluInplace(x1269); //features.denseblock4.denselayer9.relu1
	t4::release(x1269);
	t4::tensor4f x1271 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1270, ctx.features_denseblock4_denselayer9_conv1_weight); //features.denseblock4.denselayer9.conv1
	t4::release(x1270);
	t4::tensor4f x1272 = t4::BatchNormalizationInplace(x1271, ctx.features_denseblock4_denselayer9_norm2_weight, ctx.features_denseblock4_denselayer9_norm2_bias, ctx.features_denseblock4_denselayer9_norm2_running_mean, ctx.features_denseblock4_denselayer9_norm2_running_var, 1e-05f); //features.denseblock4.denselayer9.norm2
	t4::release(x1271);
	t4::tensor4f x1273 = t4::ReluInplace(x1272); //features.denseblock4.denselayer9.relu2
	t4::release(x1272);
	t4::tensor4f x1274 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1273, ctx.features_denseblock4_denselayer9_conv2_weight); //features.denseblock4.denselayer9.conv2
	t4::release(x1273);
	t4::tensor4f x1275 = t4::Concat<1>(x1268, x1274); //features.denseblock4.denselayer9
	t4::release(x1268, x1274);
	t4::tensor4f x1276 = t4::BatchNormalization(x1275, ctx.features_denseblock4_denselayer10_norm1_weight, ctx.features_denseblock4_denselayer10_norm1_bias, ctx.features_denseblock4_denselayer10_norm1_running_mean, ctx.features_denseblock4_denselayer10_norm1_running_var, 1e-05f); //features.denseblock4.denselayer10.norm1
	t4::tensor4f x1277 = t4::ReluInplace(x1276); //features.denseblock4.denselayer10.relu1
	t4::release(x1276);
	t4::tensor4f x1278 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1277, ctx.features_denseblock4_denselayer10_conv1_weight); //features.denseblock4.denselayer10.conv1
	t4::release(x1277);
	t4::tensor4f x1279 = t4::BatchNormalizationInplace(x1278, ctx.features_denseblock4_denselayer10_norm2_weight, ctx.features_denseblock4_denselayer10_norm2_bias, ctx.features_denseblock4_denselayer10_norm2_running_mean, ctx.features_denseblock4_denselayer10_norm2_running_var, 1e-05f); //features.denseblock4.denselayer10.norm2
	t4::release(x1278);
	t4::tensor4f x1280 = t4::ReluInplace(x1279); //features.denseblock4.denselayer10.relu2
	t4::release(x1279);
	t4::tensor4f x1281 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1280, ctx.features_denseblock4_denselayer10_conv2_weight); //features.denseblock4.denselayer10.conv2
	t4::release(x1280);
	t4::tensor4f x1282 = t4::Concat<1>(x1275, x1281); //features.denseblock4.denselayer10
	t4::release(x1275, x1281);
	t4::tensor4f x1283 = t4::BatchNormalization(x1282, ctx.features_denseblock4_denselayer11_norm1_weight, ctx.features_denseblock4_denselayer11_norm1_bias, ctx.features_denseblock4_denselayer11_norm1_running_mean, ctx.features_denseblock4_denselayer11_norm1_running_var, 1e-05f); //features.denseblock4.denselayer11.norm1
	t4::tensor4f x1284 = t4::ReluInplace(x1283); //features.denseblock4.denselayer11.relu1
	t4::release(x1283);
	t4::tensor4f x1285 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1284, ctx.features_denseblock4_denselayer11_conv1_weight); //features.denseblock4.denselayer11.conv1
	t4::release(x1284);
	t4::tensor4f x1286 = t4::BatchNormalizationInplace(x1285, ctx.features_denseblock4_denselayer11_norm2_weight, ctx.features_denseblock4_denselayer11_norm2_bias, ctx.features_denseblock4_denselayer11_norm2_running_mean, ctx.features_denseblock4_denselayer11_norm2_running_var, 1e-05f); //features.denseblock4.denselayer11.norm2
	t4::release(x1285);
	t4::tensor4f x1287 = t4::ReluInplace(x1286); //features.denseblock4.denselayer11.relu2
	t4::release(x1286);
	t4::tensor4f x1288 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1287, ctx.features_denseblock4_denselayer11_conv2_weight); //features.denseblock4.denselayer11.conv2
	t4::release(x1287);
	t4::tensor4f x1289 = t4::Concat<1>(x1282, x1288); //features.denseblock4.denselayer11
	t4::release(x1282, x1288);
	t4::tensor4f x1290 = t4::BatchNormalization(x1289, ctx.features_denseblock4_denselayer12_norm1_weight, ctx.features_denseblock4_denselayer12_norm1_bias, ctx.features_denseblock4_denselayer12_norm1_running_mean, ctx.features_denseblock4_denselayer12_norm1_running_var, 1e-05f); //features.denseblock4.denselayer12.norm1
	t4::tensor4f x1291 = t4::ReluInplace(x1290); //features.denseblock4.denselayer12.relu1
	t4::release(x1290);
	t4::tensor4f x1292 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1291, ctx.features_denseblock4_denselayer12_conv1_weight); //features.denseblock4.denselayer12.conv1
	t4::release(x1291);
	t4::tensor4f x1293 = t4::BatchNormalizationInplace(x1292, ctx.features_denseblock4_denselayer12_norm2_weight, ctx.features_denseblock4_denselayer12_norm2_bias, ctx.features_denseblock4_denselayer12_norm2_running_mean, ctx.features_denseblock4_denselayer12_norm2_running_var, 1e-05f); //features.denseblock4.denselayer12.norm2
	t4::release(x1292);
	t4::tensor4f x1294 = t4::ReluInplace(x1293); //features.denseblock4.denselayer12.relu2
	t4::release(x1293);
	t4::tensor4f x1295 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1294, ctx.features_denseblock4_denselayer12_conv2_weight); //features.denseblock4.denselayer12.conv2
	t4::release(x1294);
	t4::tensor4f x1296 = t4::Concat<1>(x1289, x1295); //features.denseblock4.denselayer12
	t4::release(x1289, x1295);
	t4::tensor4f x1297 = t4::BatchNormalization(x1296, ctx.features_denseblock4_denselayer13_norm1_weight, ctx.features_denseblock4_denselayer13_norm1_bias, ctx.features_denseblock4_denselayer13_norm1_running_mean, ctx.features_denseblock4_denselayer13_norm1_running_var, 1e-05f); //features.denseblock4.denselayer13.norm1
	t4::tensor4f x1298 = t4::ReluInplace(x1297); //features.denseblock4.denselayer13.relu1
	t4::release(x1297);
	t4::tensor4f x1299 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1298, ctx.features_denseblock4_denselayer13_conv1_weight); //features.denseblock4.denselayer13.conv1
	t4::release(x1298);
	t4::tensor4f x1300 = t4::BatchNormalizationInplace(x1299, ctx.features_denseblock4_denselayer13_norm2_weight, ctx.features_denseblock4_denselayer13_norm2_bias, ctx.features_denseblock4_denselayer13_norm2_running_mean, ctx.features_denseblock4_denselayer13_norm2_running_var, 1e-05f); //features.denseblock4.denselayer13.norm2
	t4::release(x1299);
	t4::tensor4f x1301 = t4::ReluInplace(x1300); //features.denseblock4.denselayer13.relu2
	t4::release(x1300);
	t4::tensor4f x1302 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1301, ctx.features_denseblock4_denselayer13_conv2_weight); //features.denseblock4.denselayer13.conv2
	t4::release(x1301);
	t4::tensor4f x1303 = t4::Concat<1>(x1296, x1302); //features.denseblock4.denselayer13
	t4::release(x1296, x1302);
	t4::tensor4f x1304 = t4::BatchNormalization(x1303, ctx.features_denseblock4_denselayer14_norm1_weight, ctx.features_denseblock4_denselayer14_norm1_bias, ctx.features_denseblock4_denselayer14_norm1_running_mean, ctx.features_denseblock4_denselayer14_norm1_running_var, 1e-05f); //features.denseblock4.denselayer14.norm1
	t4::tensor4f x1305 = t4::ReluInplace(x1304); //features.denseblock4.denselayer14.relu1
	t4::release(x1304);
	t4::tensor4f x1306 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1305, ctx.features_denseblock4_denselayer14_conv1_weight); //features.denseblock4.denselayer14.conv1
	t4::release(x1305);
	t4::tensor4f x1307 = t4::BatchNormalizationInplace(x1306, ctx.features_denseblock4_denselayer14_norm2_weight, ctx.features_denseblock4_denselayer14_norm2_bias, ctx.features_denseblock4_denselayer14_norm2_running_mean, ctx.features_denseblock4_denselayer14_norm2_running_var, 1e-05f); //features.denseblock4.denselayer14.norm2
	t4::release(x1306);
	t4::tensor4f x1308 = t4::ReluInplace(x1307); //features.denseblock4.denselayer14.relu2
	t4::release(x1307);
	t4::tensor4f x1309 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1308, ctx.features_denseblock4_denselayer14_conv2_weight); //features.denseblock4.denselayer14.conv2
	t4::release(x1308);
	t4::tensor4f x1310 = t4::Concat<1>(x1303, x1309); //features.denseblock4.denselayer14
	t4::release(x1303, x1309);
	t4::tensor4f x1311 = t4::BatchNormalization(x1310, ctx.features_denseblock4_denselayer15_norm1_weight, ctx.features_denseblock4_denselayer15_norm1_bias, ctx.features_denseblock4_denselayer15_norm1_running_mean, ctx.features_denseblock4_denselayer15_norm1_running_var, 1e-05f); //features.denseblock4.denselayer15.norm1
	t4::tensor4f x1312 = t4::ReluInplace(x1311); //features.denseblock4.denselayer15.relu1
	t4::release(x1311);
	t4::tensor4f x1313 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1312, ctx.features_denseblock4_denselayer15_conv1_weight); //features.denseblock4.denselayer15.conv1
	t4::release(x1312);
	t4::tensor4f x1314 = t4::BatchNormalizationInplace(x1313, ctx.features_denseblock4_denselayer15_norm2_weight, ctx.features_denseblock4_denselayer15_norm2_bias, ctx.features_denseblock4_denselayer15_norm2_running_mean, ctx.features_denseblock4_denselayer15_norm2_running_var, 1e-05f); //features.denseblock4.denselayer15.norm2
	t4::release(x1313);
	t4::tensor4f x1315 = t4::ReluInplace(x1314); //features.denseblock4.denselayer15.relu2
	t4::release(x1314);
	t4::tensor4f x1316 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1315, ctx.features_denseblock4_denselayer15_conv2_weight); //features.denseblock4.denselayer15.conv2
	t4::release(x1315);
	t4::tensor4f x1317 = t4::Concat<1>(x1310, x1316); //features.denseblock4.denselayer15
	t4::release(x1310, x1316);
	t4::tensor4f x1318 = t4::BatchNormalization(x1317, ctx.features_denseblock4_denselayer16_norm1_weight, ctx.features_denseblock4_denselayer16_norm1_bias, ctx.features_denseblock4_denselayer16_norm1_running_mean, ctx.features_denseblock4_denselayer16_norm1_running_var, 1e-05f); //features.denseblock4.denselayer16.norm1
	t4::tensor4f x1319 = t4::ReluInplace(x1318); //features.denseblock4.denselayer16.relu1
	t4::release(x1318);
	t4::tensor4f x1320 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1319, ctx.features_denseblock4_denselayer16_conv1_weight); //features.denseblock4.denselayer16.conv1
	t4::release(x1319);
	t4::tensor4f x1321 = t4::BatchNormalizationInplace(x1320, ctx.features_denseblock4_denselayer16_norm2_weight, ctx.features_denseblock4_denselayer16_norm2_bias, ctx.features_denseblock4_denselayer16_norm2_running_mean, ctx.features_denseblock4_denselayer16_norm2_running_var, 1e-05f); //features.denseblock4.denselayer16.norm2
	t4::release(x1320);
	t4::tensor4f x1322 = t4::ReluInplace(x1321); //features.denseblock4.denselayer16.relu2
	t4::release(x1321);
	t4::tensor4f x1323 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1322, ctx.features_denseblock4_denselayer16_conv2_weight); //features.denseblock4.denselayer16.conv2
	t4::release(x1322);
	t4::tensor4f x1324 = t4::Concat<1>(x1317, x1323); //features.denseblock4.denselayer16
	t4::release(x1317, x1323);
	t4::tensor4f x1325 = t4::BatchNormalization(x1324, ctx.features_denseblock4_denselayer17_norm1_weight, ctx.features_denseblock4_denselayer17_norm1_bias, ctx.features_denseblock4_denselayer17_norm1_running_mean, ctx.features_denseblock4_denselayer17_norm1_running_var, 1e-05f); //features.denseblock4.denselayer17.norm1
	t4::tensor4f x1326 = t4::ReluInplace(x1325); //features.denseblock4.denselayer17.relu1
	t4::release(x1325);
	t4::tensor4f x1327 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1326, ctx.features_denseblock4_denselayer17_conv1_weight); //features.denseblock4.denselayer17.conv1
	t4::release(x1326);
	t4::tensor4f x1328 = t4::BatchNormalizationInplace(x1327, ctx.features_denseblock4_denselayer17_norm2_weight, ctx.features_denseblock4_denselayer17_norm2_bias, ctx.features_denseblock4_denselayer17_norm2_running_mean, ctx.features_denseblock4_denselayer17_norm2_running_var, 1e-05f); //features.denseblock4.denselayer17.norm2
	t4::release(x1327);
	t4::tensor4f x1329 = t4::ReluInplace(x1328); //features.denseblock4.denselayer17.relu2
	t4::release(x1328);
	t4::tensor4f x1330 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1329, ctx.features_denseblock4_denselayer17_conv2_weight); //features.denseblock4.denselayer17.conv2
	t4::release(x1329);
	t4::tensor4f x1331 = t4::Concat<1>(x1324, x1330); //features.denseblock4.denselayer17
	t4::release(x1324, x1330);
	t4::tensor4f x1332 = t4::BatchNormalization(x1331, ctx.features_denseblock4_denselayer18_norm1_weight, ctx.features_denseblock4_denselayer18_norm1_bias, ctx.features_denseblock4_denselayer18_norm1_running_mean, ctx.features_denseblock4_denselayer18_norm1_running_var, 1e-05f); //features.denseblock4.denselayer18.norm1
	t4::tensor4f x1333 = t4::ReluInplace(x1332); //features.denseblock4.denselayer18.relu1
	t4::release(x1332);
	t4::tensor4f x1334 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1333, ctx.features_denseblock4_denselayer18_conv1_weight); //features.denseblock4.denselayer18.conv1
	t4::release(x1333);
	t4::tensor4f x1335 = t4::BatchNormalizationInplace(x1334, ctx.features_denseblock4_denselayer18_norm2_weight, ctx.features_denseblock4_denselayer18_norm2_bias, ctx.features_denseblock4_denselayer18_norm2_running_mean, ctx.features_denseblock4_denselayer18_norm2_running_var, 1e-05f); //features.denseblock4.denselayer18.norm2
	t4::release(x1334);
	t4::tensor4f x1336 = t4::ReluInplace(x1335); //features.denseblock4.denselayer18.relu2
	t4::release(x1335);
	t4::tensor4f x1337 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1336, ctx.features_denseblock4_denselayer18_conv2_weight); //features.denseblock4.denselayer18.conv2
	t4::release(x1336);
	t4::tensor4f x1338 = t4::Concat<1>(x1331, x1337); //features.denseblock4.denselayer18
	t4::release(x1331, x1337);
	t4::tensor4f x1339 = t4::BatchNormalization(x1338, ctx.features_denseblock4_denselayer19_norm1_weight, ctx.features_denseblock4_denselayer19_norm1_bias, ctx.features_denseblock4_denselayer19_norm1_running_mean, ctx.features_denseblock4_denselayer19_norm1_running_var, 1e-05f); //features.denseblock4.denselayer19.norm1
	t4::tensor4f x1340 = t4::ReluInplace(x1339); //features.denseblock4.denselayer19.relu1
	t4::release(x1339);
	t4::tensor4f x1341 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1340, ctx.features_denseblock4_denselayer19_conv1_weight); //features.denseblock4.denselayer19.conv1
	t4::release(x1340);
	t4::tensor4f x1342 = t4::BatchNormalizationInplace(x1341, ctx.features_denseblock4_denselayer19_norm2_weight, ctx.features_denseblock4_denselayer19_norm2_bias, ctx.features_denseblock4_denselayer19_norm2_running_mean, ctx.features_denseblock4_denselayer19_norm2_running_var, 1e-05f); //features.denseblock4.denselayer19.norm2
	t4::release(x1341);
	t4::tensor4f x1343 = t4::ReluInplace(x1342); //features.denseblock4.denselayer19.relu2
	t4::release(x1342);
	t4::tensor4f x1344 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1343, ctx.features_denseblock4_denselayer19_conv2_weight); //features.denseblock4.denselayer19.conv2
	t4::release(x1343);
	t4::tensor4f x1345 = t4::Concat<1>(x1338, x1344); //features.denseblock4.denselayer19
	t4::release(x1338, x1344);
	t4::tensor4f x1346 = t4::BatchNormalization(x1345, ctx.features_denseblock4_denselayer20_norm1_weight, ctx.features_denseblock4_denselayer20_norm1_bias, ctx.features_denseblock4_denselayer20_norm1_running_mean, ctx.features_denseblock4_denselayer20_norm1_running_var, 1e-05f); //features.denseblock4.denselayer20.norm1
	t4::tensor4f x1347 = t4::ReluInplace(x1346); //features.denseblock4.denselayer20.relu1
	t4::release(x1346);
	t4::tensor4f x1348 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1347, ctx.features_denseblock4_denselayer20_conv1_weight); //features.denseblock4.denselayer20.conv1
	t4::release(x1347);
	t4::tensor4f x1349 = t4::BatchNormalizationInplace(x1348, ctx.features_denseblock4_denselayer20_norm2_weight, ctx.features_denseblock4_denselayer20_norm2_bias, ctx.features_denseblock4_denselayer20_norm2_running_mean, ctx.features_denseblock4_denselayer20_norm2_running_var, 1e-05f); //features.denseblock4.denselayer20.norm2
	t4::release(x1348);
	t4::tensor4f x1350 = t4::ReluInplace(x1349); //features.denseblock4.denselayer20.relu2
	t4::release(x1349);
	t4::tensor4f x1351 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1350, ctx.features_denseblock4_denselayer20_conv2_weight); //features.denseblock4.denselayer20.conv2
	t4::release(x1350);
	t4::tensor4f x1352 = t4::Concat<1>(x1345, x1351); //features.denseblock4.denselayer20
	t4::release(x1345, x1351);
	t4::tensor4f x1353 = t4::BatchNormalization(x1352, ctx.features_denseblock4_denselayer21_norm1_weight, ctx.features_denseblock4_denselayer21_norm1_bias, ctx.features_denseblock4_denselayer21_norm1_running_mean, ctx.features_denseblock4_denselayer21_norm1_running_var, 1e-05f); //features.denseblock4.denselayer21.norm1
	t4::tensor4f x1354 = t4::ReluInplace(x1353); //features.denseblock4.denselayer21.relu1
	t4::release(x1353);
	t4::tensor4f x1355 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1354, ctx.features_denseblock4_denselayer21_conv1_weight); //features.denseblock4.denselayer21.conv1
	t4::release(x1354);
	t4::tensor4f x1356 = t4::BatchNormalizationInplace(x1355, ctx.features_denseblock4_denselayer21_norm2_weight, ctx.features_denseblock4_denselayer21_norm2_bias, ctx.features_denseblock4_denselayer21_norm2_running_mean, ctx.features_denseblock4_denselayer21_norm2_running_var, 1e-05f); //features.denseblock4.denselayer21.norm2
	t4::release(x1355);
	t4::tensor4f x1357 = t4::ReluInplace(x1356); //features.denseblock4.denselayer21.relu2
	t4::release(x1356);
	t4::tensor4f x1358 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1357, ctx.features_denseblock4_denselayer21_conv2_weight); //features.denseblock4.denselayer21.conv2
	t4::release(x1357);
	t4::tensor4f x1359 = t4::Concat<1>(x1352, x1358); //features.denseblock4.denselayer21
	t4::release(x1352, x1358);
	t4::tensor4f x1360 = t4::BatchNormalization(x1359, ctx.features_denseblock4_denselayer22_norm1_weight, ctx.features_denseblock4_denselayer22_norm1_bias, ctx.features_denseblock4_denselayer22_norm1_running_mean, ctx.features_denseblock4_denselayer22_norm1_running_var, 1e-05f); //features.denseblock4.denselayer22.norm1
	t4::tensor4f x1361 = t4::ReluInplace(x1360); //features.denseblock4.denselayer22.relu1
	t4::release(x1360);
	t4::tensor4f x1362 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1361, ctx.features_denseblock4_denselayer22_conv1_weight); //features.denseblock4.denselayer22.conv1
	t4::release(x1361);
	t4::tensor4f x1363 = t4::BatchNormalizationInplace(x1362, ctx.features_denseblock4_denselayer22_norm2_weight, ctx.features_denseblock4_denselayer22_norm2_bias, ctx.features_denseblock4_denselayer22_norm2_running_mean, ctx.features_denseblock4_denselayer22_norm2_running_var, 1e-05f); //features.denseblock4.denselayer22.norm2
	t4::release(x1362);
	t4::tensor4f x1364 = t4::ReluInplace(x1363); //features.denseblock4.denselayer22.relu2
	t4::release(x1363);
	t4::tensor4f x1365 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1364, ctx.features_denseblock4_denselayer22_conv2_weight); //features.denseblock4.denselayer22.conv2
	t4::release(x1364);
	t4::tensor4f x1366 = t4::Concat<1>(x1359, x1365); //features.denseblock4.denselayer22
	t4::release(x1359, x1365);
	t4::tensor4f x1367 = t4::BatchNormalization(x1366, ctx.features_denseblock4_denselayer23_norm1_weight, ctx.features_denseblock4_denselayer23_norm1_bias, ctx.features_denseblock4_denselayer23_norm1_running_mean, ctx.features_denseblock4_denselayer23_norm1_running_var, 1e-05f); //features.denseblock4.denselayer23.norm1
	t4::tensor4f x1368 = t4::ReluInplace(x1367); //features.denseblock4.denselayer23.relu1
	t4::release(x1367);
	t4::tensor4f x1369 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1368, ctx.features_denseblock4_denselayer23_conv1_weight); //features.denseblock4.denselayer23.conv1
	t4::release(x1368);
	t4::tensor4f x1370 = t4::BatchNormalizationInplace(x1369, ctx.features_denseblock4_denselayer23_norm2_weight, ctx.features_denseblock4_denselayer23_norm2_bias, ctx.features_denseblock4_denselayer23_norm2_running_mean, ctx.features_denseblock4_denselayer23_norm2_running_var, 1e-05f); //features.denseblock4.denselayer23.norm2
	t4::release(x1369);
	t4::tensor4f x1371 = t4::ReluInplace(x1370); //features.denseblock4.denselayer23.relu2
	t4::release(x1370);
	t4::tensor4f x1372 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1371, ctx.features_denseblock4_denselayer23_conv2_weight); //features.denseblock4.denselayer23.conv2
	t4::release(x1371);
	t4::tensor4f x1373 = t4::Concat<1>(x1366, x1372); //features.denseblock4.denselayer23
	t4::release(x1366, x1372);
	t4::tensor4f x1374 = t4::BatchNormalization(x1373, ctx.features_denseblock4_denselayer24_norm1_weight, ctx.features_denseblock4_denselayer24_norm1_bias, ctx.features_denseblock4_denselayer24_norm1_running_mean, ctx.features_denseblock4_denselayer24_norm1_running_var, 1e-05f); //features.denseblock4.denselayer24.norm1
	t4::tensor4f x1375 = t4::ReluInplace(x1374); //features.denseblock4.denselayer24.relu1
	t4::release(x1374);
	t4::tensor4f x1376 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1375, ctx.features_denseblock4_denselayer24_conv1_weight); //features.denseblock4.denselayer24.conv1
	t4::release(x1375);
	t4::tensor4f x1377 = t4::BatchNormalizationInplace(x1376, ctx.features_denseblock4_denselayer24_norm2_weight, ctx.features_denseblock4_denselayer24_norm2_bias, ctx.features_denseblock4_denselayer24_norm2_running_mean, ctx.features_denseblock4_denselayer24_norm2_running_var, 1e-05f); //features.denseblock4.denselayer24.norm2
	t4::release(x1376);
	t4::tensor4f x1378 = t4::ReluInplace(x1377); //features.denseblock4.denselayer24.relu2
	t4::release(x1377);
	t4::tensor4f x1379 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1378, ctx.features_denseblock4_denselayer24_conv2_weight); //features.denseblock4.denselayer24.conv2
	t4::release(x1378);
	t4::tensor4f x1380 = t4::Concat<1>(x1373, x1379); //features.denseblock4.denselayer24
	t4::release(x1373, x1379);
	t4::tensor4f x1381 = t4::BatchNormalization(x1380, ctx.features_denseblock4_denselayer25_norm1_weight, ctx.features_denseblock4_denselayer25_norm1_bias, ctx.features_denseblock4_denselayer25_norm1_running_mean, ctx.features_denseblock4_denselayer25_norm1_running_var, 1e-05f); //features.denseblock4.denselayer25.norm1
	t4::tensor4f x1382 = t4::ReluInplace(x1381); //features.denseblock4.denselayer25.relu1
	t4::release(x1381);
	t4::tensor4f x1383 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1382, ctx.features_denseblock4_denselayer25_conv1_weight); //features.denseblock4.denselayer25.conv1
	t4::release(x1382);
	t4::tensor4f x1384 = t4::BatchNormalizationInplace(x1383, ctx.features_denseblock4_denselayer25_norm2_weight, ctx.features_denseblock4_denselayer25_norm2_bias, ctx.features_denseblock4_denselayer25_norm2_running_mean, ctx.features_denseblock4_denselayer25_norm2_running_var, 1e-05f); //features.denseblock4.denselayer25.norm2
	t4::release(x1383);
	t4::tensor4f x1385 = t4::ReluInplace(x1384); //features.denseblock4.denselayer25.relu2
	t4::release(x1384);
	t4::tensor4f x1386 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1385, ctx.features_denseblock4_denselayer25_conv2_weight); //features.denseblock4.denselayer25.conv2
	t4::release(x1385);
	t4::tensor4f x1387 = t4::Concat<1>(x1380, x1386); //features.denseblock4.denselayer25
	t4::release(x1380, x1386);
	t4::tensor4f x1388 = t4::BatchNormalization(x1387, ctx.features_denseblock4_denselayer26_norm1_weight, ctx.features_denseblock4_denselayer26_norm1_bias, ctx.features_denseblock4_denselayer26_norm1_running_mean, ctx.features_denseblock4_denselayer26_norm1_running_var, 1e-05f); //features.denseblock4.denselayer26.norm1
	t4::tensor4f x1389 = t4::ReluInplace(x1388); //features.denseblock4.denselayer26.relu1
	t4::release(x1388);
	t4::tensor4f x1390 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1389, ctx.features_denseblock4_denselayer26_conv1_weight); //features.denseblock4.denselayer26.conv1
	t4::release(x1389);
	t4::tensor4f x1391 = t4::BatchNormalizationInplace(x1390, ctx.features_denseblock4_denselayer26_norm2_weight, ctx.features_denseblock4_denselayer26_norm2_bias, ctx.features_denseblock4_denselayer26_norm2_running_mean, ctx.features_denseblock4_denselayer26_norm2_running_var, 1e-05f); //features.denseblock4.denselayer26.norm2
	t4::release(x1390);
	t4::tensor4f x1392 = t4::ReluInplace(x1391); //features.denseblock4.denselayer26.relu2
	t4::release(x1391);
	t4::tensor4f x1393 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1392, ctx.features_denseblock4_denselayer26_conv2_weight); //features.denseblock4.denselayer26.conv2
	t4::release(x1392);
	t4::tensor4f x1394 = t4::Concat<1>(x1387, x1393); //features.denseblock4.denselayer26
	t4::release(x1387, x1393);
	t4::tensor4f x1395 = t4::BatchNormalization(x1394, ctx.features_denseblock4_denselayer27_norm1_weight, ctx.features_denseblock4_denselayer27_norm1_bias, ctx.features_denseblock4_denselayer27_norm1_running_mean, ctx.features_denseblock4_denselayer27_norm1_running_var, 1e-05f); //features.denseblock4.denselayer27.norm1
	t4::tensor4f x1396 = t4::ReluInplace(x1395); //features.denseblock4.denselayer27.relu1
	t4::release(x1395);
	t4::tensor4f x1397 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1396, ctx.features_denseblock4_denselayer27_conv1_weight); //features.denseblock4.denselayer27.conv1
	t4::release(x1396);
	t4::tensor4f x1398 = t4::BatchNormalizationInplace(x1397, ctx.features_denseblock4_denselayer27_norm2_weight, ctx.features_denseblock4_denselayer27_norm2_bias, ctx.features_denseblock4_denselayer27_norm2_running_mean, ctx.features_denseblock4_denselayer27_norm2_running_var, 1e-05f); //features.denseblock4.denselayer27.norm2
	t4::release(x1397);
	t4::tensor4f x1399 = t4::ReluInplace(x1398); //features.denseblock4.denselayer27.relu2
	t4::release(x1398);
	t4::tensor4f x1400 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1399, ctx.features_denseblock4_denselayer27_conv2_weight); //features.denseblock4.denselayer27.conv2
	t4::release(x1399);
	t4::tensor4f x1401 = t4::Concat<1>(x1394, x1400); //features.denseblock4.denselayer27
	t4::release(x1394, x1400);
	t4::tensor4f x1402 = t4::BatchNormalization(x1401, ctx.features_denseblock4_denselayer28_norm1_weight, ctx.features_denseblock4_denselayer28_norm1_bias, ctx.features_denseblock4_denselayer28_norm1_running_mean, ctx.features_denseblock4_denselayer28_norm1_running_var, 1e-05f); //features.denseblock4.denselayer28.norm1
	t4::tensor4f x1403 = t4::ReluInplace(x1402); //features.denseblock4.denselayer28.relu1
	t4::release(x1402);
	t4::tensor4f x1404 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1403, ctx.features_denseblock4_denselayer28_conv1_weight); //features.denseblock4.denselayer28.conv1
	t4::release(x1403);
	t4::tensor4f x1405 = t4::BatchNormalizationInplace(x1404, ctx.features_denseblock4_denselayer28_norm2_weight, ctx.features_denseblock4_denselayer28_norm2_bias, ctx.features_denseblock4_denselayer28_norm2_running_mean, ctx.features_denseblock4_denselayer28_norm2_running_var, 1e-05f); //features.denseblock4.denselayer28.norm2
	t4::release(x1404);
	t4::tensor4f x1406 = t4::ReluInplace(x1405); //features.denseblock4.denselayer28.relu2
	t4::release(x1405);
	t4::tensor4f x1407 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1406, ctx.features_denseblock4_denselayer28_conv2_weight); //features.denseblock4.denselayer28.conv2
	t4::release(x1406);
	t4::tensor4f x1408 = t4::Concat<1>(x1401, x1407); //features.denseblock4.denselayer28
	t4::release(x1401, x1407);
	t4::tensor4f x1409 = t4::BatchNormalization(x1408, ctx.features_denseblock4_denselayer29_norm1_weight, ctx.features_denseblock4_denselayer29_norm1_bias, ctx.features_denseblock4_denselayer29_norm1_running_mean, ctx.features_denseblock4_denselayer29_norm1_running_var, 1e-05f); //features.denseblock4.denselayer29.norm1
	t4::tensor4f x1410 = t4::ReluInplace(x1409); //features.denseblock4.denselayer29.relu1
	t4::release(x1409);
	t4::tensor4f x1411 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1410, ctx.features_denseblock4_denselayer29_conv1_weight); //features.denseblock4.denselayer29.conv1
	t4::release(x1410);
	t4::tensor4f x1412 = t4::BatchNormalizationInplace(x1411, ctx.features_denseblock4_denselayer29_norm2_weight, ctx.features_denseblock4_denselayer29_norm2_bias, ctx.features_denseblock4_denselayer29_norm2_running_mean, ctx.features_denseblock4_denselayer29_norm2_running_var, 1e-05f); //features.denseblock4.denselayer29.norm2
	t4::release(x1411);
	t4::tensor4f x1413 = t4::ReluInplace(x1412); //features.denseblock4.denselayer29.relu2
	t4::release(x1412);
	t4::tensor4f x1414 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1413, ctx.features_denseblock4_denselayer29_conv2_weight); //features.denseblock4.denselayer29.conv2
	t4::release(x1413);
	t4::tensor4f x1415 = t4::Concat<1>(x1408, x1414); //features.denseblock4.denselayer29
	t4::release(x1408, x1414);
	t4::tensor4f x1416 = t4::BatchNormalization(x1415, ctx.features_denseblock4_denselayer30_norm1_weight, ctx.features_denseblock4_denselayer30_norm1_bias, ctx.features_denseblock4_denselayer30_norm1_running_mean, ctx.features_denseblock4_denselayer30_norm1_running_var, 1e-05f); //features.denseblock4.denselayer30.norm1
	t4::tensor4f x1417 = t4::ReluInplace(x1416); //features.denseblock4.denselayer30.relu1
	t4::release(x1416);
	t4::tensor4f x1418 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1417, ctx.features_denseblock4_denselayer30_conv1_weight); //features.denseblock4.denselayer30.conv1
	t4::release(x1417);
	t4::tensor4f x1419 = t4::BatchNormalizationInplace(x1418, ctx.features_denseblock4_denselayer30_norm2_weight, ctx.features_denseblock4_denselayer30_norm2_bias, ctx.features_denseblock4_denselayer30_norm2_running_mean, ctx.features_denseblock4_denselayer30_norm2_running_var, 1e-05f); //features.denseblock4.denselayer30.norm2
	t4::release(x1418);
	t4::tensor4f x1420 = t4::ReluInplace(x1419); //features.denseblock4.denselayer30.relu2
	t4::release(x1419);
	t4::tensor4f x1421 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1420, ctx.features_denseblock4_denselayer30_conv2_weight); //features.denseblock4.denselayer30.conv2
	t4::release(x1420);
	t4::tensor4f x1422 = t4::Concat<1>(x1415, x1421); //features.denseblock4.denselayer30
	t4::release(x1415, x1421);
	t4::tensor4f x1423 = t4::BatchNormalization(x1422, ctx.features_denseblock4_denselayer31_norm1_weight, ctx.features_denseblock4_denselayer31_norm1_bias, ctx.features_denseblock4_denselayer31_norm1_running_mean, ctx.features_denseblock4_denselayer31_norm1_running_var, 1e-05f); //features.denseblock4.denselayer31.norm1
	t4::tensor4f x1424 = t4::ReluInplace(x1423); //features.denseblock4.denselayer31.relu1
	t4::release(x1423);
	t4::tensor4f x1425 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1424, ctx.features_denseblock4_denselayer31_conv1_weight); //features.denseblock4.denselayer31.conv1
	t4::release(x1424);
	t4::tensor4f x1426 = t4::BatchNormalizationInplace(x1425, ctx.features_denseblock4_denselayer31_norm2_weight, ctx.features_denseblock4_denselayer31_norm2_bias, ctx.features_denseblock4_denselayer31_norm2_running_mean, ctx.features_denseblock4_denselayer31_norm2_running_var, 1e-05f); //features.denseblock4.denselayer31.norm2
	t4::release(x1425);
	t4::tensor4f x1427 = t4::ReluInplace(x1426); //features.denseblock4.denselayer31.relu2
	t4::release(x1426);
	t4::tensor4f x1428 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1427, ctx.features_denseblock4_denselayer31_conv2_weight); //features.denseblock4.denselayer31.conv2
	t4::release(x1427);
	t4::tensor4f x1429 = t4::Concat<1>(x1422, x1428); //features.denseblock4.denselayer31
	t4::release(x1422, x1428);
	t4::tensor4f x1430 = t4::BatchNormalization(x1429, ctx.features_denseblock4_denselayer32_norm1_weight, ctx.features_denseblock4_denselayer32_norm1_bias, ctx.features_denseblock4_denselayer32_norm1_running_mean, ctx.features_denseblock4_denselayer32_norm1_running_var, 1e-05f); //features.denseblock4.denselayer32.norm1
	t4::tensor4f x1431 = t4::ReluInplace(x1430); //features.denseblock4.denselayer32.relu1
	t4::release(x1430);
	t4::tensor4f x1432 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1431, ctx.features_denseblock4_denselayer32_conv1_weight); //features.denseblock4.denselayer32.conv1
	t4::release(x1431);
	t4::tensor4f x1433 = t4::BatchNormalizationInplace(x1432, ctx.features_denseblock4_denselayer32_norm2_weight, ctx.features_denseblock4_denselayer32_norm2_bias, ctx.features_denseblock4_denselayer32_norm2_running_mean, ctx.features_denseblock4_denselayer32_norm2_running_var, 1e-05f); //features.denseblock4.denselayer32.norm2
	t4::release(x1432);
	t4::tensor4f x1434 = t4::ReluInplace(x1433); //features.denseblock4.denselayer32.relu2
	t4::release(x1433);
	t4::tensor4f x1435 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1434, ctx.features_denseblock4_denselayer32_conv2_weight); //features.denseblock4.denselayer32.conv2
	t4::release(x1434);
	t4::tensor4f x1436 = t4::Concat<1>(x1429, x1435); //features.denseblock4.denselayer32
	t4::release(x1429, x1435);
	t4::tensor4f x1437 = t4::BatchNormalizationInplace(x1436, ctx.features_norm5_weight, ctx.features_norm5_bias, ctx.features_norm5_running_mean, ctx.features_norm5_running_var, 1e-05f); //features.norm5
	t4::release(x1436);
	t4::tensor4f x1438 = t4::ReluInplace(x1437);
	t4::release(x1437);
	t4::tensor4f x1439 = t4::AveragePool2d<7, 7, 1, 1, 0, 0>(x1438);
	t4::release(x1438);
	t4::tensor2f x1440 = t4::Flatten<1>(x1439);
	t4::release(x1439);
	t4::tensor2f x1441 = t4::Linear(x1440, ctx.classifier_weight, ctx.classifier_bias); //classifier
	t4::release(x1440);
	return x1441;
}
