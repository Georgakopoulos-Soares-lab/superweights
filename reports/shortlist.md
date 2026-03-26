# Channel shortlist
Input: `results/full_v1/channel_correlations.parquet`

## Decision rules (encoded)
- Keep top `20` channels by `auc_label` (best over feature types).

## Why these filters
- Primary: `auc_label` (causal sensitivity, pos vs neg).
- Secondary: `corr_beta` (biological meaning / effect-size alignment).

## Summary
- Total unique channels: `4928`
- Shortlisted channels: `20`

## Top channels (preview)
| layer | channel | auc_label | feature_auc | spearman_beta | abs_spearman_beta | feature_corr_beta | spearman_delta | abs_spearman_delta | feature_corr_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_conv1.conv_layer | 1492 | 0.6854474544525146 | abs_delta_l2 | -0.27089107036590576 | 0.27089107036590576 | delta_mean | -0.24568507075309753 | 0.24568507075309753 | delta_mean |
| horizontal_conv1.conv_layer | 241 | 0.6674637794494629 | abs_delta_l2 | 0.21982072293758392 | 0.21982072293758392 | delta_l2 | 0.19271160662174225 | 0.19271160662174225 | delta_mean |
| horizontal_conv1.conv_layer | 587 | 0.6653084754943848 | abs_delta_l2 | -0.293348103761673 | 0.293348103761673 | delta_mean | -0.25990965962409973 | 0.25990965962409973 | delta_mean |
| horizontal_conv1.conv_layer | 130 | 0.6639217138290405 | abs_delta_l2 | 0.2880801260471344 | 0.2880801260471344 | delta_mean | 0.3278964161872864 | 0.3278964161872864 | delta_mean |
| horizontal_conv1.conv_layer | 858 | 0.6550193428993225 | abs_delta_l2 | -0.19954371452331543 | 0.19954371452331543 | delta_mean | 0.1550297886133194 | 0.1550297886133194 | delta_l2 |
| horizontal_conv1.conv_layer | 1500 | 0.6515304446220398 | abs_delta_l2 | -0.20686623454093933 | 0.20686623454093933 | delta_mean | -0.2329583764076233 | 0.2329583764076233 | delta_mean |
| horizontal_conv1.conv_layer | 1256 | 0.6514937281608582 | abs_delta_l2 | 0.24538731575012207 | 0.24538731575012207 | delta_mean | 0.19795642793178558 | 0.19795642793178558 | delta_mean |
| horizontal_conv1.conv_layer | 631 | 0.6509166359901428 | abs_delta_l2 | 0.26877129077911377 | 0.26877129077911377 | delta_l2 | -0.27599263191223145 | 0.27599263191223145 | delta_mean |
| horizontal_conv1.conv_layer | 979 | 0.6481494307518005 | abs_delta_l2 | -0.24658232927322388 | 0.24658232927322388 | delta_mean | -0.17230439186096191 | 0.17230439186096191 | delta_mean |
| horizontal_conv1.conv_layer | 1468 | 0.6471306085586548 | abs_delta_l2 | 0.19546177983283997 | 0.19546177983283997 | delta_l2 | 0.13235807418823242 | 0.13235807418823242 | delta_l2 |
| horizontal_conv1.conv_layer | 18 | 0.6460086107254028 | abs_delta_l2 | 0.2554683983325958 | 0.2554683983325958 | delta_l2 | -0.2546721398830414 | 0.2546721398830414 | delta_mean |
| horizontal_conv1.conv_layer | 1134 | 0.6441017985343933 | abs_delta_l2 | 0.24572888016700745 | 0.24572888016700745 | delta_l2 | 0.15520401298999786 | 0.15520401298999786 | delta_l2 |
| horizontal_conv1.conv_layer | 1456 | 0.6431055068969727 | abs_delta_l2 | 0.11988086998462677 | 0.11988086998462677 | delta_l2 | 0.12459519505500793 | 0.12459519505500793 | delta_l2 |
| horizontal_conv1.conv_layer | 839 | 0.6413019299507141 | abs_delta_l2 | 0.24830427765846252 | 0.24830427765846252 | delta_mean | 0.24223515391349792 | 0.24223515391349792 | delta_mean |
| horizontal_conv1.conv_layer | 1049 | 0.6394910216331482 | abs_delta_l2 | -0.18785463273525238 | 0.18785463273525238 | delta_mean | -0.1646866351366043 | 0.1646866351366043 | delta_mean |
| horizontal_conv1.conv_layer | 466 | 0.6394737362861633 | abs_delta_l2 | 0.16916491091251373 | 0.16916491091251373 | delta_l2 | -0.13808700442314148 | 0.13808700442314148 | delta_mean |
| horizontal_conv1.conv_layer | 824 | 0.6388315558433533 | abs_delta_l2 | 0.18268045783042908 | 0.18268045783042908 | delta_l2 | 0.12428624927997589 | 0.12428624927997589 | delta_l2 |
| horizontal_conv1.conv_layer | 321 | 0.6387748122215271 | abs_delta_l2 | -0.23375974595546722 | 0.23375974595546722 | delta_mean | -0.1909639686346054 | 0.1909639686346054 | delta_mean |
| horizontal_conv1.conv_layer | 580 | 0.6376442909240723 | abs_delta_l2 | 0.20196975767612457 | 0.20196975767612457 | delta_l2 | 0.21238920092582703 | 0.21238920092582703 | delta_mean |
| horizontal_conv1.conv_layer | 30 | 0.6375716924667358 | abs_delta_l2 | 0.2577250003814697 | 0.2577250003814697 | delta_l2 | 0.23819339275360107 | 0.23819339275360107 | delta_l2 |
